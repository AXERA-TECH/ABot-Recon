"""Video frame sources: decode a video and hand out model-ready frames.

Each source yields (chw, rgb) per sampled frame:
  chw : float32 [3,280,504]  the model input (width-lock resize, centre crop / pad)
  rgb : uint8   [280,504,3]  the same frame for colouring the point cloud

Backends
  ax  : pyaxvideo (ax-video-sdk) — hardware decode (VDEC) + colour conversion / resize (IVPS)
        on the AXCL card or the AX650 chip. Frames stay on the device; only the small RGB
        image crosses PCIe. Runs in its own thread so decoding overlaps NPU inference.
        ABOT_AX_RESIZE:
          ivps2x (default) IVPS to 2x the target width, then the exact antialiased bicubic on
                           the host for the last 2x (closest to the reference preprocessing)
          ivps             IVPS straight to the target size (smallest transfer)
          host             full-resolution RGB to the host, exact preprocessing there
  cv2 : OpenCV software decode + exact preprocessing on the host (fallback, any machine)

ABOT_DECODER = auto (ax if pyaxvideo is importable and initialises, else cv2) | ax | cv2
"""
from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np

from .preprocess import preprocess_image, resize_aa, DEFAULT_PAD_RGB

TARGET_W, TARGET_H = 504, 280


def probe(path: str):
    """(width, height, fps, frame_count) via OpenCV container parsing (no decode)."""
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(cap.get(cv2.CAP_PROP_FPS) or 30.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    finally:
        cap.release()


def sample_interval(src_fps: float, fps: int) -> int:
    return max(1, round(src_fps / max(fps, 1)))


def _crop_pad_chw(chw: np.ndarray) -> np.ndarray:
    """float32 [3,h,504] -> [3,280,504]: centre crop or ImageNet-mean pad (same rule as preprocess)."""
    h = chw.shape[1]
    if h > TARGET_H:
        top = round((h - TARGET_H) * 0.5); chw = chw[:, top:top + TARGET_H]
    elif h < TARGET_H:
        canvas = np.empty((3, TARGET_H, TARGET_W), np.float32)
        canvas[:] = np.asarray(DEFAULT_PAD_RGB, np.float32)[:, None, None]
        t = (TARGET_H - h) // 2; canvas[:, t:t + h] = chw; chw = canvas
    return np.ascontiguousarray(chw)


def _finish(chw: np.ndarray):
    """model input + uint8 colour image from a [3,h,504] float frame."""
    chw = _crop_pad_chw(chw)
    rgb = np.round(np.clip(chw, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
    return chw, np.ascontiguousarray(rgb)


def _from_hwc_u8(rgb_hwc: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb_hwc.transpose(2, 0, 1)).astype(np.float32) / np.float32(255)


# ----------------------------------------------------------------------------- OpenCV
class Cv2Source:
    name = "cv2"

    def __init__(self, path: str, fps: int):
        self.path = path
        w, h, src_fps, n = probe(path)
        self.interval = sample_interval(src_fps, fps)
        self.total = (n + self.interval - 1) // self.interval if n else 0

    def __iter__(self):
        import cv2
        cap = cv2.VideoCapture(self.path)
        try:
            i = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i % self.interval == 0:
                    chw, _ = preprocess_image(np.ascontiguousarray(bgr[:, :, ::-1]))   # exact host preprocessing
                    yield _finish(chw)
                i += 1
        finally:
            cap.release()


# ----------------------------------------------------------------------------- pyaxvideo
_axv_lock = threading.Lock()
_axv_ready = False


def ax_available(device_id: int) -> bool:
    """True if pyaxvideo imports and initialises on this host (cached)."""
    global _axv_ready
    if _axv_ready:
        return True
    try:
        import pyaxvideo as axv
        with _axv_lock:
            axv.init(device=device_id)
        _axv_ready = True
        return True
    except Exception as e:
        print(f"[video] pyaxvideo unavailable ({type(e).__name__}: {str(e)[:80]}); using cv2", flush=True)
        return False


class AxVideoSource:
    """Hardware decode in a background thread; iterates (chw, rgb) for every interval-th frame."""

    name = "ax"

    def __init__(self, path: str, fps: int, device_id: int = 0, resize: str | None = None, prefetch: int = 8):
        import pyaxvideo as axv
        self.axv, self.path = axv, path
        self.resize = (resize or os.environ.get("ABOT_AX_RESIZE", "ivps2x")).lower()
        if not ax_available(device_id):
            raise RuntimeError("pyaxvideo not available")
        w, h, src_fps, n = probe(path)
        self.w, self.h = w, h
        self.interval = sample_interval(src_fps, fps)
        self.total = (n + self.interval - 1) // self.interval if n else 0
        self.prefetch = prefetch
        # the hardware decoder handles H.264 / H.265 only: open once now so an unsupported
        # codec (e.g. MPEG-4 part 2 from OpenCV's default writer) can fall back to cv2
        with axv.VideoReader(path) as r:
            self.info = r.info

    def _plan(self):
        """(convert_w, convert_h) for the chosen mode; IVPS wants even sizes."""
        if self.resize == "host":
            return self.w, self.h
        k = 2 if self.resize == "ivps2x" else 1
        cw = TARGET_W * k
        ch = max(2, round(self.h * cw / max(self.w, 1)))
        ch += ch % 2
        return cw, ch

    # pyaxvideo 0.1.1 hands back the channels swapped: convert("rgb") is BGR in memory and
    # convert("bgr") is RGB (checked pixel-for-pixel against OpenCV on the same frame).
    # ABOT_AX_FMT overrides the format string if a future release fixes the naming.
    AX_FMT = os.environ.get("ABOT_AX_FMT", "bgr")
    # IVPS converts YUV->RGB with a full-range (pc) BT.601 matrix. Camera/phone videos are almost
    # always limited (tv) range, so their blacks come out lifted; ABOT_AX_RANGE=tv (default) expands
    # 16..235 -> 0..255 on the host, ABOT_AX_RANGE=pc leaves the IVPS output as is.
    AX_RANGE = os.environ.get("ABOT_AX_RANGE", "tv").lower()

    def _range_fix(self, rgb_u8: np.ndarray) -> np.ndarray:
        if self.AX_RANGE != "tv":
            return rgb_u8
        x = (rgb_u8.astype(np.float32) - 16.0) * (255.0 / 219.0)
        return np.clip(x + 0.5, 0, 255).astype(np.uint8)

    def _worker(self, q: queue.Queue, cw, ch):
        try:
            with self.axv.VideoReader(self.path) as r:
                for i, f in enumerate(r):
                    if i % self.interval == 0:
                        q.put(f.convert(self.AX_FMT, cw, ch).to_numpy())
            q.put(None)
        except Exception as e:
            q.put(e)

    def __iter__(self):
        cw, ch = self._plan()
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        th = threading.Thread(target=self._worker, args=(q, cw, ch), daemon=True)
        th.start()
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise RuntimeError(f"pyaxvideo decode failed: {item}")
            item = self._range_fix(item)
            if self.resize == "host":                          # full-res RGB, exact host preprocessing
                chw, _ = preprocess_image(item); yield _finish(chw)
            elif self.resize == "ivps2x":                     # IVPS did 2x of the shrink, host the rest
                yield _finish(resize_aa(_from_hwc_u8(item), max(1, round(self.h * TARGET_W / self.w)), TARGET_W))
            else:                                              # IVPS straight to 504 wide
                yield _finish(_from_hwc_u8(item))
        th.join()


def open_video(path: str, fps: int, device_id: int = 0, decoder: str | None = None):
    """Pick the frame source per ABOT_DECODER (auto | ax | cv2)."""
    mode = (decoder or os.environ.get("ABOT_DECODER", "auto")).lower()
    if mode in ("auto", "ax") and ax_available(device_id):
        try:
            return AxVideoSource(path, fps, device_id)
        except Exception as e:
            if mode == "ax":
                raise
            print(f"[video] hardware decoder cannot open {os.path.basename(path)} ({str(e)[:80]}); using cv2", flush=True)
    elif mode == "ax":
        raise RuntimeError("ABOT_DECODER=ax but pyaxvideo is not available")
    return Cv2Source(path, fps)
