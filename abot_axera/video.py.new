"""Video frame sources: decode a video and hand out model-ready frames.

Each source yields (chw, rgb) per sampled frame:
  chw : float32 [3,280,504]  the model input (width-lock resize, centre crop / pad)
  rgb : uint8   [280,504,3]  the same frame for colouring the point cloud

Backends
  ax  : pyaxvideo (ax-video-sdk) — hardware decode (VDEC) + colour conversion / resize (IVPS)
        on the AXCL card or the AX650 chip. Frames stay on the device; only the small image
        crosses PCIe.
        The decoder runs in a separate PROCESS: pyaxvideo's read()/close() can block inside the
        C layer and never return (its timeout_ms is not honoured), and a wedged thread cannot be
        killed — a child process can. If no frame arrives in time the child is killed; a stall
        before the first frame falls back to cv2, a stall mid-video fails the job.
  cv2 : OpenCV software decode + exact preprocessing on the host (fallback, any machine)

Environment
  ABOT_DECODER          auto (default, ax with cv2 fallback) | ax | cv2
  ABOT_AX_RESIZE        ivps2x (default) IVPS to 2x the target width, exact bicubic on the host
                        ivps  straight to the target size | host  full-res frame to the host
  ABOT_AX_RANGE         tv (default) expand 16..235 -> 0..255 after IVPS' full-range conversion
  ABOT_AX_FMT           format name passed to pyaxvideo convert() ("bgr" returns RGB in 0.1.1)
  ABOT_AX_TIMEOUT       seconds to wait for the first frame (default 40)
  ABOT_AX_FRAME_TIMEOUT seconds to wait for each further frame (default 30)
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading

import numpy as np

from .preprocess import preprocess_image, resize_aa, DEFAULT_PAD_RGB

TARGET_W, TARGET_H = 504, 280


class DecoderUnavailable(RuntimeError):
    """The hardware decoder produced nothing usable; the caller may fall back to cv2."""


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
                    chw, _ = preprocess_image(np.ascontiguousarray(bgr[:, :, ::-1]))
                    yield _finish(chw)
                i += 1
        finally:
            cap.release()


# ----------------------------------------------------------------------------- pyaxvideo (child process)
# The decoder runs as `python -m abot_axera.video <json>` and streams finished frames down a pipe:
#   b"F" + chw float32 [3,280,504] + rgb uint8 [280,504,3]   one per sampled frame
#   b"E" end of stream          b"X" fatal error (details on stderr)
# A wedged pyaxvideo call inside the child cannot be interrupted, but the child can be killed —
# which is exactly what the parent does when frames stop arriving.
_CHW_SHAPE, _RGB_SHAPE = (3, TARGET_H, TARGET_W), (TARGET_H, TARGET_W, 3)
_CHW_BYTES, _RGB_BYTES = 3 * TARGET_H * TARGET_W * 4, TARGET_H * TARGET_W * 3


def _child_decode(cfg: dict) -> int:
    """Child entry point: hardware-decode `cfg['path']` and write frames to stdout."""
    import pyaxvideo as axv
    out = sys.stdout.buffer
    range_fix, resize, fmt = cfg["range"], cfg["resize"], cfg["fmt"]
    interval, cw, ch, src_h, src_w = cfg["interval"], cfg["cw"], cfg["ch"], cfg["h"], cfg["w"]

    def fix(u8):
        if range_fix != "tv":
            return u8
        x = (u8.astype(np.float32) - 16.0) * (255.0 / 219.0)
        return np.clip(x + 0.5, 0, 255).astype(np.uint8)

    try:
        axv.init(device=cfg["device_id"])
        with axv.VideoReader(cfg["path"]) as r:
            for i, f in enumerate(r):
                if i % interval:
                    continue
                item = fix(f.convert(fmt, cw, ch).to_numpy())
                if resize == "host":
                    chw, _ = preprocess_image(item); chw, rgb = _finish(chw)
                elif resize == "ivps2x":
                    chw, rgb = _finish(resize_aa(_from_hwc_u8(item), max(1, round(src_h * TARGET_W / src_w)), TARGET_W))
                else:
                    chw, rgb = _finish(_from_hwc_u8(item))
                out.write(b"F"); out.write(chw.tobytes()); out.write(rgb.tobytes()); out.flush()
        out.write(b"E"); out.flush()
        os._exit(0)                                   # skip interpreter shutdown: the SDK hangs there
    except BaseException as e:                        # noqa: BLE001
        sys.stderr.write(f"[video-child] {type(e).__name__}: {e}\n"); sys.stderr.flush()
        try:
            out.write(b"X"); out.flush()
        except Exception:
            pass
        os._exit(1)


_axv_lock = threading.Lock()
_axv_state = {"ok": None}
_no_hw: set[str] = set()      # videos the hardware decoder could not handle in this process


def ax_available(device_id: int = 0) -> bool:
    """True if pyaxvideo is importable on this host (the child process does the real init)."""
    with _axv_lock:
        if _axv_state["ok"] is None:
            try:
                import pyaxvideo  # noqa: F401
                _axv_state["ok"] = True
            except Exception as e:
                print(f"[video] pyaxvideo unavailable ({type(e).__name__}: {str(e)[:70]}); using cv2", flush=True)
                _axv_state["ok"] = False
        return _axv_state["ok"]


def _read_exact(stream, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class AxVideoSource:
    """Hardware decode in a child process; iterates (chw, rgb) for every interval-th frame."""

    name = "ax"

    def __init__(self, path: str, fps: int, device_id: int = 0, resize: str | None = None, prefetch: int = 4):
        self.path, self.device_id, self.prefetch = os.path.abspath(path), device_id, prefetch
        self.resize = (resize or os.environ.get("ABOT_AX_RESIZE", "ivps2x")).lower()
        self.fmt = os.environ.get("ABOT_AX_FMT", "bgr")            # 0.1.1 swaps rgb/bgr
        self.range_fix = os.environ.get("ABOT_AX_RANGE", "tv").lower()
        self.first_timeout = float(os.environ.get("ABOT_AX_TIMEOUT", "40"))
        self.frame_timeout = float(os.environ.get("ABOT_AX_FRAME_TIMEOUT", "30"))
        w, h, src_fps, n = probe(path)
        self.w, self.h = w, h
        self.interval = sample_interval(src_fps, fps)
        self.total = (n + self.interval - 1) // self.interval if n else 0

    def _plan(self):
        """(convert_w, convert_h) for the chosen mode; IVPS wants even sizes."""
        if self.resize == "host":
            return self.w, self.h
        k = 2 if self.resize == "ivps2x" else 1
        cw = TARGET_W * k
        ch = max(2, round(self.h * cw / max(self.w, 1)))
        ch += ch % 2
        return cw, ch

    def _spawn(self):
        cw, ch = self._plan()
        cfg = {"path": self.path, "device_id": self.device_id, "interval": self.interval,
               "cw": cw, "ch": ch, "fmt": self.fmt, "range": self.range_fix,
               "resize": self.resize, "w": self.w, "h": self.h}
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return subprocess.Popen([sys.executable, "-m", "abot_axera.video", json.dumps(cfg)],
                                stdout=subprocess.PIPE, stderr=None, env=env, bufsize=0)

    def __iter__(self):
        proc = self._spawn()
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def pump():
            try:
                while not stop.is_set():
                    tag = proc.stdout.read(1)
                    if not tag or tag == b"E":
                        q.put(None); return
                    if tag == b"X":
                        q.put(RuntimeError("decoder reported an error (see log)")); return
                    a = _read_exact(proc.stdout, _CHW_BYTES); b = _read_exact(proc.stdout, _RGB_BYTES)
                    if a is None or b is None:
                        q.put(None); return
                    q.put((np.frombuffer(a, np.float32).reshape(_CHW_SHAPE),
                           np.frombuffer(b, np.uint8).reshape(_RGB_SHAPE)))
            except Exception as e:
                q.put(RuntimeError(f"decoder pipe: {e}"))

        th = threading.Thread(target=pump, name="abot-decode-pipe", daemon=True)
        th.start()
        n = 0
        try:
            while True:
                timeout = self.first_timeout if n == 0 else self.frame_timeout
                try:
                    item = q.get(timeout=timeout)
                except queue.Empty:
                    alive = proc.poll() is None
                    msg = (f"hardware decoder {'stalled' if alive else 'exited'}: "
                           f"no frame for {timeout:.0f}s after {n}")
                    raise (DecoderUnavailable(msg) if n == 0 else RuntimeError(msg)) from None
                if item is None:
                    if n == 0:
                        rc = proc.poll()
                        raise DecoderUnavailable(f"hardware decoder produced no frames (exit {rc})")
                    break
                if isinstance(item, BaseException):
                    msg = f"hardware decoder failed: {item}"
                    raise (DecoderUnavailable(msg) if n == 0 else RuntimeError(msg))
                n += 1
                yield item
        finally:
            stop.set()
            if proc.poll() is None:                   # a wedged VDEC only dies with its process
                proc.kill()
            try:
                while True:
                    q.get_nowait()                    # unblock the pump thread so it can exit
            except queue.Empty:
                pass
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.wait(timeout=5)
            th.join(timeout=5)


class AutoSource:
    """Hardware decode when it works, OpenCV when it does not (transparent to the caller)."""

    def __init__(self, path: str, fps: int, device_id: int = 0):
        self.cv = Cv2Source(path, fps)
        self.path = os.path.abspath(path)
        usable = ax_available(device_id) and self.path not in _no_hw
        self.ax = AxVideoSource(path, fps, device_id) if usable else None
        self.interval, self.total = self.cv.interval, self.cv.total
        self.name = "ax" if self.ax is not None else "cv2"

    def __iter__(self):
        if self.ax is not None:
            n = 0
            try:
                for item in self.ax:
                    n += 1
                    yield item
                return
            except DecoderUnavailable as e:
                if n:
                    raise
                _no_hw.add(self.path)          # don't pay the timeout again for this video
                print(f"[video] {e}; falling back to cv2", flush=True)
                self.name = "cv2"
        yield from self.cv


def open_video(path: str, fps: int, device_id: int = 0, decoder: str | None = None):
    """Frame source per ABOT_DECODER (auto | ax | cv2)."""
    mode = (decoder or os.environ.get("ABOT_DECODER", "auto")).lower()
    if mode == "cv2":
        return Cv2Source(path, fps)
    if mode == "ax":
        if not ax_available(device_id):
            raise RuntimeError("ABOT_DECODER=ax but pyaxvideo is not importable")
        return AxVideoSource(path, fps, device_id)
    return AutoSource(path, fps, device_id)


if __name__ == "__main__":                            # child process entry point
    sys.exit(_child_decode(json.loads(sys.argv[1])))
