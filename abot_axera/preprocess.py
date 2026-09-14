"""Frame preprocessing: width-lock antialiased bicubic resize, then center crop or ImageNet-mean
pad to 280x504 — numerically identical to what the published torch pipeline feeds the model
(torchvision ``resize(..., BICUBIC, antialias=True)`` on float tensors, overshoots kept).

Two resize implementations, same math:
  * C (csrc/resize_aa.c, loaded through cffi): an expression-for-expression copy of PyTorch's
    `_upsample_bicubic2d_aa` CPU kernel, OpenMP over lines, reads the PIL uint8 buffer directly
    (uint8 -> float/255 folded into the first pass). Built on first use with gcc (-O3 -mavx2
    -mfma -fopenmp on x86, -O3 -fopenmp -DABOT_RESIZE_FAST on aarch64) into abot_axera/_build/;
    bit-identical to torchvision on x86 AVX2 hosts (~4 ms per 1080p frame incl. uint8 conversion);
    on aarch64 a blocked SIMD layout that differs by <= 1 ulp (torch is no reference there).
  * prebuilt/ : shipped .so for hosts without a compiler (aarch64 boards); same source.
  * numpy fallback (nothing else works): exact float32 tap weights applied with BLAS matmuls;
    differs from torchvision only by fp32 accumulation order (~1e-6), ~0.1 s per 1080p frame on x86.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

DEFAULT_PAD_RGB = (0.485, 0.456, 0.406)
f32, f64 = np.float32, np.float64


@dataclass(frozen=True)
class FovTransform:
    source_height: int
    source_width: int
    resized_height: int
    target_height: int
    target_width: int
    crop_top: int = 0
    crop_bottom: int = 0
    pad_top: int = 0
    pad_bottom: int = 0


# ----------------------------------------------------------------------------- C path
_HERE = Path(__file__).resolve().parent
_CSRC = _HERE / "csrc" / "resize_aa.c"
_BUILD = _HERE / "_build"
_PREBUILT = _HERE / "prebuilt"
_lib = None
_lib_tried = False


def _cflags() -> list[str]:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return ["-O3", "-mavx2", "-mfma", "-fopenmp"]          # reference layout: bit-exact vs torchvision
    return ["-O3", "-fopenmp", "-DABOT_RESIZE_FAST"]           # aarch64: blocked SIMD layout (<= 1 ulp)


def _load_c():
    """cffi handle to resize_aa_f32, building the shared object if needed; None if unavailable."""
    global _lib, _lib_tried
    if _lib is not None or _lib_tried:
        return _lib
    _lib_tried = True
    if os.environ.get("ABOT_RESIZE_NUMPY") == "1" or not _CSRC.exists():
        return None
    try:
        from cffi import FFI

        name = f"libresize_aa_{platform.machine()}_{sys.platform}.so"
        so = _BUILD / name
        if not so.exists() or so.stat().st_mtime < _CSRC.stat().st_mtime:
            try:
                _BUILD.mkdir(exist_ok=True)
                cc = os.environ.get("CC", "gcc")
                subprocess.run([cc, "-shared", "-fPIC", *_cflags(), "-o", str(so), str(_CSRC), "-lm"],
                               check=True, capture_output=True)
            except Exception as e:          # no compiler: fall back to a shipped binary
                if (_PREBUILT / name).exists():
                    so = _PREBUILT / name
                    print(f"[preprocess] gcc unavailable ({type(e).__name__}); using prebuilt {so.name}", flush=True)
                else:
                    raise
        ffi = FFI()
        ffi.cdef("int resize_aa_f32(const float *in, int64_t C, int64_t H, int64_t W, int64_t OH, int64_t OW, float *out);\n"
                 "int resize_aa_u8(const uint8_t *in, int64_t H, int64_t W, int64_t OH, int64_t OW, float *out);")
        lib = ffi.dlopen(str(so))
        _lib = (ffi, lib)
    except Exception as e:  # no gcc / no cffi -> numpy fallback
        print(f"[preprocess] C resize unavailable ({type(e).__name__}: {str(e)[:80]}); using numpy", flush=True)
        _lib = None
    return _lib


def _resize_c(chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    ffi, lib = _lib
    x = np.ascontiguousarray(chw, f32)
    out = np.empty((x.shape[0], out_h, out_w), f32)
    rc = lib.resize_aa_f32(ffi.cast("const float *", x.ctypes.data), x.shape[0], x.shape[1], x.shape[2],
                           out_h, out_w, ffi.cast("float *", out.ctypes.data))
    if rc != 0:
        raise RuntimeError("resize_aa_f32 failed")
    return out


def _resize_c_u8(hwc: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """uint8 [H,W,3] -> float32 [3,out_h,out_w] == resize(to_tensor(img)); conversion done in C."""
    ffi, lib = _lib
    x = np.ascontiguousarray(hwc, np.uint8)
    out = np.empty((3, out_h, out_w), f32)
    rc = lib.resize_aa_u8(ffi.cast("const uint8_t *", x.ctypes.data), x.shape[0], x.shape[1],
                          out_h, out_w, ffi.cast("float *", out.ctypes.data))
    if rc != 0:
        raise RuntimeError("resize_aa_u8 failed")
    return out


# ----------------------------------------------------------------------------- numpy fallback
def _fma(a, b, c):
    return (f64(a) * f64(b) + f64(c)).astype(f32)


def _aa_filter(x: np.ndarray) -> np.ndarray:
    """aten HelperInterpCubic::aa_filter<float> (a=-0.5), with the multiply-adds fused like -mfma."""
    x = np.abs(x.astype(f32)); A = f32(-0.5)
    c1 = _fma(_fma(A + f32(2), x, -(A + f32(3))) * x, x, f32(1))
    c2 = _fma(_fma(_fma(A, x, -f32(5) * A), x, f32(8) * A), x, -f32(4) * A)
    return np.where(x < f32(1), c1, np.where(x < f32(2), c2, f32(0))).astype(f32)


def _aa_taps(n_in: int, n_out: int, interp: int = 4):
    """(idx [out,K] int64, w [out,K] float32) exactly as aten _compute_indices_min_size_weights_aa<float>."""
    scale = f32(f32(n_in) / f32(n_out))
    support = f32(f64(interp * 0.5) * f64(scale)) if scale >= 1.0 else f32(interp * 0.5)
    invscale = f32(1.0 / f64(scale)) if scale >= 1.0 else f32(1.0)
    K = int(np.ceil(float(support))) * 2 + 1
    idx = np.zeros((n_out, K), np.int64); w = np.zeros((n_out, K), f32)
    for i in range(n_out):
        center = f32(f64(scale) * (i + 0.5))
        xmin = max(int(f64(f32(center - support)) + 0.5), 0)
        xsize = min(int(f64(f32(center + support)) + 0.5), n_in) - xmin
        j = np.arange(xsize)
        ww = _aa_filter(f32((f64(f32(f32(j + xmin) - center)) + 0.5) * f64(invscale)))
        tot = f32(0.0)
        for v in ww:
            tot = f32(tot + v)
        if tot != 0:
            ww = (ww / tot).astype(f32)
        idx[i, :xsize] = xmin + j; idx[i, xsize:] = xmin; w[i, :xsize] = ww
    return idx, w


_MATS: dict[tuple[int, int], np.ndarray] = {}


def _aa_matrix(n_in: int, n_out: int) -> np.ndarray:
    """[n_out, n_in] float32 resampling matrix built from the exact aten tap weights."""
    idx, w = _aa_taps(n_in, n_out)
    M = np.zeros((n_out, n_in), f32)
    for i in range(n_out):
        np.add.at(M[i], idx[i], w[i])
    return M


def _resize_np(chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    _, h, w = chw.shape
    for key in ((w, out_w), (h, out_h)):
        if key not in _MATS:
            _MATS[key] = _aa_matrix(*key)
    x = np.ascontiguousarray(chw, f32) @ _MATS[(w, out_w)].T          # W pass  [3,h,out_w]
    return np.ascontiguousarray(np.einsum("oh,chw->cow", _MATS[(h, out_h)], x, optimize=True))


def resize_aa(chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """[3,H,W] float32 -> [3,out_h,out_w], antialiased bicubic (torchvision semantics)."""
    if _load_c() is not None:
        return _resize_c(chw, out_h, out_w)
    return _resize_np(chw, out_h, out_w)


# ----------------------------------------------------------------------------- public
def to_chw01(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"expected HWC RGB, got {arr.shape}")
    x = arr.astype(f32)
    if x.size and x.max() > 1.5:
        x = x / f32(255.0)
    return np.ascontiguousarray(np.clip(x, 0.0, 1.0).transpose(2, 0, 1))


def preprocess_image(image, *, height: int = 280, width: int = 504, pad_rgb=DEFAULT_PAD_RGB):
    """-> (float32 [3,height,width], FovTransform). Width-lock resize; crop or pad vertically."""
    u8 = None
    if isinstance(image, Image.Image):
        u8 = np.asarray(image.convert("RGB"))
    elif isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1] == 3:
        u8 = image
    if u8 is not None and _load_c() is not None:      # fast path: uint8 straight into the C kernel
        sh, sw = u8.shape[:2]
        rh = max(1, round(sh * width / max(sw, 1)))
        x = _resize_c_u8(u8, rh, width)
    else:
        x = to_chw01(image)
        _, sh, sw = x.shape
        rh = max(1, round(sh * width / max(sw, 1)))
        x = resize_aa(x, rh, width)
    crop_top = crop_bottom = pad_top = pad_bottom = 0
    if rh > height:
        crop_top = round((rh - height) * 0.5)
        crop_bottom = rh - height - crop_top
        x = x[:, crop_top:crop_top + height]
    elif rh < height:
        pad_top = (height - rh) // 2
        pad_bottom = height - rh - pad_top
        canvas = np.empty((3, height, width), f32)
        canvas[:] = np.asarray(pad_rgb, f32)[:, None, None]
        canvas[:, pad_top:pad_top + rh] = x
        x = canvas
    tf = FovTransform(sh, sw, rh, height, width, crop_top, crop_bottom, pad_top, pad_bottom)
    return np.ascontiguousarray(x), tf


def iter_preprocessed(paths: Iterable[str | Path], *, height: int = 280, width: int = 504):
    ref = None
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            x, tf = preprocess_image(im, height=height, width=width)
        if ref is None:
            ref = tf
        elif tf != ref:
            raise ValueError(f"inconsistent frame geometry at index {i}: {tf} != {ref}")
        yield x, tf
