"""Device-resident chain runner for ABot-Recon (encoder -> decoder_step -> heads).

Why this exists
---------------
pyaxengine's ``InferenceSession.run`` copies *every* input host->device and
*every* output device->host on each call.  For decoder_step that is the two
588 MB fp32 KV tensors in and out again (~2.35 GB over PCIe per frame), which
costs ~6.5 s of the 9.5 s/frame we measured, while the NPU itself only needs
~2.6 s.  This runner keeps everything that never has to be seen by the host on
the device:

* past_key/past_value <-> present_key/present_value : two device buffer pairs
  used ping-pong; each frame the decoder's *present* outputs are rebound as the
  next frame's *past* inputs (pointer swap, zero copy).
* patch_tokens (encoder out -> decoder in) and fused_hidden (decoder out ->
  heads in) are single shared device buffers.
* Only the 1.7 MB image goes in and the ~3.8 MB of heads outputs come out.

Two device backends behind one tiny interface, both built on the cffi
bindings shipped inside pyaxengine (we only borrow ``_axclrt_capi`` /
``_axe_capi``; the session/run logic here is our own):

* ``_AxclDevice`` - PCIe AXCL card (libaxcl_rt, e.g. dell 8x AX650N).
* ``_AxeDevice``  - on-chip AX650 (libax_engine + libax_sys CMM memory).

Select with ``device="axcl" | "ax650" | "auto"`` (auto = whichever runtime
library is present on this host).
"""
from __future__ import annotations

import ctypes.util
import os
import time

import numpy as np

__all__ = ["NativeChainRunner", "detect_device"]

MODEL_NAMES = ("encoder", "decoder_step", "heads")


def detect_device() -> str:
    """'axcl' if libaxcl_rt is on this host, 'ax650' if libax_engine is, else error."""
    if ctypes.util.find_library("axcl_rt"):
        return "axcl"
    if ctypes.util.find_library("ax_engine"):
        return "ax650"
    raise RuntimeError("neither libaxcl_rt (AXCL card) nor libax_engine (on-chip AX650) found")


# --------------------------------------------------------------------------- #
# Device backends. Interface (all sizes in bytes):
#   load(path) -> model            names(model) -> (in_names, out_names)
#   in_size(model, i) / out_size(model, i)   in_fp32(model,i)/out_fp32(model,i)
#   malloc(size) -> buf            free(buf)          memset0(buf)
#   h2d(buf, np_array)             d2h(np_array, buf)
#   bind_in(model, i, buf)         bind_out(model, i, buf)
#   run(model)                     unload(model)      activate()
# --------------------------------------------------------------------------- #
class _Buf:
    __slots__ = ("ptr", "phy", "vir", "size")

    def __init__(self, size: int, ptr=None, phy=0, vir=None):
        self.size = int(size)
        self.ptr = ptr  # axcl: device address (cffi void*)
        self.phy = phy  # ax650: CMM physical address
        self.vir = vir  # ax650: mapped virtual address (cffi void*)


class _AxclDevice:
    """PCIe AXCL card through libaxcl_rt (cffi decls borrowed from pyaxengine)."""

    name = "axcl"

    def __init__(self, device_id: int):
        # importing axengine._axclrt performs axclInit() once + registers atexit finalize
        import axengine._axclrt as _rt  # noqa: F401  (side effect: axclInit)
        from axengine._axclrt_capi import axclrt_cffi as ffi, axclrt_lib as lib

        self.ffi, self.lib = ffi, lib
        try:  # not declared by pyaxengine; ABI-mode cffi lets us add it lazily
            ffi.cdef("axclError axclrtMemset(void *devPtr, uint8_t value, size_t count);")
            self._has_memset = True
        except Exception:  # already declared / cdef refused -> fall back to H2D zeros
            self._has_memset = hasattr(lib, "axclrtMemset")

        lst = ffi.new("axclrtDeviceList *")
        ret = lib.axclrtGetDeviceList(lst)
        if ret != 0 or lst.num == 0:
            raise RuntimeError(f"axclrtGetDeviceList failed 0x{ret & 0xffffffff:08x}, {lst.num} devices")
        if device_id >= lst.num:
            raise RuntimeError(f"AXCL device index {device_id} out of range (total {lst.num})")
        self.device_id = lst.devices[device_id]
        _chk(lib.axclrtSetDevice(self.device_id), "axclrtSetDevice")
        ret = lib.axclrtEngineInit(ffi.cast("axclrtEngineVNpuKind", 0))  # DISABLE; may already be up
        if ret != 0:
            kind = ffi.new("axclrtEngineVNpuKind *")
            if lib.axclrtEngineGetVNpuKind(kind) != 0:
                raise RuntimeError(f"axclrtEngineInit failed 0x{ret & 0xffffffff:08x}")
        # remember the thread context; service runs jobs on worker threads (like pyaxengine)
        self._ctx = ffi.new("axclrtContext *")
        _chk(lib.axclrtGetCurrentContext(self._ctx), "axclrtGetCurrentContext")

    def activate(self):
        _chk(self.lib.axclrtSetCurrentContext(self._ctx[0]), "axclrtSetCurrentContext")

    # -- models --
    def load(self, path: str):
        ffi, lib = self.ffi, self.lib
        mid = ffi.new("uint64_t *")
        _chk(lib.axclrtEngineLoadFromFile(ffi.new("char[]", path.encode()), mid), f"load {path}")
        cid = ffi.new("uint64_t *")
        _chk(lib.axclrtEngineCreateContext(mid[0], cid), "axclrtEngineCreateContext")
        info = ffi.new("axclrtEngineIOInfo *")
        _chk(lib.axclrtEngineGetIOInfo(mid[0], info), "axclrtEngineGetIOInfo")
        io = ffi.new("axclrtEngineIO *")
        _chk(lib.axclrtEngineCreateIO(info[0], io), "axclrtEngineCreateIO")
        return {"mid": mid, "cid": cid, "info": info, "io": io, "path": path}

    def unload(self, m):
        lib = self.lib
        if m.get("io") is not None:
            lib.axclrtEngineDestroyIO(m["io"][0]); m["io"] = None
        if m.get("mid") is not None and m["mid"][0]:
            lib.axclrtEngineUnload(m["mid"][0]); m["mid"][0] = 0

    def names(self, m):
        ffi, lib, info = self.ffi, self.lib, m["info"][0]
        ins = [ffi.string(lib.axclrtEngineGetInputNameByIndex(info, i)).decode()
               for i in range(lib.axclrtEngineGetNumInputs(info))]
        outs = [ffi.string(lib.axclrtEngineGetOutputNameByIndex(info, i)).decode()
                for i in range(lib.axclrtEngineGetNumOutputs(info))]
        return ins, outs

    def in_size(self, m, i): return int(self.lib.axclrtEngineGetInputSizeByIndex(m["info"][0], 0, i))
    def out_size(self, m, i): return int(self.lib.axclrtEngineGetOutputSizeByIndex(m["info"][0], 0, i))

    def in_fp32(self, m, i):
        t = self.ffi.new("axclrtEngineDataType *")
        self.lib.axclrtEngineGetInputDataType(m["info"][0], i, t)
        return int(t[0]) == 15  # AXCL_DATA_TYPE_FP32

    def out_fp32(self, m, i):
        t = self.ffi.new("axclrtEngineDataType *")
        self.lib.axclrtEngineGetOutputDataType(m["info"][0], i, t)
        return int(t[0]) == 15

    # -- memory --
    def malloc(self, size: int) -> _Buf:
        p = self.ffi.new("void **")
        _chk(self.lib.axclrtMalloc(p, size, self.lib.AXCL_MEM_MALLOC_NORMAL_ONLY), f"axclrtMalloc({size})")
        return _Buf(size, ptr=p[0])

    def free(self, b: _Buf):
        if b.ptr is not None:
            self.lib.axclrtFree(b.ptr); b.ptr = None

    def memset0(self, b: _Buf):
        if self._has_memset:
            try:
                _chk(self.lib.axclrtMemset(b.ptr, 0, b.size), "axclrtMemset")
                return
            except Exception:
                self._has_memset = False
        z = np.zeros(b.size, np.uint8)
        self.h2d(b, z)

    def h2d(self, b: _Buf, a: np.ndarray):
        assert a.flags.c_contiguous and a.nbytes <= b.size, (a.shape, a.nbytes, b.size)
        src = self.ffi.cast("void *", a.ctypes.data)
        _chk(self.lib.axclrtMemcpy(b.ptr, src, a.nbytes, self.lib.AXCL_MEMCPY_HOST_TO_DEVICE), "H2D")

    def d2h(self, a: np.ndarray, b: _Buf):
        assert a.flags.c_contiguous and a.nbytes <= b.size, (a.shape, a.nbytes, b.size)
        dst = self.ffi.cast("void *", a.ctypes.data)
        _chk(self.lib.axclrtMemcpy(dst, b.ptr, a.nbytes, self.lib.AXCL_MEMCPY_DEVICE_TO_HOST), "D2H")

    # -- io binding / run --
    def bind_in(self, m, i, b: _Buf):
        _chk(self.lib.axclrtEngineSetInputBufferByIndex(m["io"][0], i, b.ptr, b.size), f"bind_in {i}")

    def bind_out(self, m, i, b: _Buf):
        _chk(self.lib.axclrtEngineSetOutputBufferByIndex(m["io"][0], i, b.ptr, b.size), f"bind_out {i}")

    def run(self, m):
        _chk(self.lib.axclrtEngineExecute(m["mid"][0], m["cid"][0], 0, m["io"][0]),
             f"axclrtEngineExecute({os.path.basename(m['path'])})")


class _AxeDevice:
    """On-chip AX650 through libax_engine / libax_sys (cffi decls borrowed from pyaxengine).

    Memory is CMM (physically contiguous, CPU-cached mapping).  Buffers that the
    host never touches (KV ping-pong, patch_tokens, fused_hidden) need no cache
    maintenance; host writes are flushed, host reads invalidated first."""

    name = "ax650"

    def __init__(self, device_id: int = 0):
        import axengine._axe as _axe  # noqa: F401  (side effect: AX_SYS_Init + AX_ENGINE_Init)
        from axengine._axe_capi import engine_cffi as ffi, engine_lib as elib, sys_lib as slib

        self.ffi, self.elib, self.slib = ffi, elib, slib
        self.device_id = 0
        self._align = 128
        self._token = ffi.new("AX_S8[]", b"AbotNative")

    def activate(self):  # single device, nothing to select
        return None

    def load(self, path: str):
        import mmap

        ffi, elib = self.ffi, self.elib
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        buf = ffi.from_buffer("char[]", mm)
        handle = ffi.new("uint64_t **")
        extra = ffi.new("AX_ENGINE_HANDLE_EXTRA_T *")
        nm = ffi.new("char[]", os.path.splitext(os.path.basename(path))[0].encode())
        extra.pName = nm
        _chk(elib.AX_ENGINE_CreateHandleV2(handle, buf, len(mm), extra), f"CreateHandleV2 {path}")
        ctx = ffi.new("uint64_t **")
        _chk(elib.AX_ENGINE_CreateContextV2(handle[0], ctx), "AX_ENGINE_CreateContextV2")
        info = ffi.new("AX_ENGINE_IO_INFO_T **")
        _chk(elib.AX_ENGINE_GetIOInfo(handle[0], info), "AX_ENGINE_GetIOInfo")
        n_in, n_out = info[0].nInputSize, info[0].nOutputSize
        io = ffi.new("AX_ENGINE_IO_T *")
        pin = ffi.new(f"AX_ENGINE_IO_BUFFER_T[{n_in}]")
        pout = ffi.new(f"AX_ENGINE_IO_BUFFER_T[{n_out}]")
        io[0].pInputs, io[0].nInputSize = pin, n_in
        io[0].pOutputs, io[0].nOutputSize = pout, n_out
        # the model blob must stay mapped for the handle's lifetime
        return {"handle": handle, "ctx": ctx, "info": info, "io": io, "_keep": (f, mm, buf, nm, extra, pin, pout),
                "path": path}

    def unload(self, m):
        if m.get("handle") is not None and m["handle"][0] != self.ffi.NULL:
            self.elib.AX_ENGINE_DestroyHandle(m["handle"][0]); m["handle"][0] = self.ffi.NULL
        keep = m.pop("_keep", None)
        if keep:
            try:
                keep[1].close(); keep[0].close()
            except Exception:
                pass

    def names(self, m):
        ffi, info = self.ffi, m["info"][0]
        ins = [ffi.string(info.pInputs[i].pName).decode() for i in range(info.nInputSize)]
        outs = [ffi.string(info.pOutputs[i].pName).decode() for i in range(info.nOutputSize)]
        return ins, outs

    def in_size(self, m, i): return int(m["info"][0].pInputs[i].nSize)
    def out_size(self, m, i): return int(m["info"][0].pOutputs[i].nSize)
    def in_fp32(self, m, i): return int(m["info"][0].pInputs[i].eDataType) == 3   # AX_ENGINE_DT_FLOAT32
    def out_fp32(self, m, i): return int(m["info"][0].pOutputs[i].eDataType) == 3

    def malloc(self, size: int) -> _Buf:
        phy = self.ffi.new("AX_U64 *"); vir = self.ffi.new("AX_VOID **")
        _chk(self.slib.AX_SYS_MemAllocCached(phy, vir, size, self._align, self._token), f"AX_SYS_MemAllocCached({size})")
        return _Buf(size, phy=int(phy[0]), vir=vir[0])

    def free(self, b: _Buf):
        if b.vir is not None:
            self.slib.AX_SYS_MemFree(b.phy, b.vir); b.vir = None

    def memset0(self, b: _Buf):
        # cffi has no memset; zero through a numpy view over the CMM mapping, then flush
        np.frombuffer(self.ffi.buffer(b.vir, b.size), dtype=np.uint8)[:] = 0
        self.slib.AX_SYS_MflushCache(b.phy, b.vir, b.size)

    def h2d(self, b: _Buf, a: np.ndarray):
        assert a.flags.c_contiguous and a.nbytes <= b.size, (a.shape, a.nbytes, b.size)
        self.ffi.memmove(b.vir, self.ffi.cast("void *", a.ctypes.data), a.nbytes)
        self.slib.AX_SYS_MflushCache(b.phy, b.vir, b.size)

    def d2h(self, a: np.ndarray, b: _Buf):
        assert a.flags.c_contiguous and a.nbytes <= b.size, (a.shape, a.nbytes, b.size)
        self.slib.AX_SYS_MinvalidateCache(b.phy, b.vir, b.size)
        self.ffi.memmove(self.ffi.cast("void *", a.ctypes.data), b.vir, a.nbytes)

    def bind_in(self, m, i, b: _Buf):
        e = m["io"][0].pInputs[i]; e.phyAddr, e.pVirAddr, e.nSize = b.phy, b.vir, b.size

    def bind_out(self, m, i, b: _Buf):
        e = m["io"][0].pOutputs[i]; e.phyAddr, e.pVirAddr, e.nSize = b.phy, b.vir, b.size

    def run(self, m):
        _chk(self.elib.AX_ENGINE_RunSyncV2(m["handle"][0], m["ctx"][0], m["io"]),
             f"AX_ENGINE_RunSyncV2({os.path.basename(m['path'])})")


def _chk(ret: int, what: str):
    if ret != 0:
        raise RuntimeError(f"{what} failed 0x{ret & 0xffffffff:08x}")


# --------------------------------------------------------------------------- #
class NativeChainRunner:
    """encoder -> decoder_step -> heads with device-resident KV cache.

    Public API (used by abot_axera.backend.NpuReleasedModel):
        reset()                                   zero the KV cache (start of a sequence)
        step(image[1,3,H,W] f32, frame_index) ->  {"camera_features","local_points","confidence"} (host)
        close()
    Also exposes run_encoder/run_decoder/run_heads-free ``provider`` string for logs.
    """

    def __init__(self, model_dir: str, device_id: int = 6, suffix: str = "_kitti02", device: str = "auto",
                 verbose: bool = True):
        dev = detect_device() if device in (None, "", "auto") else device.lower()
        if dev == "axcl":
            self.dev = _AxclDevice(device_id)
        elif dev in ("ax650", "axe", "onchip"):
            self.dev = _AxeDevice(device_id)
        else:
            raise ValueError(f"unknown device {device!r} (axcl | ax650 | auto)")
        self.provider = f"{self.dev.name}-native"
        self.device_id = self.dev.device_id
        self._log = print if verbose else (lambda *a, **k: None)

        t0 = time.time()
        self.m = {n: self.dev.load(os.path.join(model_dir, f"{n}{suffix}.axmodel")) for n in MODEL_NAMES}
        self._log(f"[native] {self.provider}: 3 models loaded in {time.time() - t0:.1f}s", flush=True)

        self.idx = {n: self._index(self.m[n]) for n in MODEL_NAMES}
        for n in MODEL_NAMES:
            ins, outs = self.idx[n]
            for i in range(len(ins)):
                assert self.dev.in_fp32(self.m[n], i), f"{n} input {i} is not fp32"
            for i in range(len(outs)):
                assert self.dev.out_fp32(self.m[n], i), f"{n} output {i} is not fp32"
        enc, dec, hd = self.m["encoder"], self.m["decoder_step"], self.m["heads"]
        ei, eo = self.idx["encoder"]; di, do = self.idx["decoder_step"]; hi, ho = self.idx["heads"]

        # --- device buffers ---
        self._bufs: list[_Buf] = []
        M = self._malloc
        self.b_image = M(self.dev.in_size(enc, ei["image"]))
        self.b_patch = M(max(self.dev.out_size(enc, eo["patch_tokens"]), self.dev.in_size(dec, di["patch_tokens"])))
        kv_k = max(self.dev.in_size(dec, di["past_key"]), self.dev.out_size(dec, do["present_key"]))
        kv_v = max(self.dev.in_size(dec, di["past_value"]), self.dev.out_size(dec, do["present_value"]))
        vd = max(self.dev.in_size(dec, di["past_valid"]), self.dev.out_size(dec, do["present_valid"]))
        self.kv = [(M(kv_k), M(kv_v), M(vd)), (M(kv_k), M(kv_v), M(vd))]  # ping-pong pairs
        self.b_frame = M(self.dev.in_size(dec, di["frame_index"]))
        self.b_fused = M(max(self.dev.out_size(dec, do["fused_hidden"]), self.dev.in_size(hd, hi["fused_hidden"])))
        self.b_heads = {k: M(self.dev.out_size(hd, ho[k])) for k in ("local_points", "camera_features", "confidence")}
        self._log(f"[native] device buffers: {sum(b.size for b in self._bufs) / 1e6:.0f} MB "
                  f"(KV ping-pong {2 * (kv_k + kv_v) / 1e6:.0f} MB)", flush=True)

        # --- static bindings ---
        self.dev.bind_in(enc, ei["image"], self.b_image)
        self.dev.bind_out(enc, eo["patch_tokens"], self.b_patch)
        self.dev.bind_in(dec, di["patch_tokens"], self.b_patch)
        self.dev.bind_in(dec, di["frame_index"], self.b_frame)
        self.dev.bind_out(dec, do["fused_hidden"], self.b_fused)
        self.dev.bind_in(hd, hi["fused_hidden"], self.b_fused)
        for k, b in self.b_heads.items():
            self.dev.bind_out(hd, ho[k], b)

        # host-side output shapes (from the model meta, via pyaxengine-free query)
        self.out_shapes = {"local_points": (1, 280, 504, 3), "camera_features": (1, 725, 512), "confidence": (1, 280, 504, 1)}
        self._cur = 0
        self._frame_host = np.zeros((1,), np.float32)
        self.reset()

    # -- helpers --
    def _malloc(self, size: int) -> _Buf:
        b = self.dev.malloc(size); self._bufs.append(b); return b

    def _index(self, m):
        ins, outs = self.dev.names(m)
        return {n: i for i, n in enumerate(ins)}, {n: i for i, n in enumerate(outs)}

    # -- public --
    def reset(self) -> None:
        """Zero the *current* past KV + past_valid (== torch: fresh empty cache)."""
        self.dev.activate()
        k, v, vd = self.kv[self._cur]
        for b in (k, v, vd):
            self.dev.memset0(b)

    def step(self, image: np.ndarray, frame_index: int) -> dict[str, np.ndarray]:
        dev = self.dev
        dev.activate()
        enc, dec, hd = self.m["encoder"], self.m["decoder_step"], self.m["heads"]
        di, do = self.idx["decoder_step"]

        img = np.ascontiguousarray(image, dtype=np.float32)
        dev.h2d(self.b_image, img)
        dev.run(enc)

        past, pres = self.kv[self._cur], self.kv[1 - self._cur]
        dev.bind_in(dec, di["past_key"], past[0]); dev.bind_in(dec, di["past_value"], past[1]); dev.bind_in(dec, di["past_valid"], past[2])
        dev.bind_out(dec, do["present_key"], pres[0]); dev.bind_out(dec, do["present_value"], pres[1]); dev.bind_out(dec, do["present_valid"], pres[2])
        self._frame_host[0] = float(frame_index)
        dev.h2d(self.b_frame, self._frame_host)
        dev.run(dec)
        self._cur = 1 - self._cur  # present becomes next past

        dev.run(hd)
        out = {}
        for k, shp in self.out_shapes.items():
            a = np.empty(shp, np.float32)
            dev.d2h(a, self.b_heads[k])
            out[k] = a
        return out

    def read_kv(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Debug: copy the current past KV (i.e. last present) back to host."""
        self.dev.activate()
        k, v, vd = self.kv[self._cur]
        K = np.empty((1, 18, 16, 11, 725, 64), np.float32); V = np.empty_like(K); D = np.empty((1,), np.float32)
        self.dev.d2h(K, k); self.dev.d2h(V, v); self.dev.d2h(D, vd)
        return K, V, D

    def close(self) -> None:
        if getattr(self, "dev", None) is None:
            return
        try:
            self.dev.activate()
        except Exception:
            pass
        for b in getattr(self, "_bufs", []):
            try:
                self.dev.free(b)
            except Exception:
                pass
        self._bufs = []
        for n in list(getattr(self, "m", {}).keys()):
            try:
                self.dev.unload(self.m[n])
            except Exception:
                pass
        self.m = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
