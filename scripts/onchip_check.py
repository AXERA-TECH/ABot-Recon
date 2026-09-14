"""On-chip AX650 check of the native runner's _AxeDevice: run encoder + heads (fits in ~1 GB CMM)
on tensors an AXCL card produced and compare.

  on the card host:  python scripts/onchip_check.py --make-ref REF.npz          (native runner, one frame)
  on the board:      python3 scripts/onchip_check.py MODEL_DIR REF.npz          (python3 + numpy + cffi + pyaxengine)

The board needs no torch: only abot_axera/native_runner.py is imported. decoder_step (2.6 GB + 2.35 GB KV)
needs a board with >= 6 GB CMM, so it is not exercised here — same code path, bigger buffers."""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_ref(path):
    """Card side: dump image / patch_tokens / fused_hidden / heads outputs of frame 0 (testdata/frames12)."""
    import glob
    from abot_recon.preprocessing import iter_preprocessed
    from abot_axera.native_runner import NativeChainRunner
    p = sorted(glob.glob(os.path.join(ROOT, "testdata", "frames12", "*.jpg")))[:1]
    img = [np.ascontiguousarray(t.unsqueeze(0).numpy(), dtype=np.float32)
           for t, _ in iter_preprocessed(p, height=280, width=504)][0]
    r = NativeChainRunner(os.environ.get("ABOT_MODELS", "/home/axera/ABot-Recon"),
                          device_id=int(os.environ.get("ABOT_DEVICE_ID", "0")))
    out = r.step(img, 0)
    patch = np.empty((1, 720, 1024), np.float32); r.dev.d2h(patch, r.b_patch)
    fused = np.empty((1, 725, 2048), np.float32); r.dev.d2h(fused, r.b_fused)
    np.savez(path, image=img, patch_tokens=patch, fused_hidden=fused, **out)
    r.close(); print("saved", path)


if len(sys.argv) == 3 and sys.argv[1] == "--make-ref":
    make_ref(sys.argv[2]); sys.exit(0)

from abot_axera.native_runner import _AxeDevice  # noqa: E402

MD = sys.argv[1]            # model dir with *_kitti02.axmodel
REF = sys.argv[2]           # ref npz from --make-ref
ref = np.load(REF)

dev = _AxeDevice()
bufs = []
def M(n): b = dev.malloc(n); bufs.append(b); return b

# ---- encoder ----
t0 = time.time(); enc = dev.load(os.path.join(MD, "encoder_kitti02.axmodel")); print(f"encoder loaded {time.time()-t0:.1f}s", flush=True)
ins, outs = dev.names(enc); print("encoder io:", ins, outs)
b_img = M(dev.in_size(enc, 0)); b_patch = M(dev.out_size(enc, 0))
dev.bind_in(enc, 0, b_img); dev.bind_out(enc, 0, b_patch)
dev.h2d(b_img, np.ascontiguousarray(ref["image"]))
t0 = time.time(); dev.run(enc); t_enc = time.time() - t0
patch = np.empty((1, 720, 1024), np.float32); dev.d2h(patch, b_patch)
d = np.abs(patch - ref["patch_tokens"]).max(); rel = d / (np.abs(ref["patch_tokens"]).max() + 1e-9)
print(f"encoder {t_enc:.2f}s  patch_tokens max|diff|={d:.3e} rel={rel:.2e} {'OK' if rel < 1e-3 else 'MISMATCH'}", flush=True)
dev.unload(enc)

# ---- heads (input = the card's fused_hidden) ----
t0 = time.time(); hd = dev.load(os.path.join(MD, "heads_kitti02.axmodel")); print(f"heads loaded {time.time()-t0:.1f}s", flush=True)
ins, outs = dev.names(hd); print("heads io:", ins, outs)
b_fused = M(dev.in_size(hd, 0)); dev.bind_in(hd, 0, b_fused)
ob = {n: M(dev.out_size(hd, i)) for i, n in enumerate(outs)}
for i, n in enumerate(outs): dev.bind_out(hd, i, ob[n])
dev.h2d(b_fused, np.ascontiguousarray(ref["fused_hidden"]))
t0 = time.time(); dev.run(hd); t_hd = time.time() - t0
shapes = {"local_points": (1, 280, 504, 3), "camera_features": (1, 725, 512), "confidence": (1, 280, 504, 1)}
worst = 0.0
for n, shp in shapes.items():
    a = np.empty(shp, np.float32); dev.d2h(a, ob[n])
    d = np.abs(a - ref[n]).max(); rel = d / (np.abs(ref[n]).max() + 1e-9); worst = max(worst, rel)
    print(f"heads {n:16s} max|diff|={d:.3e} rel={rel:.2e} {'OK' if rel < 1e-3 else 'MISMATCH'}", flush=True)
print(f"heads {t_hd:.2f}s")
dev.unload(hd)
for b in bufs: dev.free(b)
print("ONCHIP RESULT:", "PASS" if worst < 1e-3 else "FAIL")
