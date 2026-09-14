"""Numeric + timing check: NativeChainRunner (device-resident KV) vs the pyaxengine reference runner.

  python scripts/validate_native.py --runner pyaxengine --dump /tmp/ref.npz  [--dev 7]
  python scripts/validate_native.py --runner native     --dump /tmp/nat.npz  [--dev 7]
  python scripts/validate_native.py --compare /tmp/ref.npz /tmp/nat.npz

Run the two runners in separate processes (each loads ~3.2 GB of models on the card).
Frames: testdata/frames12 (or --frames DIR), preprocessed exactly like the backend.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

KV_SHAPE = (1, 18, 16, 11, 725, 64)


def load_frames(frames_dir, n):
    from abot_axera.preprocess import iter_preprocessed
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))[:n]
    assert paths, f"no jpgs in {frames_dir}"
    return paths, [np.ascontiguousarray(x[None], dtype=np.float32) for x, _ in iter_preprocessed(paths, height=280, width=504)]


def run_pyaxengine(imgs, dev, model_dir, suffix):
    from abot_axera.runners import PyAxEngineRunner
    t0 = time.time(); r = PyAxEngineRunner(model_dir, device_id=dev, suffix=suffix); print(f"load {time.time()-t0:.1f}s", flush=True)
    pk = np.zeros(KV_SHAPE, np.float32); pv = np.zeros(KV_SHAPE, np.float32); pvld = np.zeros((1,), np.float32)
    outs, times = [], []
    for fi, img in enumerate(imgs):
        a = time.time()
        pt = r.run_encoder(img)
        d = r.run_decoder({"patch_tokens": np.ascontiguousarray(pt, dtype=np.float32), "past_key": pk, "past_value": pv,
                           "past_valid": pvld, "frame_index": np.array([fi], np.float32)})
        pk = np.ascontiguousarray(d["present_key"], dtype=np.float32); pv = np.ascontiguousarray(d["present_value"], dtype=np.float32)
        pvld = np.ascontiguousarray(d["present_valid"], dtype=np.float32)
        h = r.run_heads(np.ascontiguousarray(d["fused_hidden"], dtype=np.float32))
        times.append(time.time() - a)
        outs.append({k: np.asarray(h[k], np.float32) for k in ("camera_features", "local_points", "confidence")})
        print(f"frame {fi}: {times[-1]:.2f}s", flush=True)
    return outs, times, (pk, pv, pvld)


def run_native(imgs, dev, model_dir, suffix, device):
    from abot_axera.native_runner import NativeChainRunner
    t0 = time.time(); r = NativeChainRunner(model_dir, device_id=dev, suffix=suffix, device=device); print(f"load {time.time()-t0:.1f}s", flush=True)
    outs, times = [], []
    r.reset()
    for fi, img in enumerate(imgs):
        a = time.time()
        h = r.step(img, fi)
        times.append(time.time() - a)
        outs.append(h)
        print(f"frame {fi}: {times[-1]:.2f}s", flush=True)
    kv = r.read_kv()
    # second sequence after reset must reproduce frame 0 exactly (cache really cleared)
    r.reset(); h0 = r.step(imgs[0], 0)
    print("reset check (frame0 rerun max|diff|):", max(float(np.abs(h0[k] - outs[0][k]).max()) for k in h0), flush=True)
    r.close()
    return outs, times, kv


def dump(path, outs, times, kv):
    d = {"times": np.array(times, np.float64), "kv_key": kv[0], "kv_value": kv[1], "kv_valid": kv[2]}
    for i, o in enumerate(outs):
        for k, v in o.items():
            d[f"{i}:{k}"] = v
    np.savez(path, **d)
    print(f"dumped {path}; mean {np.mean(times[1:]) if len(times) > 1 else times[0]:.2f}s/frame", flush=True)


def compare(a, b):
    A, B = np.load(a), np.load(b)
    worst = 0.0
    for k in sorted(A.files):
        if k == "times":
            continue
        x, y = A[k], B[k]
        diff = np.abs(x - y); m = float(diff.max()); rel = m / (float(np.abs(x).max()) + 1e-9)
        worst = max(worst, rel)
        print(f"{k:28s} max|diff|={m:.3e}  rel={rel:.2e}  {'OK' if rel < 1e-3 else 'MISMATCH'}")
    ta, tb = A["times"], B["times"]
    print(f"\n{os.path.basename(a)}: {np.mean(ta[1:]):.2f}s/frame   {os.path.basename(b)}: {np.mean(tb[1:]):.2f}s/frame   "
          f"speedup x{np.mean(ta[1:]) / np.mean(tb[1:]):.2f}")
    print("RESULT:", "PASS" if worst < 1e-3 else "FAIL", f"(worst rel {worst:.2e})")
    return worst < 1e-3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", choices=["pyaxengine", "native"])
    ap.add_argument("--device", default="auto", help="native only: axcl | ax650 | auto")
    ap.add_argument("--dev", type=int, default=int(os.environ.get("ABOT_DEVICE_ID", "0")))
    ap.add_argument("--models", default=os.environ.get("ABOT_MODELS", "/home/axera/ABot-Recon"))
    ap.add_argument("--suffix", default=os.environ.get("ABOT_MODEL_SUFFIX", "_kitti02"))
    ap.add_argument("--frames", default=os.path.join(ROOT, "testdata", "frames12"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--dump")
    ap.add_argument("--compare", nargs=2, metavar=("REF", "NEW"))
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    paths, imgs = load_frames(args.frames, args.n)
    print(f"{len(imgs)} frames from {args.frames}", flush=True)
    if args.runner == "pyaxengine":
        outs, times, kv = run_pyaxengine(imgs, args.dev, args.models, args.suffix)
    else:
        outs, times, kv = run_native(imgs, args.dev, args.models, args.suffix, args.device)
    if args.dump:
        dump(args.dump, outs, times, kv)
