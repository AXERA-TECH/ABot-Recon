"""Host-side + NPU timing on an AX650 board (torch-free; needs python3 + numpy + cffi + Pillow).

  python3 scripts/onchip_bench.py MODEL_DIR REF.npz [FEATS.npz]

REF.npz  : from scripts/onchip_check.py --make-ref (image / patch_tokens / fused_hidden / heads outs)
FEATS.npz: camera_features [N,725,512] (scripts/dump_feats.py output, or a slice of it)

Reports per-frame cost of: JPEG decode + preprocess (C kernel and numpy fallback), pose head,
VoxelGrid, and encoder / heads NPU execution through _AxeDevice. decoder_step needs ~5.6 GB CMM
in total and is not run here.
"""
from __future__ import annotations

import glob
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def bench(fn, n=5):
    fn(); t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n


def main():
    md, ref_path = sys.argv[1], sys.argv[2]
    feats_path = sys.argv[3] if len(sys.argv) > 3 else None
    print(f"machine {platform.machine()}  cpus {os.cpu_count()}  load {os.getloadavg()}")

    # ---- preprocess ----
    from PIL import Image
    from abot_axera import preprocess as pp
    frames = sorted(glob.glob(os.path.join(ROOT, "testdata", "ref_2915_frames", "*.jpg"))) or \
        sorted(glob.glob(os.path.join(ROOT, "testdata", "frames12", "*.jpg")))[:2]
    if frames:
        im = Image.open(frames[0]); im.load()
        print(f"frame {im.size[0]}x{im.size[1]}")
        t_dec = bench(lambda: Image.open(frames[0]).convert("RGB").load(), 3)
        c_ok = pp._load_c() is not None
        if c_ok:
            t_c = bench(lambda: pp.preprocess_image(im))
            print(f"preprocess C kernel : {t_c*1000:.1f} ms/frame   (jpeg decode alone {t_dec*1000:.1f} ms)")
            xc, _ = pp.preprocess_image(im)
        pp._lib, pp._lib_tried = None, True          # force numpy fallback
        t_np = bench(lambda: pp.preprocess_image(im), 2)
        xn, _ = pp.preprocess_image(im)
        print(f"preprocess numpy    : {t_np*1000:.1f} ms/frame" + (f"   max|C-numpy|={np.abs(xc-xn).max():.2e}" if c_ok else "  (C kernel unavailable)"))
        pp._lib_tried = False

    # ---- pose head ----
    if feats_path and os.path.exists(feats_path):
        from abot_axera.pose_head import AdjacentPoseHead
        d = os.environ.get("ABOT_DELIVERY", "/mnt/axera/abot650/ABot-Recon_AX650_交付包_20260907")
        ph = AdjacentPoseHead(os.path.join(d, "host_pose_head", "pose_head.safetensors"),
                              os.path.join(d, "host_pose_head", "pose_head_config.json"))
        F = np.load(feats_path)["camera_features"][:40]
        ph.run(F[:3]); t0 = time.time(); poses = ph.run(F); t_ph = (time.time() - t0) / len(F)
        print(f"pose head numpy     : {t_ph*1000:.1f} ms/frame ({len(F)} frames)")

    # ---- voxel grid (1M points) ----
    from abot_axera.pointcloud import VoxelGrid
    rng = np.random.default_rng(0); P = (rng.random((1_000_000, 3)) * 8).astype(np.float32)
    t0 = time.time(); g = VoxelGrid(P, 8 / 550); g.mean(P); g.mean(P); t_v = time.time() - t0
    print(f"VoxelGrid 1M points : {t_v:.2f} s (grid + 2 means)")

    # ---- NPU: encoder + heads ----
    from abot_axera.native_runner import _AxeDevice
    ref = np.load(ref_path)
    dev = _AxeDevice(); bufs = []
    M = lambda n: (bufs.append(dev.malloc(n)) or bufs[-1])
    for name, inp, in_key, outs in (("encoder", "image", "image", ("patch_tokens",)),
                                    ("heads", "fused_hidden", "fused_hidden", ("local_points", "camera_features", "confidence"))):
        t0 = time.time(); m = dev.load(os.path.join(md, f"{name}_kitti02.axmodel")); t_load = time.time() - t0
        ins, outn = dev.names(m)
        bi = M(dev.in_size(m, 0)); dev.bind_in(m, 0, bi)
        ob = [M(dev.out_size(m, i)) for i in range(len(outn))]
        for i, b in enumerate(ob): dev.bind_out(m, i, b)
        dev.h2d(bi, np.ascontiguousarray(ref[in_key]))
        dev.run(m); t0 = time.time()
        for _ in range(5): dev.run(m)
        t_run = (time.time() - t0) / 5
        worst = 0.0
        for i, n in enumerate(outn):
            a = np.empty(ref[n].shape, np.float32); dev.d2h(a, ob[i]); worst = max(worst, float(np.abs(a - ref[n]).max()))
        print(f"NPU {name:8s}: load {t_load:.1f}s  run {t_run*1000:.0f} ms  max|diff vs card|={worst:.1e}")
        dev.unload(m)
    for b in bufs: dev.free(b)
    print("done")


if __name__ == "__main__":
    main()
