"""One-off parity check of the torch-free host side against the torch/torchvision/open3d originals.

  python scripts/validate_torchfree.py --frames DIR [--feats REF.npz]

Needs torch + torchvision (+ open3d) installed ONLY for this script; the delivery's
host_pose_head/adjacent_pose_head.py is loaded as the torch reference. ABOT_DELIVERY points at
the delivery package. REF.npz (from scripts/dump_feats.py) holds camera_features [N,725,512]
plus local_points/confidence for a few frames.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DELIV = os.environ.get("ABOT_DELIVERY", "/home/axera/abot650/ABot-Recon_AX650_交付包_20260907")


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max()), float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def check_preprocess(frames):
    import torch, torchvision.transforms.functional as tvf
    from torchvision.transforms import InterpolationMode
    from PIL import Image
    from abot_axera.preprocess import preprocess_image
    worst = 0.0
    for p in frames:
        with Image.open(p) as im:
            t = tvf.to_tensor(im.convert("RGB"))
            _, sh, sw = t.shape
            rh = max(1, round(sh * 504 / max(sw, 1)))
            r = tvf.resize(t, [rh, 504], interpolation=InterpolationMode.BICUBIC, antialias=True)
            if rh > 280:
                top = round((rh - 280) * 0.5); r = tvf.crop(r, top, 0, 280, 504)
            elif rh < 280:
                pt = (280 - rh) // 2
                canvas = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).expand(3, 280, 504).clone()
                canvas[:, pt:pt + rh] = r; r = canvas
            ref = r.numpy()
            t0 = time.time(); x, tf = preprocess_image(im); dt = time.time() - t0
        d, rr = rel(x, ref); worst = max(worst, rr)
        print(f"preprocess {os.path.basename(p):14s} {sw}x{sh} max|diff|={d:.2e} rel={rr:.2e} ({dt*1000:.0f} ms)")
    return worst


def check_pose_head(feats):
    import torch
    spec = importlib.util.spec_from_file_location("ref_pose", os.path.join(DELIV, "host_pose_head", "adjacent_pose_head.py"))
    ref_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref_mod)
    import json
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(DELIV, "host_pose_head", "pose_head_config.json"))); cfg.pop("class", None)
    ref = ref_mod.AdjacentPoseHead(**cfg).eval()
    ref.load_state_dict(load_file(os.path.join(DELIV, "host_pose_head", "pose_head.safetensors")), strict=True)
    state, ref_poses = None, []
    t0 = time.time()
    with torch.inference_mode():
        for f in feats:
            pose, state = ref(torch.from_numpy(f).float()[None, None], camera_state=state, return_state=True)
            ref_poses.append(pose[0, 0].numpy())
    t_ref = time.time() - t0
    ref_poses = np.stack(ref_poses)

    from abot_axera.pose_head import AdjacentPoseHead
    ph = AdjacentPoseHead(os.path.join(DELIV, "host_pose_head", "pose_head.safetensors"),
                          os.path.join(DELIV, "host_pose_head", "pose_head_config.json"))
    t0 = time.time(); poses = ph.run(feats); t_np = time.time() - t0
    d, rr = rel(poses, ref_poses)
    c_ref, c_np = ref_poses[:, :3, 3], poses[:, :3, 3]
    cd = np.linalg.norm(c_ref - c_np, axis=1)
    span = float(np.linalg.norm(c_ref.max(0) - c_ref.min(0)))
    print(f"pose_head {len(feats)} frames: max|diff|={d:.2e} rel={rr:.2e}; cam-center diff mean={cd.mean():.2e} "
          f"max={cd.max():.2e} (trajectory span {span:.3f}); torch {t_ref:.2f}s vs numpy {t_np:.2f}s")
    return rr, poses


def check_points(ref, poses):
    import torch
    from abot_axera.backend import transform_local_points
    keep = ref["keep"]; lp = ref["local_points"]
    wp = transform_local_points(lp, poses[keep])
    R = torch.from_numpy(poses[keep][:, :3, :3]); t = torch.from_numpy(poses[keep][:, :3, 3])
    wp_t = (torch.einsum("nij,nhwj->nhwi", R, torch.from_numpy(lp)) + t[:, None, None]).numpy()
    d, rr = rel(wp, wp_t)
    print(f"world_points {lp.shape[0]} frames: max|diff|={d:.2e} rel={rr:.2e}")
    conf = 1 / (1 + np.exp(-ref["confidence"])); conf_t = torch.sigmoid(torch.from_numpy(ref["confidence"])).numpy()
    print(f"confidence sigmoid: max|diff|={rel(conf, conf_t)[0]:.2e}")
    # voxel down-sample vs open3d
    from abot_axera.pointcloud import voxel_down_sample, write_ply, read_ply
    P = wp[:, ::2, ::2].reshape(-1, 3); C = np.clip(conf[:, ::2, ::2].reshape(-1, 1).repeat(3, 1), 0, 1)
    m = np.isfinite(P).all(1); P, C = P[m], C[m]
    span = float((np.percentile(P, 99, 0) - np.percentile(P, 1, 0)).mean()); vox = max(1e-3, span / 550)
    p2, c2 = voxel_down_sample(P, C, vox)
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(C.astype(np.float64)); pc = pc.voxel_down_sample(vox)
        print(f"voxel_down_sample: numpy {len(p2)} pts vs open3d {len(pc.points)} pts (in {len(P)})")
    except ImportError:
        print(f"voxel_down_sample: numpy {len(p2)} pts (open3d not installed, skipped)")
    tmp = "/tmp/_abot_ply_check.ply"; write_ply(tmp, p2, c2); q, qc = read_ply(tmp)
    print(f"ply roundtrip: {len(q)} pts, max|diff|={np.abs(q - p2).max():.2e}, colors {qc.dtype} {qc.shape}")
    return rr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.join(ROOT, "testdata", "frames12"))
    ap.add_argument("--feats", default=os.path.join(ROOT, "testdata", "ref_2915.npz"))
    a = ap.parse_args()
    frames = sorted(glob.glob(os.path.join(a.frames, "*.jpg")))[:4]
    extra = sorted(glob.glob(a.feats.replace(".npz", "_frames") + "/*.jpg"))
    worst = check_preprocess(frames + extra)
    ref = np.load(a.feats)
    r2, poses = check_pose_head(ref["camera_features"])
    r3 = check_points(ref, poses)
    worst = max(worst, r2, r3)
    print("RESULT:", "PASS" if worst < 1e-3 else "CHECK", f"(worst rel {worst:.2e})")
