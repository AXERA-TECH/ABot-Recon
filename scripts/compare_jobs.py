"""Compare two job output dirs: meta counts, camera centers (recon.npz), cloud sizes."""
import sys, json, os, numpy as np
a, b = sys.argv[1], sys.argv[2]
ma, mb = json.load(open(os.path.join(a, "meta.json"))), json.load(open(os.path.join(b, "meta.json")))
for k in ("frames", "points_raw", "walked_area_kept", "cloud_points", "infer_s", "total_s"):
    print(f"{k:18s} {str(ma.get(k)):>22s} {str(mb.get(k)):>22s}")
za, zb = np.load(os.path.join(a, "recon.npz")), np.load(os.path.join(b, "recon.npz"))
ca, cb = za["cams"], zb["cams"]
d = np.linalg.norm(ca - cb, axis=1); span = float(np.linalg.norm(ca.max(0) - ca.min(0)))
print(f"cams: mean diff {d.mean():.3e}  max {d.max():.3e}  (trajectory span {span:.3f})")
pa, pb = za["poses"], zb["poses"]
print(f"poses: max|diff| {np.abs(pa - pb).max():.3e}")
