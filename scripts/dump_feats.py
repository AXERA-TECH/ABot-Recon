"""Run the native runner over all frames of a video and dump camera_features (all frames)
+ local_points/confidence (every 20th frame) as a reference for scripts/validate_torchfree.py.
  python scripts/dump_feats.py VIDEO OUT_PREFIX [fps]"""
import os, sys, glob, time, tempfile, shutil
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "service"))
from mapping_pipeline import _extract_frames
from abot_axera.preprocess import iter_preprocessed
from abot_axera.native_runner import NativeChainRunner

video, out, fps = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8
fdir = tempfile.mkdtemp(prefix="abot_dump_")
frames = _extract_frames(video, fps, fdir)
print(len(frames), "frames", flush=True)
r = NativeChainRunner(os.environ.get("ABOT_MODELS", "/home/axera/ABot-Recon"), device_id=int(os.environ.get("ABOT_DEVICE_ID", "0")))
r.reset()
feats, lp, cf, keep = [], [], [], []
t0 = time.time()
for fi, (chw, _) in enumerate(iter_preprocessed(frames, height=280, width=504)):
    img = np.ascontiguousarray(chw[None], dtype=np.float32)
    h = r.step(img, fi)
    feats.append(h["camera_features"][0])
    if fi % 20 == 0:
        keep.append(fi); lp.append(h["local_points"][0]); cf.append(h["confidence"][0, ..., 0])
    if fi % 25 == 0:
        print(f"frame {fi} {time.time()-t0:.0f}s", flush=True)
r.close()
# keep a few raw frames for the preprocessing check
os.makedirs(out + "_frames", exist_ok=True)
for i in (0, len(frames) // 2, len(frames) - 1):
    shutil.copy(frames[i], out + "_frames/")
shutil.rmtree(fdir, ignore_errors=True)
np.savez(out, camera_features=np.stack(feats), keep=np.array(keep), local_points=np.stack(lp), confidence=np.stack(cf))
print("saved", out, np.stack(feats).shape, flush=True)
