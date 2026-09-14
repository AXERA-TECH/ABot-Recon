#!/usr/bin/env bash
# Start the ABot-Recon mapping service (FastAPI dashboard/API + viser 3D) on an Axera NPU host.
# Every setting is an env var with a default; override before calling, e.g.
#   ABOT_DEVICE_ID=7 MAP_PORT=8011 bash start_service.sh
export PATH=/usr/bin/axcl:$PATH

# --- inference backend ---
export ABOT_BACKEND="${ABOT_BACKEND:-axera}"          # axera = Axera NPU path (abot_axera); anything else = upstream torch/GPU
export ABOT_RUNNER="${ABOT_RUNNER:-native}"           # native (KV on device, ~3 s/frame) | pyaxengine (reference, ~9.5 s/frame)
export ABOT_DEVICE="${ABOT_DEVICE:-auto}"             # auto | axcl (PCIe card) | ax650 (on-chip)
export ABOT_DEVICE_ID="${ABOT_DEVICE_ID:-6}"          # AXCL card index (dell: 0-5 belong to ax-llm)
export ABOT_MODELS="${ABOT_MODELS:-/home/axera/ABot-Recon}"          # encoder/decoder_step/heads *.axmodel
export ABOT_MODEL_SUFFIX="${ABOT_MODEL_SUFFIX:-_kitti02}"
export ABOT_DELIVERY="${ABOT_DELIVERY:-/home/axera/abot650/ABot-Recon_AX650_交付包_20260907}"  # host_pose_head/ lives here

# --- service ---
export MAP_PORT="${MAP_PORT:-8011}"                   # dashboard + API   (8000-8005/8080 are taken by ax-llm on dell)
export VISER_PORT="${VISER_PORT:-8082}"               # viser 3D canvas (embedded in the dashboard)
export MAP_DATA="${MAP_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/jobs}"
export MAP_FPS="${MAP_FPS:-8}"
export MAP_IDLE_UNLOAD="${MAP_IDLE_UNLOAD:-100000}"   # seconds idle before the model is unloaded (keep it hot)
export MAP_EST_LOAD_SEC="${MAP_EST_LOAD_SEC:-25}"     # model-load progress bar estimate

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYBIN="${PYBIN:-/home/axera/miniforge3/envs/py312/bin}"

cd "$REPO/service"
exec "$PYBIN/uvicorn" service:app --host 0.0.0.0 --port "$MAP_PORT"
