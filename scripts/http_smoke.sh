#!/usr/bin/env bash
# End-to-end HTTP smoke against a running service: upload a video, poll until done, fetch floorplan.
#   scripts/http_smoke.sh [video] [base_url]
set -e
V="${1:-$(dirname "$0")/../testdata/villa3f_seg2.mp4}"
B="${2:-http://127.0.0.1:${MAP_PORT:-8011}}"
echo "=== GET /status ==="; curl -s $B/status; echo
echo "=== POST /jobs ($V) ==="
RESP=$(curl -s -F "file=@$V" -F "fps=${MAP_FPS:-8}" $B/jobs); echo "$RESP"
JID=$(echo "$RESP" | sed -n 's/.*"job_id"[ :]*"\([a-f0-9]*\)".*/\1/p')
for i in $(seq 1 720); do
  J=$(curl -s $B/jobs/$JID)
  ST=$(echo "$J" | sed -n 's/.*"status"[ :]*"\([a-z]*\)".*/\1/p')
  P=$(echo "$J" | python3 -c 'import sys,json; p=json.load(sys.stdin).get("progress") or {}; print(p.get("phase",""), p.get("frame",""), "/", p.get("total",""), "eta", p.get("eta_s",""))' 2>/dev/null)
  echo "poll $i: $ST $P"
  if [ "$ST" = "done" ] || [ "$ST" = "error" ]; then echo "$J" | head -c 800; echo; break; fi
  sleep 3
done
OUT="${TMPDIR:-/tmp}/${JID}_floorplan.png"
curl -s -o "$OUT" $B/jobs/$JID/floorplan.png && ls -la "$OUT"
