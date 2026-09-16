"""In-process job progress + ETA (single job at a time, so one slot is enough).

Writers:
  mapping_pipeline.run      set_phase("extract") / set_phase("post") / clear()
  backend.infer_paths       start_infer(total, hint_spf) then upd(frame) per frame
Reader:
  service GET /jobs         get() -> attached to the running job; index.html draws
                            "帧 i/N · 已用 · 预计剩余".

ETA model:  remaining_infer = (total - frame) * sec_per_frame
            remaining_post  = total * POST_SEC_PER_FRAME + POST_BASE_SEC
sec_per_frame is measured from this job as soon as one frame is done; before that
we use the last job's value (or the runner's hint: ~3 s native, ~9.5 s pyaxengine).
Post-processing (host pose head, cloud, renders) was ~45 s for 346 frames.
"""
from __future__ import annotations

import time

POST_SEC_PER_FRAME = 0.13
POST_BASE_SEC = 5.0

_S = {"phase": "", "frame": 0, "total": 0, "t0": 0.0, "t_phase": 0.0, "t_prev": 0.0, "spf": None}
_dts: list[float] = []                     # recent per-frame seconds (first frame excluded)
_last_spf: float | None = None


def clear() -> None:
    _S.update(phase="", frame=0, total=0, t0=0.0, t_phase=0.0, t_prev=0.0, spf=None)
    _dts.clear()


def set_phase(phase: str) -> None:
    now = time.time()
    if not _S["t0"]:
        _S["t0"] = now
    _S.update(phase=phase, t_phase=now)


def start_infer(total: int, hint_spf: float | None = None) -> None:
    now = time.time()
    if not _S["t0"]:
        _S["t0"] = now
    _dts.clear()
    _S.update(phase="infer", frame=0, total=int(total), t_phase=now, t_prev=now,
              spf=_last_spf or hint_spf)


def upd(frame: int, total: int | None = None) -> None:
    """Record progress. The rate comes from frame-to-frame deltas, so a slow start
    (model load, a decoder that had to be restarted) does not skew the estimate."""
    global _last_spf
    now = time.time()
    prev, _S["frame"] = _S["frame"], int(frame)
    if total is not None:
        _S["total"] = int(total)
    step = _S["frame"] - prev
    if prev > 0 and step > 0 and _S["t_prev"]:
        _dts.append((now - _S["t_prev"]) / step)
        del _dts[:-20]
        _S["spf"] = sum(_dts) / len(_dts)
        _last_spf = _S["spf"]
    _S["t_prev"] = now


# backwards-compatible alias (older backend called reset()+upd(frame,total))
def reset() -> None:
    clear()


def get() -> dict:
    now = time.time()
    phase, frame, total, spf = _S["phase"], _S["frame"], _S["total"], _S["spf"]
    d = {"phase": phase, "frame": frame, "total": total,
         "elapsed_s": round(now - _S["t0"], 1) if _S["t0"] else None,
         "sec_per_frame": round(spf, 2) if spf else None, "eta_s": None}
    if not phase:
        return d
    post_total = total * POST_SEC_PER_FRAME + POST_BASE_SEC if total else None
    if phase == "infer" and total and spf:
        d["eta_s"] = round((total - frame) * spf + post_total)
    elif phase == "post" and post_total is not None:
        d["eta_s"] = round(max(0.0, post_total - (now - _S["t_phase"])))
    return d
