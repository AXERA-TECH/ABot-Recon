#!/usr/bin/env python3
"""ABot-Recon 建图服务 — 任务队列 + 模型热加载 + 网页可视化。

- **热加载**:模型不在启动时加载;首次提交任务(或网页点「加载」)时才载入 NPU。
  空闲超过 MAP_IDLE_UNLOAD 秒自动卸载;下次用时再载。
- **网页(index.html, MAP_PORT)**:上传视频 / 任务列表(进度条 + 预计剩余)/ 模型状态 / 3D 点云(viser iframe)。
- **API(FastAPI, MAP_PORT)**:

  POST /jobs                 上传视频(multipart 'file'[, 'fps', 'ceiling_cut', 'ceiling_keep']) → {job_id}
  GET  /jobs                 所有 job(运行中的带 progress)
  GET  /jobs/{id}            job 状态 + 产物 + meta
  GET  /jobs/{id}/{file}     下载产物
  POST /jobs/{id}/view       把该 job 点云载入 viser
  POST /jobs/{id}/rerun      重跑;DELETE /jobs/{id} 删除
  GET  /status               模型热加载状态
  GET  /models               模型列表 + 当前加载
  POST /models/load?name=    立即加载;POST /models/unload 卸载
"""
import os, re, uuid, threading, traceback, shutil, json, time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import numpy as np

DATA = os.environ.get("MAP_DATA", "/data/jobs")
FPS = int(os.environ.get("MAP_FPS", "8"))
IDLE_UNLOAD = int(os.environ.get("MAP_IDLE_UNLOAD", "1800"))   # 30 min
EST_LOAD_SEC = float(os.environ.get("MAP_EST_LOAD_SEC", "30")) # progress-bar estimate
os.makedirs(DATA, exist_ok=True)

# ABot-Recon: single model entry (name shown in the dashboard dropdown).
MODELS = {"ABot-Recon": os.environ.get("ABOT_MODEL_ID", "acvlab/ABot-Recon")}
DEFAULT_MODEL = "ABot-Recon"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="ABot-Recon 建图服务")
_jobs = {}                 # id -> {status, out, meta, ...}
_viser = {"server": None, "status_md": None, "jobs_dd": None, "show_traj": None,
          "traj_handle": None, "cloud_handle": None, "psize_mult": None, "base_psize": None,
          "cloud_pts": None, "cloud_col": None, "cloud_zspan": None, "ceil_slider": None}


class ModelManager:
    """热加载:按需载入 GPU、空闲自动卸载、可切换 3 个模型。"""
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None            # loaded AbotRecon or None
        self.loaded_name = None      # which model is in GPU
        self.name = DEFAULT_MODEL    # currently SELECTED (target)
        self.status = "offline"      # offline | loading | online | error
        self.progress = 0            # 0..100 during load
        self.last_used = 0.0
        self.busy = False            # a job is actively processing (no idle countdown)
        self.err = None

    # ---- internals (assume lock held) ----
    def _unload_locked(self):
        import gc
        m, self.model, self.loaded_name = self.model, None, None
        close = getattr(getattr(m, "runner", None), "close", None)
        if close is not None:
            try: close()                       # free device buffers + unload axmodels
            except Exception as e: print("runner close:", e)
        gc.collect()
        self.status = "offline"; self.progress = 0

    def _load_locked(self, name):
        if name not in MODELS:
            raise ValueError(f"unknown model {name}")
        if self.loaded_name == name and self.model is not None:
            self.status = "online"; self.last_used = time.time(); return
        if self.model is not None:
            self._unload_locked()
        path = MODELS[name]     # HF repo id (auto-download) or local dir — no local-file gate
        self.name = name; self.status = "loading"; self.progress = 0; self.err = None
        stop = threading.Event()
        def _tick():
            t0 = time.time()
            while not stop.is_set():
                self.progress = min(95, round((time.time() - t0) / EST_LOAD_SEC * 95))
                time.sleep(0.4)
        th = threading.Thread(target=_tick, daemon=True); th.start()
        try:
            from mapping_pipeline import load_ready_model
            self.model = load_ready_model(path)
            self.loaded_name = name; self.status = "online"; self.progress = 100
            self.last_used = time.time()
        except Exception as e:
            self.status = "error"; self.err = str(e)[:300]
            self.model = None; self.loaded_name = None
            raise
        finally:
            stop.set()

    def _ensure_locked(self, name):
        if self.loaded_name == name and self.model is not None:
            self.last_used = time.time(); return
        self._load_locked(name)

    # ---- public ----
    def load(self, name=None):
        with self.lock:
            self._load_locked(name or self.name)

    def unload(self):
        with self.lock:
            self._unload_locked()

    def select(self, name):
        if name not in MODELS:
            raise ValueError(name)
        self.name = name

    def gpu_mem(self):
        return None, None                      # NPU memory is device-resident; nothing to report here

    def snapshot(self):
        alloc, res = self.gpu_mem()
        # idle countdown only when online AND not actively processing a job
        idle = int(time.time() - self.last_used) if (self.status == "online" and not self.busy) else None
        return {"status": self.status, "selected": self.name, "loaded": self.loaded_name,
                "progress": self.progress, "busy": self.busy, "idle_sec": idle,
                "auto_unload_sec": IDLE_UNLOAD, "gpu_alloc_gb": alloc, "gpu_reserved_gb": res,
                "error": self.err}


mgr = ModelManager()


def _idle_watch():
    while True:
        time.sleep(30)
        try:
            if mgr.status == "online" and not mgr.busy and time.time() - mgr.last_used > IDLE_UNLOAD:
                with mgr.lock:
                    if mgr.status == "online" and not mgr.busy and time.time() - mgr.last_used > IDLE_UNLOAD:
                        print(f"[idle] unloading {mgr.loaded_name} after {IDLE_UNLOAD}s idle")
                        mgr._unload_locked()
        except Exception as e:
            print("idle-watch err:", e)


def _worker(job_id, video_path, mask_sky, model_name, fps, ceiling_cut, ceiling_keep):
    from mapping_pipeline import run
    out = os.path.join(DATA, job_id)
    try:
        _jobs[job_id]["status"] = "running"
        print(f"[job] {job_id} running: {video_path} fps={fps} model={model_name}", flush=True)
        with mgr.lock:                          # serialize the NPU + hold model during job
            print(f"[job] {job_id} got model lock; ensuring model loaded…", flush=True)
            mgr._ensure_locked(model_name)      # lazy-load if needed
            mgr.busy = True; mgr.last_used = time.time()
            try:
                # ABot run: world_points + c2w poses → dense RGB cloud / floorplan / 3D.
                meta = run(video_path, out, fps=fps, ceiling_cut=ceiling_cut,
                           ceiling_keep=ceiling_keep, model=mgr.model)
            finally:
                mgr.busy = False; mgr.last_used = time.time()
        _jobs[job_id].update(status="done", meta=meta, products=sorted(os.listdir(out)))
        print(f"[job] {job_id} done: {meta.get('frames','?')} 帧 {meta.get('cloud_points','?')} 点", flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        _jobs[job_id].update(status="error", error=(str(e) or repr(e)), traceback=tb[-2000:])
        print(f"[job] {job_id} FAILED:\n{tb}", flush=True)


# ---------------- job API ----------------
def _safe_name(filename):
    """Keep the user's own filename (so nothing gets renamed to input.MOV), but sanitize it
    to a safe basename — no path traversal, alnum/._- only. Ensure it has an extension."""
    base = os.path.basename(filename or "video.mp4")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".") or "video.mp4"
    if not os.path.splitext(base)[1]:
        base += ".mp4"
    return base


def _submit_job(filename, mask_sky, model_name, writer, fps, ceiling_cut=False, ceiling_keep=0.85):
    """Register + launch a job. `writer(path)` writes the uploaded video to path."""
    if model_name not in MODELS:
        model_name = mgr.name
    fps = int(fps) if fps and int(fps) > 0 else FPS
    fps = max(1, min(fps, 15))                           # clamp: sane抽帧率范围
    ceiling_cut = bool(ceiling_cut)
    ceiling_keep = max(0.3, min(float(ceiling_keep or 0.85), 1.0))   # clamp fraction
    job_id = uuid.uuid4().hex[:12]
    out = os.path.join(DATA, job_id); os.makedirs(out, exist_ok=True)
    vpath = os.path.join(out, _safe_name(filename))     # preserve the user's filename
    writer(vpath)
    _jobs[job_id] = {"status": "queued", "out": out, "t": time.time(),
                     "video": os.path.basename(vpath), "vpath": vpath, "mask_sky": mask_sky,
                     "model": model_name, "fps": fps,
                     "ceiling_cut": ceiling_cut, "ceiling_keep": ceiling_keep}
    threading.Thread(target=_worker,
                     args=(job_id, vpath, mask_sky, model_name, fps, ceiling_cut, ceiling_keep),
                     daemon=True).start()
    return job_id


@app.post("/jobs")
async def submit(file: UploadFile = File(...), mask_sky: bool = Form(False),
                 model: str = Form(""), fps: int = Form(0),
                 ceiling_cut: bool = Form(False), ceiling_keep: float = Form(0.85)):
    model_name = model or mgr.name
    if model_name not in MODELS:
        raise HTTPException(400, f"unknown model {model_name}")
    def _w(p):
        with open(p, "wb") as f:
            shutil.copyfileobj(file.file, f)
    job_id = _submit_job(file.filename, mask_sky, model_name, _w, fps, ceiling_cut, ceiling_keep)
    return {"job_id": job_id, "model": model_name}


def _public(j):
    """Job dict for the API: drop server paths; attach live progress/ETA to the running job
    (single job at a time, so the process-wide progress slot belongs to it)."""
    out = {k: v for k, v in j.items() if k not in ("out", "vpath")}
    if out.get("status") == "running":
        try:
            from abot_axera import progress as _prog
            p = _prog.get()
            if p.get("phase"):
                out["progress"] = p
        except Exception:
            pass
    return out


@app.get("/jobs")
def list_jobs():
    return {jid: _public(j) for jid, j in _jobs.items()}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "no such job")
    return _public(_jobs[job_id])


@app.get("/jobs/{job_id}/{fname}")
def download(job_id: str, fname: str):
    p = os.path.join(DATA, job_id, os.path.basename(fname))
    if not os.path.exists(p):
        raise HTTPException(404, "no such product")
    return FileResponse(p, filename=fname)


@app.post("/jobs/{job_id}/view")
def view(job_id: str):
    ply = os.path.join(DATA, job_id, "cloud.ply")
    if not os.path.exists(ply):
        raise HTTPException(404, "cloud not ready")
    _load_viser_cloud(job_id)
    return {"loaded": job_id, "viser_port": os.environ.get("VISER_PORT", "8080")}


@app.post("/jobs/{job_id}/rerun")
def rerun(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "no such job")
    j = _jobs[job_id]; out = j.get("out", os.path.join(DATA, job_id))
    src = j.get("vpath")
    if not src or not os.path.exists(src):        # fallback: find the source VIDEO file in the dir
        vext = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")   # a FILE (skip input_frames/ dir)
        cand = [f for f in os.listdir(out)
                if os.path.isfile(os.path.join(out, f)) and f.lower().endswith(vext)] if os.path.isdir(out) else []
        if not cand:
            raise HTTPException(400, "input video not found")
        src = os.path.join(out, cand[0])
    new_id = _submit_job(j.get("video") or os.path.basename(src), j.get("mask_sky", False),
                         j.get("model") or mgr.name, lambda p: shutil.copyfile(src, p),
                         j.get("fps") or FPS,
                         j.get("ceiling_cut", False), j.get("ceiling_keep", 0.85))
    return {"job_id": new_id}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "no such job")
    out = _jobs.pop(job_id).get("out")
    if out and os.path.isdir(out):
        shutil.rmtree(out, ignore_errors=True)
    return {"deleted": job_id}


# ---------------- model control API ----------------
@app.get("/status")
def status():
    return mgr.snapshot()


@app.get("/models")
def models():
    return {"selected": mgr.name, "loaded": mgr.loaded_name,
            "models": {n: {"path": p, "available": os.path.exists(p),
                           "loaded": n == mgr.loaded_name} for n, p in MODELS.items()}}


@app.post("/models/select")
def model_select(name: str):
    try: mgr.select(name)
    except ValueError: raise HTTPException(400, f"unknown model {name}")
    return {"selected": mgr.name}


@app.post("/models/load")
def model_load(name: str = ""):
    target = name or mgr.name
    if target not in MODELS:
        raise HTTPException(400, f"unknown model {target}")
    threading.Thread(target=lambda: _safe_load(target), daemon=True).start()
    return {"loading": target}


@app.post("/models/unload")
def model_unload():
    mgr.unload(); return {"status": mgr.status}


def _safe_load(name):
    try: mgr.load(name)
    except Exception as e: print("load err:", e)


@app.get("/", response_class=HTMLResponse)
def root():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(p):
        html = open(p, encoding="utf-8").read().replace("__VISER_PORT__", os.environ.get("VISER_PORT", "8080"))
        return HTMLResponse(html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return HTMLResponse("<h1>ABot-Recon</h1><p>index.html missing; use /api or /docs</p>")


@app.get("/api")
def api_info():
    return {"service": "ABot-Recon 建图(热加载)", "jobs": len(_jobs),
            "viser": f"port {os.environ.get('VISER_PORT','8080')}", "model": mgr.snapshot(),
            "api": ["POST /jobs (file[,mask_sky,model])", "GET /jobs/{id}", "GET /jobs/{id}/{product}",
                    "GET /status", "GET /models", "POST /models/load?name=", "POST /models/unload"]}


# ---------------- viser web (3D + control panel) ----------------
def _redraw_cloud():
    """(Re)draw /cloud from the stored points, applying the live ceiling-height filter. Called
    on load and whenever the '保留高度' slider moves — no re-run, just re-filter + re-upload."""
    srv = _viser["server"]; pts = _viser.get("cloud_pts"); col = _viser.get("cloud_col")
    if srv is None or pts is None or not len(pts):
        return
    keep = float(_viser["ceil_slider"].value) if _viser.get("ceil_slider") is not None else 1.0
    if keep < 0.999:
        lo, hi = _viser["cloud_zspan"]
        m = pts[:, 2] <= lo + keep * (hi - lo)          # z is up (recentered) → cut the top
        P, C = pts[m], col[m]
    else:
        P, C = pts, col
    base = _viser.get("base_psize") or 0.01
    mult = float(_viser["psize_mult"].value) if _viser.get("psize_mult") is not None else 1.0
    _viser["cloud_handle"] = srv.scene.add_point_cloud(
        "/cloud", points=P, colors=C, point_size=base * mult, point_shape="circle")


def _load_viser_cloud(job_id):
    from abot_axera.pointcloud import read_ply
    srv = _viser["server"]
    if srv is None:
        return
    d = os.path.join(DATA, job_id)
    # prefer the REAL-RGB cloud (looks like the actual room — understandable/immersive);
    # fall back to the time-colored one only if RGB is missing.
    ply = os.path.join(d, "cloud.ply")
    if not os.path.exists(ply):
        ply = os.path.join(d, "cloud_viz.ply")
    pts, col = read_ply(ply)
    if col is None:
        col = np.full((len(pts), 3), 200, np.uint8)

    # ABot mapping_pipeline already writes cloud.ply floor-aligned AND upright, and stores the
    # matching aligned camera centers as recon.npz["cams"] — so here we just load them (no flip,
    # no basis recompute) and recenter both below.
    cams = None
    npz = os.path.join(d, "recon.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        if "cams" in z:
            cams = z["cams"].astype(np.float32)

    # recenter on the model's own bbox center so viser's orbit pivot / zoom target sits ON the
    # model. viser orbits/zooms toward the look-at point (default = world origin), but our cloud
    # lives in the floor-frame far from origin → orbit felt off and zoom drifted into empty space.
    if len(pts):
        c = (np.percentile(pts, 1, 0) + np.percentile(pts, 99, 0)) / 2
        pts = pts - c
        if cams is not None:
            cams = cams - c

    _viser["cloud_pts"] = pts
    _viser["cloud_col"] = col
    _viser["cloud_zspan"] = ((float(np.percentile(pts[:, 2], 1)), float(np.percentile(pts[:, 2], 99)))
                             if len(pts) else (0.0, 1.0))
    _viser["base_psize"] = (float(np.ptp(pts, 0).mean()) / 400) if len(pts) else 0.01

    srv.scene.reset()
    _redraw_cloud()                                    # draws /cloud with current ceiling + point size
    if cams is not None:
        th = srv.scene.add_spline_catmull_rom("/trajectory", positions=cams,
                                              color=(45, 212, 191), line_width=3.0)
        th.visible = bool(_viser["show_traj"].value) if _viser.get("show_traj") is not None else True
        _viser["traj_handle"] = th


def _build_viser():
    # viser is embedded in the :8000 dashboard as the 3D canvas. GUI here is VIEW-only —
    # trajectory show/hide + point size (model load/unload, upload, job selection live in the
    # dashboard). Content is driven by _load_viser_cloud (startup newest + "在3D查看" → /view).
    import viser
    srv = viser.ViserServer(host="0.0.0.0", port=int(os.environ.get("VISER_PORT", "8080")))
    _viser["server"] = srv
    srv.scene.set_up_direction("+z")                # room is z-up → natural orbit / reset view

    cb = srv.gui.add_checkbox("显示相机轨迹", True)
    _viser["show_traj"] = cb

    @cb.on_update
    def _(_ev):
        h = _viser.get("traj_handle")
        if h is not None:
            h.visible = bool(cb.value)

    ps = srv.gui.add_slider("点大小", min=0.2, max=4.0, step=0.1, initial_value=1.0)  # ×base
    _viser["psize_mult"] = ps

    @ps.on_update
    def _(_ev):
        h = _viser.get("cloud_handle"); b = _viser.get("base_psize")
        if h is not None and b:
            h.point_size = b * float(ps.value)      # cheap: just resize, no re-upload

    # live 去天花板: fraction of floor→ceiling height kept from the floor (1.0 = keep all).
    # Filters the already-loaded cloud on the fly — no re-run.
    cs = srv.gui.add_slider("保留高度(去天花板)", min=0.3, max=1.0, step=0.05, initial_value=1.0)
    _viser["ceil_slider"] = cs

    @cs.on_update
    def _(_ev):
        _redraw_cloud()
    return srv


@app.on_event("startup")
def _startup():
    # rehydrate jobs from disk (survive restarts). Do NOT load the model (lazy).
    newest = None
    for jid in sorted(os.listdir(DATA)) if os.path.isdir(DATA) else []:
        d = os.path.join(DATA, jid)
        if not os.path.isdir(d):
            continue
        has_cloud = os.path.exists(os.path.join(d, "cloud.ply"))
        _jobs[jid] = {"status": "done" if has_cloud else "unknown", "out": d,
                      "t": os.path.getmtime(d), "products": sorted(os.listdir(d))}
        vext = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")   # locate source video (for rerun)
        vids = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f.lower().endswith(vext)]
        if vids:
            _jobs[jid]["vpath"] = os.path.join(d, vids[0]); _jobs[jid]["video"] = vids[0]
        mp = os.path.join(d, "meta.json")
        if os.path.exists(mp):
            try:
                m = json.load(open(mp)); _jobs[jid]["meta"] = m
                if m.get("fps"): _jobs[jid]["fps"] = m["fps"]
                cc = m.get("ceiling_cut")                 # meta stores the result dict when applied
                if isinstance(cc, dict):
                    _jobs[jid]["ceiling_cut"] = True
                    _jobs[jid]["ceiling_keep"] = cc.get("keep_frac", 0.85)
            except Exception: pass
        if has_cloud:
            newest = jid
    threading.Thread(target=_idle_watch, daemon=True).start()
    try:
        _build_viser()
        if newest:                              # show latest reconstruction (viz only, no GPU model)
            _load_viser_cloud(newest)
    except Exception as e:
        print("viser startup skipped:", e)
