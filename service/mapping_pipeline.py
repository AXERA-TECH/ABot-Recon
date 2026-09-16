#!/usr/bin/env python3
"""ABot-Recon 建图流水线 — 一个视频 → 点云 + 户型俯视图 + 3D 视图。

推理核心是 ABot-Recon(acvlab/ABot-Recon):流式前馈重建,直接输出 c2w 位姿 + 世界坐标点图 +
置信度。encoder/decoder/heads 跑在 Axera NPU 上,位姿头和后处理是 numpy(abot_axera 包,无 torch)。

产物(写入 out_dir):
  recon.npz     对齐后的相机中心 cams[N,3] + 原始 poses[N,4,4](网页轨迹用)
  cloud.ply     真实 RGB 点云(置信过滤 + 地面对齐 + 可选去天花板)
  cloud_viz.ply 按时间上色点云(备用)
  splats.ply    高斯 splat(3DGS PLY 格式,免训练,由点图法线/间距生成;网页 3D 默认用它)
  floorplan.png 俯视户型图(相机轨迹 PCA 基底 + magma hexbin)
  view_ob.png   3D 斜视(真实 RGB, 立正)
  view_top.png  俯视时间色
  meta.json     帧数/耗时/点数

CLI:  mapping_pipeline.py <video> <out_dir> [--fps 8] [--ceiling-cut] [--ceiling-keep 0.85]
"""
import os, sys, time, json, argparse

def load_ready_model(model_id=None, use_sdpa=True, device=None):
    """Build the NPU-backed ABot-Recon (abot_axera.backend.AbotRecon). All settings come from env:
    ABOT_MODELS / ABOT_MODEL_SUFFIX (axmodels), ABOT_DEVICE_ID, ABOT_RUNNER, ABOT_DEVICE,
    ABOT_DELIVERY (host_pose_head/) or ABOT_POSE_WEIGHTS / ABOT_POSE_CONFIG. `model_id` is ignored."""
    from abot_axera.backend import build_abot_recon
    deliv = os.environ.get("ABOT_DELIVERY", "/home/axera/abot650/ABot-Recon_AX650_交付包_20260907")
    return build_abot_recon(
        model_dir=os.environ.get("ABOT_MODELS", "/home/axera/ABot-Recon"),
        suffix=os.environ.get("ABOT_MODEL_SUFFIX", "_kitti02"),
        device_id=int(os.environ.get("ABOT_DEVICE_ID", "0")),
        pose_weights=os.environ.get("ABOT_POSE_WEIGHTS",
                                    os.path.join(deliv, "host_pose_head", "pose_head.safetensors")),
        pose_config=os.environ.get("ABOT_POSE_CONFIG",
                                   os.path.join(deliv, "host_pose_head", "pose_head_config.json")),
    )


def _progress():
    try:
        from abot_axera import progress
        return progress
    except Exception:
        class _Null:
            def __getattr__(self, _):
                return lambda *a, **k: None
        return _Null()


def _grav_basis(cams):
    """world→floor basis (columns e1,e2,up) from PCA of the camera trajectory. Smallest-variance
    axis of the camera centers is the floor normal (up); largest is the main walking axis (e1).
    Gives a clean top-down (the camera down-axis would tilt with pitch and smear walls)."""
    import numpy as np
    cc = cams - cams.mean(0)
    _, _, Vt = np.linalg.svd(cc, full_matrices=False)
    up = Vt[2]; e1 = Vt[0]
    e2 = np.cross(up, e1); e2 /= np.linalg.norm(e2) + 1e-9
    return np.stack([e1, e2, up], 1).astype(np.float32)


def run(video, out_dir, fps=8, ceiling_cut=False, ceiling_keep=0.85,
        model=None, model_id=None, device=None, should_stop=None):
    import numpy as np
    from abot_axera.video import open_video
    from abot_axera.pointcloud import VoxelGrid, write_ply
    from abot_axera.splats import frame_geometry, build_gaussians, write_splat_ply
    prog = _progress()
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    meta = {"video": os.path.basename(video), "fps": fps, "engine": "ABot-Recon"}

    # ---- frames: uniform fps sampling straight from the decoder (hardware when available) ----
    prog.clear(); prog.set_phase("extract")
    src = open_video(video, fps, device_id=int(os.environ.get("ABOT_DEVICE_ID", "0")))
    print(f"[abot] decoder={src.name} interval={src.interval} ~{src.total} frames @ fps={fps} from {os.path.basename(video)}", flush=True)

    # ---- inference (reuse a pre-loaded model if given) ----
    if model is None:
        model = load_ready_model(model_id, True, device)
    t_inf = time.time()
    result = model.infer(src, output_world_points=True, output_confidence=True, output_colors=True,
                         total=src.total, should_stop=should_stop)
    meta["decoder"] = src.name                       # AutoSource may have fallen back to cv2
    meta["frames"] = int(result.camera_poses.shape[0]); meta["infer_s"] = round(time.time() - t_inf, 1)

    poses = result.camera_poses.astype(np.float32)   # [N,4,4] c2w
    wp = result.world_points                         # [M,H,W,3] world
    conf = result.confidence                         # [M,H,W] or None
    col = result.colors                              # [M,H,W,3] uint8 (same frames the model saw)
    dense = list(range(len(poses)))

    cams = poses[:, :3, 3].astype(np.float32)
    M, H, W = wp.shape[0], wp.shape[1], wp.shape[2]
    st = 2
    P = wp[:, ::st, ::st, :].reshape(-1, 3).astype(np.float32)
    C = (col[:, ::st, ::st, :].reshape(-1, 3).astype(np.float32) / 255.0).clip(0, 1)
    NRM, SPC = frame_geometry(wp, cams, st)                       # surfel normals + spacing per sample
    NRM = NRM.reshape(-1, 3); SPC = SPC.reshape(-1, 2)
    hh, ww = (H + st - 1) // st, (W + st - 1) // st
    T = np.repeat(np.arange(M, dtype=np.float32), hh * ww)
    F = conf[:, ::st, ::st].reshape(-1) if conf is not None else np.ones(len(P), np.float32)

    m = np.isfinite(P).all(1)
    if conf is not None:
        pos = F > 0
        m &= pos & (F >= np.percentile(F[pos], 45))     # conf gate (45th pctile)
    P, C, T, F, NRM, SPC = P[m], C[m], T[m], F[m], NRM[m], SPC[m]
    meta["points_raw"] = int(len(P))

    # gravity-align to the floor basis, then orient +z = up via the camera down-axis (c2w col1),
    # rotating 180° about x if inverted (proper rotation) so the room stands upright everywhere.
    Bg = _grav_basis(cams); P = (P @ Bg).astype(np.float32); cams = cams @ Bg; NRM = (NRM @ Bg).astype(np.float32)
    if float(poses[:, :3, 1].mean(0) @ Bg[:, 2]) > 0:
        P[:, 1] *= -1; P[:, 2] *= -1; cams[:, 1] *= -1; cams[:, 2] *= -1; NRM[:, 1] *= -1; NRM[:, 2] *= -1

    # drop far global-drift geometry: keep only points within the WALKED area (camera-trajectory
    # xy bbox + margin). ABot's sequential composition can fling a chunk of geometry far from the
    # path (no loop closure); without this the floorplan/3D frame around empty space. The camera
    # path covers the real rooms, so bbox+margin keeps the rooms and cuts the drift.
    if len(P):
        cxy0, cxy1 = cams[:, :2].min(0), cams[:, :2].max(0)
        mrg = 0.35 * float((cxy1 - cxy0).max()) + 1e-6
        inb = ((P[:, 0] > cxy0[0] - mrg) & (P[:, 0] < cxy1[0] + mrg) &
               (P[:, 1] > cxy0[1] - mrg) & (P[:, 1] < cxy1[1] + mrg))
        meta["walked_area_kept"] = [int(inb.sum()), int(len(inb))]
        P, C, T, F, NRM, SPC = P[inb], C[inb], T[inb], F[inb], NRM[inb], SPC[inb]

    # optional ceiling removal: keep a fraction of the floor→ceiling span from the floor (+z up)
    if ceiling_cut and len(P):
        up_h = P[:, 2]; lo, hi = np.percentile(up_h, 1), np.percentile(up_h, 99)
        keep = up_h <= lo + float(ceiling_keep) * (hi - lo)
        meta["ceiling_cut"] = {"keep_frac": round(float(ceiling_keep), 2),
                               "kept": int(keep.sum()), "of": int(len(keep))}
        P, C, T, F, NRM, SPC = P[keep], C[keep], T[keep], F[keep], NRM[keep], SPC[keep]

    np.savez_compressed(os.path.join(out_dir, "recon.npz"),
                        cams=cams.astype(np.float32), poses=poses)

    import matplotlib.cm as cm
    tnorm = (T - T.min()) / max(1.0, float(T.max() - T.min()))
    Ctime = cm.turbo(tnorm)[:, :3].astype(np.float32)

    # ---- point clouds: cloud.ply (真实 RGB, 网页主视图) + cloud_viz.ply (时间色, 备用) ----
    # voxel size from the ROBUST (1–99 pctile) extent, not full ptp: ABot's global drift throws
    # a few points far out and inflates full-ptp ~5x, which would over-coarsen the cloud to ~200k.
    span = float((np.percentile(P, 99, 0) - np.percentile(P, 1, 0)).mean())
    vox = max(1e-3, span / 550)
    grid = VoxelGrid(P, vox); Pv = grid.mean(P)                # one voxel assignment, two color sets
    write_ply(os.path.join(out_dir, "cloud.ply"), Pv, grid.mean(C))
    write_ply(os.path.join(out_dir, "cloud_viz.ply"), Pv, grid.mean(Ctime))
    meta["cloud_points"] = int(len(Pv))

    # ---- Gaussian splats (surfels): same cloud, coarser grid if it would exceed MAP_SPLATS_MAX ----
    smax = int(os.environ.get("MAP_SPLATS_MAX", "1500000"))
    gs, vox_s = grid, vox
    for _ in range(4):                                          # points sit on surfaces: count ~ 1/vox^2
        if len(gs) <= smax:
            break
        vox_s *= float(np.sqrt(len(gs) / smax)) * 1.05; gs = VoxelGrid(P, vox_s)
    g = build_gaussians(gs.mean(P), gs.mean(C), gs.mean(NRM), gs.mean(SPC), 0.95, voxel=vox_s)
    write_splat_ply(os.path.join(out_dir, "splats.ply"), g)
    meta["splats"] = int(len(g["centers"]))

    _render(out_dir, P, C, Ctime, cams, meta)

    meta["total_s"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print("products:", sorted(os.listdir(out_dir)))
    print("meta:", json.dumps(meta, ensure_ascii=False))
    prog.clear()
    return meta


def _render(out_dir, P, Crgb, Ctime, cams, meta):
    # P, cams already aligned to the floor basis: x=e1, y=e2, z=up (upright).
    import numpy as np, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    BG = "#0a0e13"
    x, y, hgt = P[:, 0], P[:, 1], P[:, 2]; cx_, cy_ = cams[:, 0], cams[:, 1]

    # ---- floorplan: top-down density (magma hexbin) + trajectory, clipped to 1–99 pctile ----
    xl, xh = np.percentile(x, 1), np.percentile(x, 99)
    yl, yh = np.percentile(y, 1), np.percentile(y, 99)
    k = (x > xl) & (x < xh) & (y > yl) & (y < yh)
    fig = plt.figure(figsize=(13, 9)); ax = fig.add_subplot(111); ax.set_facecolor(BG)
    ax.hexbin(x[k], y[k], gridsize=300, cmap="magma", bins="log", linewidths=0)
    ax.plot(cx_, cy_, color="#35d0c0", lw=2.4)
    ax.scatter(cx_[0], cy_[0], c="#7CFF6B", s=90, edgecolors="white", linewidths=1.2, zorder=5)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off"); fig.patch.set_facecolor(BG)
    fig.savefig(os.path.join(out_dir, "floorplan.png"), dpi=130, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # ---- 3D oblique, REAL RGB, upright ----
    sel = np.random.default_rng(0).choice(len(P), min(500000, len(P)), replace=False)
    Q, QC = P[sel], np.clip(Crgb[sel], 0, 1); lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
    fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c=QC, s=.45, linewidths=0)
    ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], color="#e6edf3", lw=1.1)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]))
    ax.view_init(elev=32, azim=-110); ax.axis("off")
    fig.savefig(os.path.join(out_dir, "view_ob.png"), dpi=120, facecolor=BG, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    # ---- top-down, time-colored ----
    fig = plt.figure(figsize=(11, 9)); ax = fig.add_subplot(111); ax.set_facecolor(BG)
    ax.scatter(x[k], y[k], c=Ctime[k], s=.4, linewidths=0)
    ax.plot(cx_, cy_, color="#e6edf3", lw=1.0)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off"); fig.patch.set_facecolor(BG)
    fig.savefig(os.path.join(out_dir, "view_top.png"), dpi=120, facecolor=BG, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out_dir")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--model", default=None, help="HF repo id or local checkpoint dir")
    ap.add_argument("--ceiling-cut", action="store_true")
    ap.add_argument("--ceiling-keep", type=float, default=0.85)
    args = ap.parse_args()
    run(args.video, args.out_dir, fps=args.fps, model_id=args.model,
        ceiling_cut=args.ceiling_cut, ceiling_keep=args.ceiling_keep)
