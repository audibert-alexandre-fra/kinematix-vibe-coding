"""Pipeline complète : vidéo -> pose 3D -> calibration -> IK MuJoCo anti-glissement -> sorties."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .calibration import FOOT_MARKERS, MarkerSet, calibrate, estimate_world_transform, select_frames
from .calibration.scaling import initial_dimensions, measure_segments
from .inverse_kinematix import IKConfig, MarkerIK, root_from_markers
from .inverse_kinematix.angles import joint_angles, save_angles_csv
from .inverse_kinematix.contacts import ContactConfig, contact_extras, detect_contacts, foot_sliding
from .models import PoseSequence, get_backend
from .models.humanoid import Humanoid
from .utils.filters import interpolate_invalid, lowpass


@dataclass
class PipelineConfig:
    height: float = 1.75
    backend: str = "instanthmr"
    backend_kwargs: dict = field(default_factory=dict)
    root_cutoff_hz: float = 3.0      # filtrage de la trajectoire globale (profondeur bruitée)
    local_cutoff_hz: float = 6.0     # filtrage de la pose locale
    calib_frames: int = 40
    calib_rounds: int = 3
    reproj_weight: float = 10.0      # contrainte 2D (meilleure cohérence multi-vues, cf. rapport)
    anti_sliding: bool = True
    render: bool = True
    draw_keypoints: bool = False
    contact: ContactConfig = field(default_factory=ContactConfig)


@dataclass
class PipelineResult:
    qpos: np.ndarray
    angles: dict[str, np.ndarray]
    times: np.ndarray
    summary: dict
    out_dir: Path


def filter_markers(X: np.ndarray, valid: np.ndarray, names: list[str], fps: float, cfg: PipelineConfig):
    """Filtrage zero-phase : trajectoire du bassin (basse fréquence) + pose relative."""
    X = interpolate_invalid(X, valid)
    li, ri = names.index("left_hip"), names.index("right_hip")
    root = X[:, [li, ri]].mean(1, keepdims=True)
    return lowpass(root, fps, cfg.root_cutoff_hz) + lowpass(X - root, fps, cfg.local_cutoff_hz)


def get_pose_sequence(video: str | Path, cfg: PipelineConfig, cache: Path | None) -> PoseSequence:
    if cache is not None and cache.exists():
        return PoseSequence.load(cache)
    seq = get_backend(cfg.backend, **cfg.backend_kwargs).process_video(video)
    if cache is not None:
        seq.save(cache)
    return seq


def run_pipeline(video: str | Path, out_dir: str | Path, cfg: PipelineConfig | None = None,
                 verbose: bool = True) -> PipelineResult:
    cfg = cfg or PipelineConfig()
    video = Path(video)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    log = print if verbose else (lambda *a, **k: None)

    # 1. pose 3D --------------------------------------------------------------------------
    t0 = time.perf_counter()
    seq = get_pose_sequence(video, cfg, out / f"pose_{cfg.backend}.npz")
    timings["pose_estimation_s"] = seq.timings.get("detection_s", 0) + seq.timings.get("pose_s", 0)
    timings["pose_loading_s"] = time.perf_counter() - t0
    T, fps = len(seq), seq.fps
    log(f"[{video.name}] {T} images, {fps:.1f} fps, pose {cfg.backend}: {timings['pose_estimation_s']:.1f} s")

    # 2. marqueurs + filtrage -----------------------------------------------------------------
    t0 = time.perf_counter()
    ms = MarkerSet.default()
    Xc, uv_obs, W = ms.extract(seq)
    Xc = filter_markers(Xc, seq.valid, ms.names, fps, cfg)
    mk = {n: Xc[:, i] for i, n in enumerate(ms.names)}

    # 3. étalonnage géométrique initial (taille connue) + repère monde (sol) -------------------
    dims0, s0 = initial_dimensions(measure_segments(mk, seq.valid), cfg.height)
    world = estimate_world_transform(mk, FOOT_MARKERS, seq.intrinsics.K, s0)
    Xw = world.to_world(Xc)

    # 4. calibration automatique MuJoCo ---------------------------------------------------------
    hum = Humanoid(dims0)
    frames = select_frames(seq.valid, Xw, cfg.calib_frames)
    calib = calibrate(hum, ms.names, Xw, W, cfg.height, frames, rounds=cfg.calib_rounds)
    world = world.with_scale(s0 * calib.scale_ratio)
    Xw = world.to_world(Xc)
    timings["calibration_s"] = time.perf_counter() - t0
    log(f"  calibration: échelle {world.scale:.3f}, rmse marqueurs {calib.rmse_before * 100:.1f} -> "
        f"{calib.rmse_after * 100:.1f} cm ({timings['calibration_s']:.1f} s)")

    # 5. IK passe 1 (sans contacts) ------------------------------------------------------------
    t0 = time.perf_counter()
    ik_cfg = IKConfig(reproj_weight=cfg.reproj_weight)
    ik = MarkerIK(hum, ms.names, ik_cfg)
    ik.update_ground_offsets()
    cam = world.camera_pose()
    q_init = root_from_markers(hum, ms.names, Xw[0])
    qpos, _ = ik.solve_sequence(Xw, W, q_init=q_init, obs2d=uv_obs, w2d=W, cam=cam)

    # recalage du sol : le point de pied le plus bas doit toucher z = 0 en appui
    foot_ids = [hum.site_id(n) for n in FOOT_MARKERS]
    foot = hum.sites_world(qpos)[:, foot_ids]
    lowest = (foot[..., 2] - ik.ground_offsets).min(1)
    dz = float(np.percentile(lowest, 15))
    if abs(dz) > 0.005:
        world = world.shifted_ground(dz)
        Xw = world.to_world(Xc)
        cam = world.camera_pose()
        qpos[:, 2] -= dz
        foot[..., 2] -= dz
    timings["ik_pass1_s"] = time.perf_counter() - t0

    # 6. contacts + IK passe 2 anti-glissement ---------------------------------------------------
    t0 = time.perf_counter()
    groups = [[i for i, n in enumerate(FOOT_MARKERS) if n.startswith(side)] for side in ("left", "right")]
    contacts = detect_contacts(foot, ik.ground_offsets, fps, cfg.contact, groups, root_pos=qpos[:, :3])
    sliding_before = foot_sliding(foot, contacts.mask, contacts.belt_velocity, fps)
    if cfg.anti_sliding and contacts.mask.any():
        extras = contact_extras(contacts, foot_ids, cfg.contact.weight)
        qpos, _ = ik.solve_sequence(Xw, W, q_init=qpos[0], extras=extras, obs2d=uv_obs, w2d=W, cam=cam)
        foot = hum.sites_world(qpos)[:, foot_ids]
    sliding_after = foot_sliding(foot, contacts.mask, contacts.belt_velocity, fps)
    timings["ik_pass2_s"] = time.perf_counter() - t0

    # 7. sorties ------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    sites = hum.sites_world(qpos)
    sid = [hum.site_id(n) for n in ms.names]
    err3d = np.linalg.norm(sites[:, sid] - Xw, axis=-1)
    uv_model = cam.project(sites)
    err2d = np.linalg.norm(uv_model[:, sid] - uv_obs, axis=-1)
    angles = joint_angles(hum, qpos)
    times = np.arange(T) / fps
    np.save(out / "qpos.npy", qpos)
    save_angles_csv(out / "joint_angles.csv", times, angles)
    np.savez_compressed(out / "trajectories.npz", qpos=qpos, sites=sites, site_names=np.array(hum.site_names),
                        markers_world=Xw, marker_names=np.array(ms.names), contacts=contacts.mask,
                        uv_model=uv_model, uv_obs=uv_obs)
    with open(out / "model.xml", "w") as f:
        f.write(hum.rebuilt().spec.to_xml())
    if cfg.render:
        from .utils.render import render_overlay, render_skeleton_overlay

        label = f"{video.parent.name}/{video.stem} - squelette MuJoCo ({cfg.backend})"
        render_skeleton_overlay(video, out / f"overlay_{video.stem}.mp4", hum.rebuilt(), qpos, cam,
                                contacts.mask, foot_ids, label=label)
        if cfg.draw_keypoints:
            render_overlay(video, out / f"debug_markers_{video.stem}.mp4", uv_model, hum.site_names,
                           contacts.mask, FOOT_MARKERS, uv_obs, label=label)
    timings["outputs_s"] = time.perf_counter() - t0
    timings["total_s"] = timings["pose_estimation_s"] + sum(
        v for k, v in timings.items() if k not in ("pose_estimation_s", "pose_loading_s"))

    summary = {
        "video": str(video), "backend": cfg.backend, "n_frames": T, "fps": fps, "height_m": cfg.height,
        "data_scale": world.scale, "dimensions_m": hum.dims.to_dict(),
        "model_standing_height_m": hum.dims.standing_height(),
        "calibration_rmse_cm": {"before": calib.rmse_before * 100, "after": calib.rmse_after * 100},
        "ik_marker_error_cm": {"median": float(np.median(err3d[W > 0]) * 100),
                               "mean": float(err3d[W > 0].mean() * 100)},
        "reprojection_error_px": {"median": float(np.median(err2d[W > 0])), "mean": float(err2d[W > 0].mean())},
        "treadmill_belt_velocity_m_s": contacts.belt_velocity.tolist(),
        "foot_sliding_before": sliding_before, "foot_sliding_after": sliding_after,
        "contact_config": asdict(cfg.contact),
        "timings_s": timings,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  IK: erreur marqueurs médiane {summary['ik_marker_error_cm']['median']:.1f} cm, "
        f"reprojection {summary['reprojection_error_px']['median']:.1f} px ; glissement "
        f"{sliding_before['sliding_speed_mean_cm_s']:.1f} -> {sliding_after['sliding_speed_mean_cm_s']:.1f} cm/s ; "
        f"total {timings['total_s']:.1f} s")
    return PipelineResult(qpos=qpos, angles=angles, times=times, summary=summary, out_dir=out)
