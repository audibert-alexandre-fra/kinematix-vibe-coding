"""Projection du squelette MuJoCo sur la vidéo d'origine.

Deux rendus : `render_skeleton_overlay` (os du modèle MuJoCo projetés en volume : capsules,
tête, pieds, articulations) et `render_overlay` (segments fins entre marqueurs, debug)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..video import VideoWriter, iter_frames, read_video_info

# segments du squelette affiché (noms des sites MuJoCo)
SKELETON_SEGMENTS: list[tuple[str, str]] = [
    ("pelvis_center", "neck"), ("neck", "nose"), ("left_hip", "right_hip"),
    ("left_shoulder", "right_shoulder"),
]
for _s in ("left", "right"):
    SKELETON_SEGMENTS += [
        (f"{_s}_hip", f"{_s}_knee"), (f"{_s}_knee", f"{_s}_ankle"), (f"{_s}_ankle", f"{_s}_heel"),
        (f"{_s}_heel", f"{_s}_big_toe_tip"), (f"{_s}_ankle", f"{_s}_big_toe_tip"),
        (f"{_s}_shoulder", f"{_s}_elbow"), (f"{_s}_elbow", f"{_s}_wrist"),
        (f"{_s}_wrist", f"{_s}_middle_third_joint"),
    ]

COLORS = {"left": (255, 140, 30), "right": (40, 40, 230), "center": (60, 220, 60)}  # BGR


def _color(a: str, b: str) -> tuple[int, int, int]:
    if a.startswith("left") and b.startswith("left"):
        return COLORS["left"]
    if a.startswith("right") and b.startswith("right"):
        return COLORS["right"]
    return COLORS["center"]


def draw_skeleton(img: np.ndarray, uv: dict[str, np.ndarray], thickness: int = 3,
                  contacts: dict[str, bool] | None = None, keypoints2d: np.ndarray | None = None) -> np.ndarray:
    if keypoints2d is not None:
        for p in keypoints2d:
            cv2.circle(img, (int(p[0]), int(p[1])), 2, (200, 200, 200), -1, cv2.LINE_AA)
    for a, b in SKELETON_SEGMENTS:
        pa, pb = uv[a], uv[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), _color(a, b), thickness, cv2.LINE_AA)
    for n, p in uv.items():
        if not np.isfinite(p).all():
            continue
        c = (0, 255, 255)
        if contacts is not None and contacts.get(n, False):
            c = (0, 255, 0)
            cv2.circle(img, (int(p[0]), int(p[1])), thickness + 4, c, 2, cv2.LINE_AA)
        cv2.circle(img, (int(p[0]), int(p[1])), thickness + 1, c, -1, cv2.LINE_AA)
    return img


def render_overlay(video_path: str | Path, out_path: str | Path, uv_sites: np.ndarray, site_names: list[str],
                   contact_mask: np.ndarray | None = None, contact_names: list[str] | None = None,
                   keypoints2d: np.ndarray | None = None, label: str = "") -> str:
    """Écrit une vidéo avec le squelette du modèle MuJoCo projeté. `uv_sites` : (T, nsite, 2)."""
    info = read_video_info(video_path)
    used = {n for seg in SKELETON_SEGMENTS for n in seg}
    idx = {n: site_names.index(n) for n in used}
    thick = max(2, int(round(min(info.width, info.height) / 300)))
    with VideoWriter(out_path, info.fps, info.width, info.height) as w:
        for t, frame in enumerate(iter_frames(video_path, rgb=False)):
            if t >= len(uv_sites):
                break
            uv = {n: uv_sites[t, i] for n, i in idx.items()}
            contacts = None
            if contact_mask is not None and contact_names is not None:
                contacts = {n: bool(contact_mask[t, k]) for k, n in enumerate(contact_names)}
            draw_skeleton(frame, uv, thick, contacts, None if keypoints2d is None else keypoints2d[t])
            if label:
                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            w.write(frame)
    return str(out_path)


def _side_color(body_name: str) -> tuple[int, int, int]:
    if body_name.endswith("_l"):
        return COLORS["left"]
    if body_name.endswith("_r"):
        return COLORS["right"]
    return (90, 200, 90)


def project_bones(model, data, cam) -> list[tuple[float, str, tuple]]:
    """Primitives 2D (profondeur, type, paramètres) des géométries du modèle pour l'état courant de `data`."""
    import mujoco

    prims = []
    K = cam.K
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        if body == "world":
            continue
        gtype = model.geom_type[g]
        pos = data.geom_xpos[g]
        R = data.geom_xmat[g].reshape(3, 3)
        color = _side_color(body)
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r, hl = model.geom_size[g, 0], model.geom_size[g, 1]
            a, b = pos - R[:, 2] * hl, pos + R[:, 2] * hl
            pc = cam.to_cam(np.stack([a, b]))
            if (pc[:, 2] < 0.05).any():
                continue
            uv = cam.project(np.stack([a, b]))
            rad = K[0, 0] * r / cam.scale / pc[:, 2].mean()
            prims.append((float(pc[:, 2].mean()), "capsule", (uv, rad, color)))
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            pc = cam.to_cam(pos[None])[0]
            if pc[2] < 0.05:
                continue
            prims.append((float(pc[2]), "sphere", (cam.project(pos[None])[0], K[0, 0] * model.geom_size[g, 0]
                                                   / cam.scale / pc[2], (205, 220, 240))))
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            h = model.geom_size[g]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
            pw = pos + corners @ R.T
            pc = cam.to_cam(pw)
            if (pc[:, 2] < 0.05).any():
                continue
            prims.append((float(pc[:, 2].mean()), "box", (cam.project(pw), color)))
    prims.sort(key=lambda p: -p[0])  # du plus loin au plus proche
    return prims


def draw_bones(img: np.ndarray, prims, joints_uv: np.ndarray | None = None, alpha: float = 0.55) -> np.ndarray:
    layer = img.copy()
    for _, kind, par in prims:
        if kind == "capsule":
            uv, rad, color = par
            p0, p1 = tuple(int(v) for v in uv[0]), tuple(int(v) for v in uv[1])
            t = max(2, int(2 * rad))
            cv2.line(layer, p0, p1, color, t, cv2.LINE_AA)
            cv2.line(layer, p0, p1, tuple(int(c * 0.55) for c in color), max(1, t // 6), cv2.LINE_AA)
        elif kind == "sphere":
            c, rad, color = par
            cv2.circle(layer, (int(c[0]), int(c[1])), max(3, int(rad)), color, -1, cv2.LINE_AA)
        else:
            uv, color = par
            hull = cv2.convexHull(uv.astype(np.int32))
            cv2.fillConvexPoly(layer, hull, color, cv2.LINE_AA)
    out = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
    if joints_uv is not None:
        for p in joints_uv:
            cv2.circle(out, (int(p[0]), int(p[1])), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (int(p[0]), int(p[1])), 4, (40, 40, 40), 1, cv2.LINE_AA)
    return out


def render_skeleton_overlay(video_path: str | Path, out_path: str | Path, hum, qpos: np.ndarray, cam,
                            contact_mask: np.ndarray | None = None, contact_site_ids: list[int] | None = None,
                            label: str = "") -> str:
    """Vidéo avec le squelette 3D du modèle MuJoCo (os en volume) projeté sur chaque image."""
    import mujoco

    info = read_video_info(video_path)
    m, d = hum.model, hum.data
    joint_bodies = [b for b in range(1, m.nbody)]
    with VideoWriter(out_path, info.fps, info.width, info.height) as w:
        for t, frame in enumerate(iter_frames(video_path, rgb=False)):
            if t >= len(qpos):
                break
            d.qpos[:] = qpos[t]
            mujoco.mj_kinematics(m, d)
            prims = project_bones(m, d, cam)
            joints = cam.project(d.xpos[joint_bodies])
            frame = draw_bones(frame, prims, joints)
            if contact_mask is not None and contact_site_ids is not None:
                for k, sid in enumerate(contact_site_ids):
                    if contact_mask[t, k]:
                        p = cam.project(d.site_xpos[sid][None])[0]
                        cv2.circle(frame, (int(p[0]), int(p[1])), 7, (0, 255, 0), 2, cv2.LINE_AA)
            if label:
                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            w.write(frame)
    return str(out_path)
