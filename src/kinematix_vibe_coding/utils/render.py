"""Projection du squelette MuJoCo sur la vidéo d'origine."""

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
