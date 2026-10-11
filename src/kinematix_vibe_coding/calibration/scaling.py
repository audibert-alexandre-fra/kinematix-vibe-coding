"""Étalonnage géométrique initial à partir de la taille connue du sujet.

On mesure les longueurs de segments sur les keypoints 3D (médiane temporelle, moyenne
gauche/droite), puis on choisit l'échelle s des données telle que la hauteur debout du
modèle construit avec ces longueurs soit exactement la taille connue :
    H = s * (somme des segments verticaux mesurés) + (segments non observés, anthropométriques).
"""

from __future__ import annotations

import numpy as np

from ..models.humanoid import BodyDimensions


def _dist(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(a[valid] - b[valid], axis=-1)))


def measure_segments(m: dict[str, np.ndarray], valid: np.ndarray) -> dict[str, float]:
    """Longueurs médianes (unités des données) des segments observables."""
    def both(a: str, b: str) -> float:
        return 0.5 * (_dist(m[f"left_{a}"], m[f"left_{b}"], valid) + _dist(m[f"right_{a}"], m[f"right_{b}"], valid))

    mid_hip = (m["left_hip"] + m["right_hip"]) / 2
    mid_eye = (m["left_eye"] + m["right_eye"]) / 2
    return {
        "thigh": both("hip", "knee"),
        "shank": both("knee", "ankle"),
        "upper_arm": both("shoulder", "elbow"),
        "forearm": both("elbow", "wrist"),
        "hand": both("wrist", "middle_third_joint"),
        "foot_length": both("heel", "big_toe_tip"),
        "hip_width": _dist(m["left_hip"], m["right_hip"], valid),
        "shoulder_width": _dist(m["left_shoulder"], m["right_shoulder"], valid),
        "trunk": _dist(mid_hip, m["neck"], valid),
        "neck_to_eyes": _dist(m["neck"], mid_eye, valid),
    }


VERTICAL_MEASURED = ["shank", "thigh", "trunk", "neck_to_eyes"]


def initial_dimensions(segments: dict[str, float], height: float) -> tuple[BodyDimensions, float]:
    """Renvoie (dimensions initiales en m, échelle s à appliquer aux données)."""
    prior = BodyDimensions.from_height(height)
    A = sum(segments[k] for k in VERTICAL_MEASURED)
    B = prior.ankle_height + prior.eyes_to_top
    s = (height - B) / A
    trunk = s * segments["trunk"]
    ratio = prior.pelvis_to_lumbar / (prior.pelvis_to_lumbar + prior.torso)
    d = BodyDimensions(
        ankle_height=prior.ankle_height, shank=s * segments["shank"], thigh=s * segments["thigh"],
        hip_width=s * segments["hip_width"], pelvis_to_lumbar=ratio * trunk, torso=(1 - ratio) * trunk,
        shoulder_width=s * segments["shoulder_width"], shoulder_drop=prior.shoulder_drop,
        neck_to_eyes=s * segments["neck_to_eyes"], eyes_to_top=prior.eyes_to_top,
        upper_arm=s * segments["upper_arm"], forearm=s * segments["forearm"], hand=s * segments["hand"],
        foot_length=s * segments["foot_length"], heel_back=prior.heel_back,
    )
    return d, float(s)
