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


# segment -> keypoints nécessaires
SEGMENTS: dict[str, tuple[str, ...]] = {
    "thigh": ("hip", "knee"), "shank": ("knee", "ankle"), "upper_arm": ("shoulder", "elbow"),
    "forearm": ("elbow", "wrist"), "hand": ("wrist", "middle_third_joint"), "foot_length": ("heel", "big_toe_tip"),
    "hip_width": ("hip",), "shoulder_width": ("shoulder",), "trunk": ("hip", "neck"), "neck_to_eyes": ("neck", "eye"),
}


def _needed(seg: str) -> set[str]:
    out = set()
    for k in SEGMENTS[seg]:
        out |= {k} if k == "neck" else {f"left_{k}", f"right_{k}"}
    return out


def measure_segments(m: dict[str, np.ndarray], valid: np.ndarray, available: set[str] | None = None) -> dict[str, float]:
    """Longueurs médianes (unités des données) des segments observables.

    `available` : keypoints réellement fournis par le modèle de pose (les autres segments sont
    omis et seront déduits des proportions anthropométriques)."""
    def both(a: str, b: str) -> float:
        return 0.5 * (_dist(m[f"left_{a}"], m[f"left_{b}"], valid) + _dist(m[f"right_{a}"], m[f"right_{b}"], valid))

    mid_hip = (m["left_hip"] + m["right_hip"]) / 2
    out = {}
    for seg, kp in SEGMENTS.items():
        if available is not None and not _needed(seg) <= available:
            continue
        if seg == "hip_width":
            out[seg] = _dist(m["left_hip"], m["right_hip"], valid)
        elif seg == "shoulder_width":
            out[seg] = _dist(m["left_shoulder"], m["right_shoulder"], valid)
        elif seg == "trunk":
            out[seg] = _dist(mid_hip, m["neck"], valid)
        elif seg == "neck_to_eyes":
            out[seg] = _dist(m["neck"], (m["left_eye"] + m["right_eye"]) / 2, valid)
        else:
            out[seg] = both(*kp)
    return out


VERTICAL_MEASURED = ["shank", "thigh", "trunk", "neck_to_eyes"]


PRIOR_KEY = {"trunk": ("pelvis_to_lumbar", "torso")}


def clamp_to_prior(segments: dict[str, float], height: float, tol: float = 0.3) -> dict[str, float]:
    """Borne chaque segment mesuré à +-tol de sa proportion anthropométrique, après une mise à
    l'échelle globale (médiane des rapports) : protège contre un keypoint mal défini.
    Les segments non mesurés sont complétés par leur proportion anthropométrique."""
    prior = BodyDimensions.from_height(height)
    ref = {k: sum(getattr(prior, n) for n in PRIOR_KEY.get(k, (k,))) for k in SEGMENTS}
    g = float(np.median([segments[k] / ref[k] for k in segments]))
    out = {k: float(np.clip(v, (1 - tol) * g * ref[k], (1 + tol) * g * ref[k])) for k, v in segments.items()}
    for k in SEGMENTS:
        out.setdefault(k, g * ref[k])
    return out


def initial_dimensions(segments: dict[str, float], height: float) -> tuple[BodyDimensions, float]:
    """Renvoie (dimensions initiales en m, échelle s à appliquer aux données)."""
    prior = BodyDimensions.from_height(height)
    segments = clamp_to_prior(segments, height)
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
