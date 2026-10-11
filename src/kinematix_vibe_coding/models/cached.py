"""Backends lisant des poses pré-calculées par des modèles tournant dans un environnement séparé
(tools/run_<modèle>.py) : MediaPipe Pose, MMPose RTMW3D, NLF.

Format commun du cache `weights/<backend>_cache/<sous-dossier>__<vidéo>.npz` :
    joints3d (T, J, 3) m, joints2d (T, J, 2) px, scores (T, J), boxes (T, 4) xyxy, frame_valid (T,),
    coords : "camera"          -> joints3d déjà en repère caméra (NLF)
             "root_relative"   -> 3D métrique centrée au bassin, axes caméra (MediaPipe world landmarks) :
                                  translation estimée par moindres carrés sur la reprojection 2D
             "relative_depth"  -> seule la profondeur relative z (m) est connue (RTMW3D) : profondeur
                                  absolue du bassin estimée par longueurs de segments anthropométriques
    focal, fps, image_size, inference_s, load_s.
Chaque squelette source est converti vers les noms MHR70 de nos marqueurs (moyenne de points
source possible) ; un point sans équivalent a une confiance nulle (ignoré).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from ..utils.filters import interpolate_invalid
from ..utils.skeleton import MHR70_JOINT_NAMES
from ..video import default_intrinsics, read_video_info
from .base import PoseSequence

Mapping = dict[str, tuple[tuple[str, ...], float]]


def _sided(pairs: dict[str, tuple[str, float]]) -> Mapping:
    """{"{s}_knee": ("{s}_knee", 1.0)} -> entrées gauche/droite (s = left/right)."""
    out: Mapping = {}
    for s in ("left", "right"):
        for dst, (src, w) in pairs.items():
            out[dst.format(s=s)] = (tuple(x.format(s=s) for x in src.split("+")), w)
    return out


# --- MediaPipe Pose (BlazePose, 33 points) ------------------------------------------------------
MEDIAPIPE33_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
MHR70_FROM_MEDIAPIPE: Mapping = {"nose": (("nose",), 1.0), "neck": (("left_shoulder", "right_shoulder"), 0.5)}
MHR70_FROM_MEDIAPIPE |= _sided({
    "{s}_eye": ("{s}_eye", 1.0), "{s}_ear": ("{s}_ear", 1.0), "{s}_shoulder": ("{s}_shoulder", 1.0),
    "{s}_elbow": ("{s}_elbow", 1.0), "{s}_wrist": ("{s}_wrist", 1.0), "{s}_hip": ("{s}_hip", 1.0),
    "{s}_knee": ("{s}_knee", 1.0), "{s}_ankle": ("{s}_ankle", 1.0), "{s}_heel": ("{s}_heel", 1.0),
    "{s}_big_toe_tip": ("{s}_foot_index", 0.7), "{s}_index_third_joint": ("{s}_index", 0.5),
    "{s}_pinky_third_joint": ("{s}_pinky", 0.5),
})

# --- COCO-WholeBody 133 (MMPose RTMW3D) : corps 0-22, visage 23-90, mains 91-132 ------------------
COCO_WB133_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel",
] + [f"face_{i}" for i in range(68)] + [
    f"{s}_hand_{i}" for s in ("left", "right") for i in range(21)  # 0 poignet, 5 index MCP, 9 majeur, 17 auriculaire
]
MHR70_FROM_COCO_WB: Mapping = {"nose": (("nose",), 1.0), "neck": (("left_shoulder", "right_shoulder"), 0.5)}
MHR70_FROM_COCO_WB |= _sided({
    "{s}_eye": ("{s}_eye", 1.0), "{s}_ear": ("{s}_ear", 1.0), "{s}_shoulder": ("{s}_shoulder", 1.0),
    "{s}_elbow": ("{s}_elbow", 1.0), "{s}_wrist": ("{s}_wrist", 1.0), "{s}_hip": ("{s}_hip", 1.0),
    "{s}_knee": ("{s}_knee", 1.0), "{s}_ankle": ("{s}_ankle", 1.0), "{s}_heel": ("{s}_heel", 1.0),
    "{s}_big_toe_tip": ("{s}_big_toe", 1.0), "{s}_small_toe_tip": ("{s}_small_toe", 1.0),
    "{s}_index_third_joint": ("{s}_hand_5", 0.5), "{s}_middle_third_joint": ("{s}_hand_9", 0.5),
    "{s}_pinky_third_joint": ("{s}_hand_17", 0.5),
})

# --- NLF : 24 articulations SMPL + sommets du maillage (tools/run_nlf.py) --------------------------
SMPL24_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle",
    "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]
NLF_NAMES = SMPL24_NAMES + ["nose", "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe",
                            "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"]
MHR70_FROM_NLF: Mapping = {"nose": (("nose",), 1.0), "neck": (("neck",), 1.0)}
MHR70_FROM_NLF |= _sided({
    "{s}_eye": ("{s}_eye", 1.0), "{s}_ear": ("{s}_ear", 1.0), "{s}_shoulder": ("{s}_shoulder", 1.0),
    "{s}_elbow": ("{s}_elbow", 1.0), "{s}_wrist": ("{s}_wrist", 1.0), "{s}_hip": ("{s}_hip", 1.0),
    "{s}_knee": ("{s}_knee", 1.0), "{s}_ankle": ("{s}_ankle", 1.0), "{s}_heel": ("{s}_heel", 1.0),
    "{s}_big_toe_tip": ("{s}_big_toe", 1.0), "{s}_small_toe_tip": ("{s}_small_toe", 1.0),
    "{s}_middle_third_joint": ("{s}_hand", 0.5),
})


def to_mhr70(src_names: list[str], mapping: Mapping, j3: np.ndarray, j2: np.ndarray, scores: np.ndarray):
    """Convertit (T, J_src, ·) vers les 70 points MHR70 (moyenne des points source, confiance x poids)."""
    T = len(j3)
    o3, o2, oc = np.zeros((T, 70, 3)), np.zeros((T, 70, 2)), np.zeros((T, 70))
    for dst, (srcs, w) in mapping.items():
        k = [src_names.index(s) for s in srcs]
        i = MHR70_JOINT_NAMES.index(dst)
        o3[:, i], o2[:, i] = j3[:, k].mean(1), j2[:, k].mean(1)
        oc[:, i] = w * scores[:, k].min(1)
    return o3, o2, oc


def solve_translation(rel3d: np.ndarray, uv: np.ndarray, K: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Translation t (3,) telle que la projection de rel3d + t colle aux points 2D (moindres carrés
    linéaires pondérés) : fx (X + tx) - (u - cx)(Z + tz) = 0, idem en y."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    du, dv = uv[:, 0] - cx, uv[:, 1] - cy
    n = len(rel3d)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[:n, 0], A[:n, 2], b[:n] = fx, -du, du * rel3d[:, 2] - fx * rel3d[:, 0]
    A[n:, 1], A[n:, 2], b[n:] = fy, -dv, dv * rel3d[:, 2] - fy * rel3d[:, 1]
    sw = np.sqrt(np.concatenate([w, w]))
    return np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]


def backproject(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    rays = np.stack([(uv[..., 0] - K[0, 2]) / K[0, 0], (uv[..., 1] - K[1, 2]) / K[1, 1], np.ones(uv.shape[:-1])], -1)
    return rays * depth[..., None]


@lru_cache(maxsize=1)
def reference_bones(height: float = 1.70) -> dict[tuple[str, str], float]:
    """Longueurs de référence (m) entre sites MHR70 du modèle MuJoCo neutre (pour fixer l'échelle)."""
    from .humanoid import BodyDimensions, Humanoid

    hum = Humanoid(BodyDimensions.from_height(height))
    p = hum.sites_world(hum.model.qpos0.copy())[0]
    names = hum.site_names
    pairs = [("{s}_hip", "{s}_knee"), ("{s}_knee", "{s}_ankle"), ("{s}_shoulder", "{s}_elbow"),
             ("{s}_elbow", "{s}_wrist"), ("{s}_shoulder", "{s}_hip"), ("{s}_heel", "{s}_big_toe_tip")]
    out = {}
    for s in ("left", "right"):
        for a, b in pairs:
            a, b = a.format(s=s), b.format(s=s)
            out[(a, b)] = float(np.linalg.norm(p[names.index(a)] - p[names.index(b)]))
    for a, b in (("left_hip", "right_hip"), ("left_shoulder", "right_shoulder")):
        out[(a, b)] = float(np.linalg.norm(p[names.index(a)] - p[names.index(b)]))
    return out


def root_depth_from_bones(uv: np.ndarray, z_rel: np.ndarray, conf: np.ndarray, K: np.ndarray,
                          bones: dict[tuple[str, str], float], names: list[str]) -> np.ndarray:
    """Profondeur du bassin Z0 (T,) : pour chaque segment, |d Z0 + e| = L (rayons de vue + profondeur
    relative) ; médiane pondérée des solutions positives sur les segments fiables."""
    T = len(uv)
    rays = backproject(uv, np.ones(uv.shape[:-1]), K)
    Z0 = np.full(T, np.nan)
    for t in range(T):
        sols, ws = [], []
        for (a, b), L in bones.items():
            i, j = names.index(a), names.index(b)
            c = min(conf[t, i], conf[t, j])
            if c <= 0:
                continue
            d = rays[t, i] - rays[t, j]
            e = rays[t, i] * z_rel[t, i] - rays[t, j] * z_rel[t, j]
            qa, qb, qc = d @ d, 2 * d @ e, e @ e - L**2
            disc = qb**2 - 4 * qa * qc
            if qa < 1e-12 or disc < 0:
                continue
            r = (-qb + np.sqrt(disc)) / (2 * qa)
            if r > 0.3:
                sols.append(r)
                ws.append(c)
        if len(sols) >= 3:
            o = np.argsort(sols)
            cw = np.cumsum(np.array(ws)[o])
            Z0[t] = np.array(sols)[o][np.searchsorted(cw, cw[-1] / 2)]
    ok = np.isfinite(Z0)
    if ok.sum() == 0:
        raise RuntimeError("profondeur du bassin non estimable (aucun segment fiable)")
    Z0 = interpolate_invalid(Z0[:, None], ok)[:, 0]
    return median_filter(Z0, size=min(9, T), mode="nearest")


@dataclass
class CachedModel:
    name: str
    license: str
    src_names: list[str]
    mapping: Mapping


CACHED_MODELS: dict[str, CachedModel] = {
    "mediapipe": CachedModel("mediapipe", "Apache-2.0 (code et modèle) : usage commercial autorisé",
                             MEDIAPIPE33_NAMES, MHR70_FROM_MEDIAPIPE),
    "rtmw3d": CachedModel("rtmw3d", "code MMPose Apache-2.0 ; poids entraînés sur Cocktail14 (dont H36M, "
                          "licences non commerciales) : recherche uniquement", COCO_WB133_NAMES, MHR70_FROM_COCO_WB),
    "nlf": CachedModel("nlf", "NON COMMERCIAL (poids NLF + modèle SMPL)", NLF_NAMES, MHR70_FROM_NLF),
}


class CachedBackend:
    """Relit `weights/<name>_cache/<sous-dossier>__<vidéo>.npz` (produit par tools/run_<name>.py)."""

    def __init__(self, name: str, cache_dir: str | Path | None = None):
        self.spec = CACHED_MODELS[name]
        self.name = name
        self.license = self.spec.license
        env = os.environ.get(f"{name.upper()}_CACHE")
        self.cache_dir = Path(cache_dir or env or f"weights/{name}_cache")

    def cache_file(self, video_path: Path) -> Path:
        return self.cache_dir / f"{video_path.parent.name}__{video_path.stem}.npz"

    def process_video(self, video_path: str | Path, max_frames: int | None = None) -> PoseSequence:
        video_path = Path(video_path)
        f = self.cache_file(video_path)
        if not f.exists():
            raise FileNotFoundError(f"{f} absent : lancer tools/run_{self.name}.py")
        d = np.load(f)
        info = read_video_info(video_path)
        intr = default_intrinsics(info.width, info.height, float(d["focal"]))
        K = intr.K
        j3, j2, sc = d["joints3d"], d["joints2d"], d["scores"]
        valid = d["frame_valid"].astype(bool) & np.isfinite(j2).all(axis=(1, 2)) & np.isfinite(j3).all(axis=(1, 2))
        j3, j2, sc = np.nan_to_num(j3), np.nan_to_num(j2), np.nan_to_num(sc) * valid[:, None]
        m3, m2, conf = to_mhr70(self.spec.src_names, self.spec.mapping, j3, j2, sc)
        conf *= valid[:, None]
        coords = str(d["coords"])
        if coords == "root_relative":
            for t in np.flatnonzero(valid):
                w = conf[t] > 0
                m3[t] += solve_translation(m3[t, w], m2[t, w], K, conf[t, w])
        elif coords == "relative_depth":
            Z0 = root_depth_from_bones(m2, m3[..., 2], conf, K, reference_bones(), MHR70_JOINT_NAMES)
            m3 = backproject(m2, Z0[:, None] + m3[..., 2], K)
        elif coords != "camera":
            raise ValueError(f"coords inconnu : {coords}")
        if max_frames is not None:
            m3, m2, conf, valid = m3[:max_frames], m2[:max_frames], conf[:max_frames], valid[:max_frames]
        T = len(m3)
        return PoseSequence(
            joints3d_cam=m3, joints2d=m2, confidence=conf, valid=valid, bbox=np.nan_to_num(d["boxes"][:T]),
            fps=info.fps, image_size=(info.width, info.height), intrinsics=intr,
            joint_names=list(MHR70_JOINT_NAMES), backend=self.name,
            timings={"pose_s": float(d["inference_s"]), "n_frames": T,
                     "ms_per_frame": 1000 * float(d["inference_s"]) / max(T, 1), "model_load_s": float(d["load_s"])},
        )
