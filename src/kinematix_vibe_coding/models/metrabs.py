"""Backend MeTRAbs (référence MonocularBiomechanics) — USAGE NON COMMERCIAL UNIQUEMENT.

MeTRAbs-ACAE-L tourne sous TensorFlow dans l'environnement de MonocularBiomechanics
(tools/run_metrabs.py) ; ce backend relit ses résultats (weights/metrabs_cache/) et convertit le
squelette bml_movi_87 vers les noms MHR70 utilisés par nos marqueurs. Les points sans équivalent
reçoivent une confiance nulle (ignorés par la calibration et l'IK).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..utils.skeleton import MHR70_JOINT_NAMES
from ..video import default_intrinsics, read_video_info
from .base import PoseSequence

# MHR70 -> (point bml_movi_87, confiance)
# pas d'équivalent exact des yeux dans bml_movi_87 : "Head" sert de repère de tête (faible poids) ;
# le segment cou -> yeux mal défini est borné par l'a priori anthropométrique (calibration/scaling.py)
MHR70_FROM_MOVI: dict[str, tuple[str, float]] = {"neck": ("Neck", 1.0), "left_eye": ("Head", 0.3),
                                                 "right_eye": ("Head", 0.3)}
for _s, _m in (("left", "L"), ("right", "R")):
    _l = _m.lower()
    MHR70_FROM_MOVI.update({
        f"{_s}_hip": (f"{_m}Hip", 1.0), f"{_s}_knee": (f"{_m}Knee", 1.0), f"{_s}_ankle": (f"{_m}Ankle", 1.0),
        f"{_s}_heel": (f"{_m}Heel", 1.0), f"{_s}_big_toe_tip": (f"{_m}BigToe", 1.0),
        f"{_s}_small_toe_tip": (f"{_l}fifthmetatarsal", 0.5), f"{_s}_shoulder": (f"{_m}Shoulder", 1.0),
        f"{_s}_acromion": (f"{_l}shom", 0.5), f"{_s}_elbow": (f"{_m}Elbow", 1.0), f"{_s}_wrist": (f"{_m}Wrist", 1.0),
        f"{_s}_middle_third_joint": (f"{_m}Hand", 0.5),
    })

MOVI87_NAMES = [
    "backneck", "upperback", "clavicle", "sternum", "umbilicus", "lfronthead", "lbackhead", "lback", "lshom",
    "lupperarm", "lelbm", "lforearm", "lwrithumbside", "lwripinkieside", "lfin", "lasis", "lpsis", "lfrontthigh",
    "lthigh", "lknem", "lankm", "LHeel", "lfifthmetatarsal", "LBigToe", "lcheek", "lbreast", "lelbinner", "lwaist",
    "lthumb", "lfrontinnerthigh", "linnerknee", "lshin", "lfirstmetatarsal", "lfourthtoe", "lscapula", "lbum",
    "rfronthead", "rbackhead", "rback", "rshom", "rupperarm", "relbm", "rforearm", "rwrithumbside",
    "rwripinkieside", "rfin", "rasis", "rpsis", "rfrontthigh", "rthigh", "rknem", "rankm", "RHeel",
    "rfifthmetatarsal", "RBigToe", "rcheek", "rbreast", "relbinner", "rwaist", "rthumb", "rfrontinnerthigh",
    "rinnerknee", "rshin", "rfirstmetatarsal", "rfourthtoe", "rscapula", "rbum", "Head", "mhip", "CHip", "Neck",
    "LAnkle", "LElbow", "LHip", "LHand", "LKnee", "LShoulder", "LWrist", "LFoot", "RAnkle", "RElbow", "RHip",
    "RHand", "RKnee", "RShoulder", "RWrist", "RFoot",
]


def movi_to_mhr70(poses3d: np.ndarray, poses2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(poses3d)
    j3, j2, conf = np.zeros((T, 70, 3)), np.zeros((T, 70, 2)), np.zeros((T, 70))
    for name, (src, c) in MHR70_FROM_MOVI.items():
        i, k = MHR70_JOINT_NAMES.index(name), MOVI87_NAMES.index(src)
        j3[:, i], j2[:, i], conf[:, i] = poses3d[:, k], poses2d[:, k], c
    return j3, j2, conf


class MetrabsBackend:
    name = "metrabs"
    license = "NON COMMERCIAL : poids MeTRAbs entraînés sur des données à licence non commerciale"

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir or os.environ.get("METRABS_CACHE", "weights/metrabs_cache"))

    def process_video(self, video_path: str | Path, max_frames: int | None = None) -> PoseSequence:
        video_path = Path(video_path)
        f = self.cache_dir / f"{video_path.parent.name}__{video_path.stem}.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f} absent : lancer tools/run_metrabs.py (environnement MonocularBiomechanics)")
        d = np.load(f)
        info = read_video_info(video_path)
        p3, p2 = d["poses3d"], d["poses2d"]
        valid = np.isfinite(p3).all(axis=(1, 2))
        p3, p2 = np.nan_to_num(p3), np.nan_to_num(p2)
        j3, j2, conf = movi_to_mhr70(p3, p2)
        b = d["boxes"]
        bbox = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)
        return PoseSequence(
            joints3d_cam=j3, joints2d=j2, confidence=conf * valid[:, None], valid=valid, bbox=bbox, fps=info.fps,
            image_size=(info.width, info.height), intrinsics=default_intrinsics(info.width, info.height, float(d["focal"])),
            joint_names=list(MHR70_JOINT_NAMES), backend=self.name,
            timings={"pose_s": float(d["inference_s"]), "n_frames": len(p3),
                     "ms_per_frame": 1000 * float(d["inference_s"]) / len(p3), "model_load_s": float(d["load_s"])},
        )
