"""Jeu de marqueurs virtuels : association keypoints du modèle de pose <-> sites MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.base import PoseSequence

# (keypoint MHR70 == nom du site MuJoCo, poids)
DEFAULT_MARKERS: dict[str, float] = {
    "left_hip": 1.0, "right_hip": 1.0,
    "left_knee": 1.0, "right_knee": 1.0,
    "left_ankle": 1.0, "right_ankle": 1.0,
    "left_heel": 0.7, "right_heel": 0.7,
    "left_big_toe_tip": 0.7, "right_big_toe_tip": 0.7,
    "left_small_toe_tip": 0.5, "right_small_toe_tip": 0.5,
    "left_shoulder": 1.0, "right_shoulder": 1.0,
    "left_acromion": 0.5, "right_acromion": 0.5,
    "left_elbow": 1.0, "right_elbow": 1.0,
    "left_olecranon": 0.3, "right_olecranon": 0.3,
    "left_cubital_fossa": 0.3, "right_cubital_fossa": 0.3,
    "left_wrist": 1.0, "right_wrist": 1.0,
    "left_index_third_joint": 0.3, "right_index_third_joint": 0.3,
    "left_pinky_third_joint": 0.3, "right_pinky_third_joint": 0.3,
    "left_middle_third_joint": 0.3, "right_middle_third_joint": 0.3,
    "neck": 1.0, "nose": 0.5,
    "left_eye": 0.3, "right_eye": 0.3, "left_ear": 0.4, "right_ear": 0.4,
}

FOOT_MARKERS = ["left_heel", "right_heel", "left_big_toe_tip", "right_big_toe_tip",
                "left_small_toe_tip", "right_small_toe_tip"]


@dataclass
class MarkerSet:
    names: list[str]
    weights: np.ndarray

    @classmethod
    def default(cls) -> MarkerSet:
        return cls(list(DEFAULT_MARKERS), np.array(list(DEFAULT_MARKERS.values())))

    def extract(self, seq: PoseSequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Renvoie (positions 3D caméra (T,M,3), positions 2D (T,M,2), poids (T,M))."""
        idx = [seq.joint_names.index(n) for n in self.names]
        w = seq.confidence[:, idx] * self.weights[None]
        w = w * seq.valid[:, None]
        return seq.joints3d_cam[:, idx], seq.joints2d[:, idx], w

    def index(self, name: str) -> int:
        return self.names.index(name)
