"""Interface commune des backends de pose : tout modèle (InstantHMR, SAM 3D Body, ...)
produit une `PoseSequence`, ce qui rend le reste de la pipeline indépendant du modèle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ..video import CameraIntrinsics


@dataclass
class PoseSequence:
    """Séquence de poses d'une personne.

    Attributs (T = nb d'images, J = nb de keypoints) :
        joints3d_cam : (T, J, 3) keypoints 3D en repère caméra OpenCV (m, X droite, Y bas, Z avant).
        joints2d : (T, J, 2) keypoints en pixels.
        confidence : (T, J) confiance dans [0, 1].
        valid : (T,) image avec une personne détectée.
        bbox : (T, 4) boîte englobante [x1, y1, x2, y2].
    """

    joints3d_cam: np.ndarray
    joints2d: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    bbox: np.ndarray
    fps: float
    image_size: tuple[int, int]  # (width, height)
    intrinsics: CameraIntrinsics
    joint_names: list[str]
    backend: str
    timings: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.joints3d_cam)

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self)) / self.fps

    def joint(self, name: str) -> np.ndarray:
        return self.joints3d_cam[:, self.joint_names.index(name)]

    def save(self, path: str | Path) -> None:
        k = self.intrinsics
        np.savez_compressed(
            path,
            joints3d_cam=self.joints3d_cam, joints2d=self.joints2d, confidence=self.confidence,
            valid=self.valid, bbox=self.bbox, fps=self.fps, image_size=np.array(self.image_size),
            intrinsics=np.array([k.fx, k.fy, k.cx, k.cy]), joint_names=np.array(self.joint_names),
            backend=self.backend, timings_keys=np.array(list(self.timings)),
            timings_values=np.array(list(self.timings.values()), dtype=float),
        )

    @classmethod
    def load(cls, path: str | Path) -> PoseSequence:
        d = np.load(path, allow_pickle=False)
        fx, fy, cx, cy = d["intrinsics"]
        return cls(
            joints3d_cam=d["joints3d_cam"], joints2d=d["joints2d"], confidence=d["confidence"],
            valid=d["valid"], bbox=d["bbox"], fps=float(d["fps"]),
            image_size=tuple(int(v) for v in d["image_size"]),
            intrinsics=CameraIntrinsics(fx, fy, cx, cy), joint_names=[str(n) for n in d["joint_names"]],
            backend=str(d["backend"]),
            timings=dict(zip([str(k) for k in d["timings_keys"]], d["timings_values"].tolist(), strict=True)),
        )


class PoseBackend(Protocol):
    """Un backend transforme une vidéo en `PoseSequence` pour la personne principale."""

    name: str
    license: str

    def process_video(self, video_path: str | Path, max_frames: int | None = None) -> PoseSequence: ...
