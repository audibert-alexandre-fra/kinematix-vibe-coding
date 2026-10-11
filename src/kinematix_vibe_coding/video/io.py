"""Lecture / écriture vidéo (OpenCV) et intrinsèques caméra par défaut."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class CameraIntrinsics:
    """Modèle sténopé (pas de distorsion) : focale en pixels, point principal."""

    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        """Projette des points (..., 3) en repère caméra OpenCV vers des pixels (..., 2)."""
        z = np.clip(pts_cam[..., 2], 1e-6, None)
        u = self.fx * pts_cam[..., 0] / z + self.cx
        v = self.fy * pts_cam[..., 1] / z + self.cy
        return np.stack([u, v], axis=-1)


def default_intrinsics(width: int, height: int, focal: float | None = None) -> CameraIntrinsics:
    """Intrinsèques par défaut : focale = diagonale de l'image (convention d'entraînement
    d'InstantHMR / SAM 3D Body quand la caméra n'est pas calibrée), centre optique au centre."""
    f = float(focal) if focal is not None else math.hypot(width, height)
    return CameraIntrinsics(f, f, width / 2.0, height / 2.0)


def read_video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {path}")
    try:
        return VideoInfo(
            path=str(path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(path: str | Path, rgb: bool = True, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Itère sur les images de la vidéo (RGB uint8 par défaut)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {path}")
    n = 0
    try:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if rgb else frame
            n += 1
    finally:
        cap.release()


class VideoWriter:
    """Écriture mp4 (H.264 si disponible, sinon mp4v). Les images sont en BGR."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._w = None
        for codec in ("avc1", "mp4v"):
            w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
            if w.isOpened():
                self._w = w
                break
        if self._w is None:
            raise RuntimeError(f"Impossible de créer la vidéo {path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        self._w.write(frame_bgr)

    def close(self) -> None:
        self._w.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
