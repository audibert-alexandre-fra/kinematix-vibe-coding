"""Génération de séquences synthétiques (marche) vues par une caméra fixe, pour les tests."""

import numpy as np

from kinematix_vibe_coding.calibration import MarkerSet
from kinematix_vibe_coding.models import PoseSequence
from kinematix_vibe_coding.models.humanoid import BodyDimensions, Humanoid
from kinematix_vibe_coding.video import default_intrinsics


def walking_qpos(hum: Humanoid, T: int = 90, fps: float = 30.0, speed: float = 1.0, treadmill: bool = False):
    t = np.arange(T) / fps
    ph = 2 * np.pi * 1.0 * t
    q = np.tile(hum.model.qpos0, (T, 1))
    q[:, 0] = 0.0 if treadmill else -1.0 + speed * t
    q[:, 2] -= 0.02 * (1 - np.cos(2 * ph))
    for side, sgn in (("l", 1), ("r", -1)):
        q[:, hum.joint_qposadr(f"hip_flexion_{side}")] = np.deg2rad(10 + 25 * np.sin(ph * sgn))
        q[:, hum.joint_qposadr(f"knee_flexion_{side}")] = np.deg2rad(25 + 25 * np.sin(ph * sgn + 1.2))
        q[:, hum.joint_qposadr(f"ankle_flexion_{side}")] = np.deg2rad(5 * np.sin(ph * sgn))
        q[:, hum.joint_qposadr(f"elbow_flexion_{side}")] = np.deg2rad(30)
    return q


def synthetic_sequence(height: float = 1.8, T: int = 90, fps: float = 30.0, noise: float = 0.0, seed: int = 0,
                       scale_error: float = 1.0):
    hum = Humanoid(BodyDimensions.from_height(height))
    q = walking_qpos(hum, T, fps)
    ms = MarkerSet.default()
    ids = [hum.site_id(n) for n in ms.names]
    Xw = hum.sites_world(q)[:, ids]
    # caméra : à 4 m sur le côté (+y), 1.2 m de haut, légèrement plongeante
    tilt = np.deg2rad(8)
    R_cw = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float)  # regarde vers -y monde
    Rt = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    R_cw = Rt @ R_cw
    C = np.array([0.0, 4.0, 1.2])
    Xc = (Xw - C) @ R_cw.T * scale_error
    rng = np.random.default_rng(seed)
    Xc = Xc + rng.normal(scale=noise, size=Xc.shape)
    W, H = 1280, 720
    K = default_intrinsics(W, H)
    full3 = np.zeros((T, 70, 3))
    names = list(MarkerSet.default().names)
    from kinematix_vibe_coding.utils.skeleton import MHR70_JOINT_NAMES

    jn = list(MHR70_JOINT_NAMES)
    for i, n in enumerate(names):
        full3[:, jn.index(n)] = Xc[:, i]
    full3[:, [jn.index(n) for n in jn if n not in names]] = Xc[:, :1]
    seq = PoseSequence(
        joints3d_cam=full3, joints2d=K.project(full3), confidence=np.ones((T, 70)), valid=np.ones(T, bool),
        bbox=np.zeros((T, 4)), fps=fps, image_size=(W, H), intrinsics=K, joint_names=jn, backend="synthetic",
    )
    return hum, q, seq
