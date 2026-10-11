"""Extraction des angles articulaires (degrés) d'une trajectoire qpos."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..models.humanoid import Humanoid


def shoulder_angles(quat_wxyz: np.ndarray, side: str) -> dict[str, np.ndarray]:
    """Décompose la rotule d'épaule en flexion / adduction / rotation interne (séquence YXZ
    intrinsèque, mêmes conventions de signe que la hanche) + élévation (angle bras-tronc)."""
    q = np.asarray(quat_wxyz)
    R = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    a, b, c = R.as_euler("YXZ").T
    sg = 1.0 if side == "l" else -1.0
    down = R.apply([0.0, 0.0, -1.0])
    elevation = np.degrees(np.arccos(np.clip(-down[:, 2], -1, 1)))
    return {
        f"shoulder_flexion_{side}": np.degrees(-a),
        f"shoulder_adduction_{side}": np.degrees(-sg * b),
        f"shoulder_rotation_{side}": np.degrees(-sg * c),
        f"shoulder_elevation_{side}": elevation,
    }


def joint_angles(hum: Humanoid, qpos: np.ndarray) -> dict[str, np.ndarray]:
    """Dictionnaire nom -> série (T,) en degrés (pivots + épaules + orientation du bassin)."""
    out = {n: np.degrees(v) for n, v in hum.hinge_angles(qpos).items()}
    for s in ("l", "r"):
        a = hum.joint_qposadr(f"shoulder_{s}")
        out.update(shoulder_angles(qpos[:, a:a + 4], s))
    R = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    yaw, pitch, roll = R.as_euler("ZYX", degrees=True).T
    out["pelvis_tilt"] = pitch
    out["pelvis_list"] = roll
    out["pelvis_rotation"] = yaw
    return out


def save_angles_csv(path, times: np.ndarray, angles: dict[str, np.ndarray]) -> None:
    names = list(angles)
    data = np.column_stack([times] + [angles[n] for n in names])
    np.savetxt(path, data, delimiter=",", header=",".join(["time_s"] + names), comments="", fmt="%.4f")


def load_angles_csv(path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[:, 0], {n: data[:, i] for i, n in enumerate(header) if i > 0}
