"""Repère monde à partir d'une caméra fixe : verticale + plan du sol estimés sur le corps.

p_world = scale * R_wc @ p_cam + center      (Z monde vers le haut, sol en z = 0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..inverse_kinematix.solver import CameraPose


@dataclass
class WorldTransform:
    R_wc: np.ndarray        # rotation caméra -> monde
    ground_d: float         # plan du sol en repère caméra (non mis à l'échelle) : n . p = ground_d
    scale: float            # facteur d'échelle métrique appliqué aux données caméra
    K: np.ndarray

    @property
    def center(self) -> np.ndarray:
        """Centre optique de la caméra dans le monde."""
        return np.array([0.0, 0.0, -self.scale * self.ground_d])

    def to_world(self, p_cam: np.ndarray) -> np.ndarray:
        return self.scale * p_cam @ self.R_wc.T + self.center

    def camera_pose(self) -> CameraPose:
        return CameraPose(R_cw=self.R_wc.T.copy(), center=self.center, scale=self.scale, K=self.K)

    def with_scale(self, scale: float) -> WorldTransform:
        return WorldTransform(self.R_wc, self.ground_d, scale, self.K)

    def shifted_ground(self, dz_world: float) -> WorldTransform:
        """Déplace le sol de `dz_world` (m, monde) : les points montent de -dz."""
        return WorldTransform(self.R_wc, self.ground_d + dz_world / self.scale, self.scale, self.K)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def body_up_vector(markers: dict[str, np.ndarray]) -> np.ndarray:
    """Verticale approchée : direction chevilles -> cou, pondérée vers les images jambes tendues."""
    mid_ankle = (markers["left_ankle"] + markers["right_ankle"]) / 2
    mid_hip = (markers["left_hip"] + markers["right_hip"]) / 2
    up = _unit(markers["neck"] - mid_ankle)
    straight = []
    for s in ("left", "right"):
        a = _unit(markers[f"{s}_hip"] - markers[f"{s}_knee"])
        b = _unit(markers[f"{s}_ankle"] - markers[f"{s}_knee"])
        straight.append(np.einsum("ij,ij->i", a, b) < -0.94)  # angle genou > ~160°
    w = (straight[0] & straight[1]).astype(float)
    trunk = _unit(markers["neck"] - mid_hip)
    if w.sum() < 3:
        w = np.ones(len(up))
        up = _unit(up + trunk)
    return _unit((up * w[:, None]).sum(0))


def fit_ground(
    foot_points: np.ndarray,
    n_prior: np.ndarray,
    sigma_m: float = 0.02,
    sigma_angle_deg: float = 8.0,
    iters: int = 4,
) -> tuple[np.ndarray, float]:
    """Plan du sol robuste à partir des points de pied les plus bas de chaque image.

    `foot_points` : (T, F, 3). Minimise sum w_i (n.p_i - d)^2 / sigma^2 + |n - n_prior|^2 / sigma_ang^2,
    en ne gardant que les points proches du bas (rejet des phases de vol / pied levé)."""
    n = _unit(n_prior.copy())
    sig_ang = np.sin(np.deg2rad(sigma_angle_deg))
    T = len(foot_points)
    # point le plus bas par image selon la verticale courante
    for _ in range(iters):
        h = foot_points @ n
        lowest = foot_points[np.arange(T), np.argmin(h, axis=1)]
        hl = lowest @ n
        thr = np.percentile(hl, 40) + 0.03
        P = lowest[hl <= thr]
        if len(P) < 3:
            P = lowest
        c = P.mean(0)
        C = (P - c).T @ (P - c) / sigma_m**2

        def cost(n_vec, C=C):
            u = _unit(n_vec)
            return u @ C @ u + np.sum((u - n_prior) ** 2) / sig_ang**2

        n = _unit(minimize(cost, n, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8}).x)
        d = float(np.median(P @ n))
    return n, d


def estimate_world_transform(
    markers_cam: dict[str, np.ndarray],
    foot_names: list[str],
    K: np.ndarray,
    scale: float,
    foot_offset: float = 0.015,
) -> WorldTransform:
    """Verticale (corps + plan des pieds) -> rotation ; sol -> translation ; échelle donnée."""
    up_cam = body_up_vector(markers_cam)
    feet = np.stack([markers_cam[n] for n in foot_names], axis=1)
    n, d = fit_ground(feet, up_cam)
    # les marqueurs de pied sont ~1.5 cm au-dessus de la semelle (en unités caméra non mises à l'échelle)
    d -= foot_offset / scale
    z = n
    x = np.array([0.0, 0.0, 1.0]) - n * n[2]       # axe optique projeté sur le sol
    if np.linalg.norm(x) < 1e-3:
        x = np.array([1.0, 0.0, 0.0]) - n * n[0]
    x = _unit(x)
    y = np.cross(z, x)
    R_wc = np.stack([x, y, z])  # lignes = axes monde exprimés en repère caméra
    return WorldTransform(R_wc=R_wc, ground_d=d, scale=scale, K=K)
