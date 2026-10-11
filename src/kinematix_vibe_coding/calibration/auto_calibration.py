"""Calibration automatique du modèle MuJoCo (longueurs de segments, offsets des marqueurs,
échelle des données) par optimisation alternée :

1. IK MuJoCo sur K images représentatives avec le modèle courant ;
2. moindres carrés (scipy) sur [dimensions, échelle relative, offsets marqueurs] à poses fixées,
   les positions des marqueurs étant calculées par la cinématique directe MuJoCo
   (`mj_kinematics` après écriture de `body_pos` / `site_pos`).

Contraintes : hauteur debout du modèle = taille connue (forte), a priori anthropométriques
sur les dimensions, offsets de marqueurs petits et symétriques gauche/droite.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

from ..inverse_kinematix.solver import IKConfig, MarkerIK, root_from_markers
from ..models.humanoid import BodyDimensions, Humanoid

# dimensions peu observables : a priori plus serré (fraction de la valeur)
TIGHT_PRIOR = {"ankle_height": 0.08, "eyes_to_top": 0.08, "heel_back": 0.15, "shoulder_drop": 0.3,
               "pelvis_to_lumbar": 0.25}
LOOSE_OFFSET = {"acromion", "olecranon", "cubital_fossa", "toe_tip", "heel", "ear", "eye", "nose",
                "third_joint"}


@dataclass
class CalibrationResult:
    dims: BodyDimensions
    site_offsets: np.ndarray
    scale_ratio: float          # facteur multiplicatif à appliquer à l'échelle des données
    rmse_before: float
    rmse_after: float
    frames: np.ndarray


def select_frames(valid: np.ndarray, markers: np.ndarray, k: int = 40) -> np.ndarray:
    """K images valides réparties sur la séquence en favorisant la diversité de poses."""
    idx = np.flatnonzero(valid)
    if len(idx) <= k:
        return idx
    # échantillonnage le plus éloigné (farthest point) sur la pose centrée
    X = markers[idx] - markers[idx].mean(1, keepdims=True)
    X = X.reshape(len(idx), -1)
    chosen = [len(idx) // 2]
    dmin = np.linalg.norm(X - X[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(X - X[nxt], axis=1))
    return np.sort(idx[chosen])


def _mirror_pairs(names: list[str]) -> list[tuple[int, int]]:
    pairs = []
    for i, n in enumerate(names):
        if n.startswith("left_") and "right_" + n[5:] in names:
            pairs.append((i, names.index("right_" + n[5:])))
    return pairs


def calibrate(
    hum: Humanoid,
    marker_names: list[str],
    markers: np.ndarray,
    weights: np.ndarray,
    height: float,
    frames: np.ndarray,
    rounds: int = 3,
    fit_scale: bool = True,
    sigma_marker: float = 0.015,
    verbose: bool = False,
) -> CalibrationResult:
    """`markers` : (T, M, 3) positions monde ; `weights` : (T, M)."""
    model, data = hum.model, hum.data
    sids = np.array([hum.site_id(n) for n in marker_names])
    Y = markers[frames]
    W = np.sqrt(weights[frames])
    K, M = len(frames), len(marker_names)
    centers = Y[:, [marker_names.index("left_hip"), marker_names.index("right_hip")]].mean(1)

    names = BodyDimensions.names()
    prior = BodyDimensions.from_height(height).as_array()
    sig_dim = np.array([TIGHT_PRIOR.get(n, 0.2) for n in names]) * prior
    sig_off = np.array([0.04 if any(k in n for k in LOOSE_OFFSET) else 0.02 for n in marker_names])
    pairs = _mirror_pairs(marker_names)
    offsets = hum.site_offsets[sids].copy()

    ik = MarkerIK(hum, marker_names, IKConfig(ground_weight=0.0, smooth_weight=0.0, max_iters=40))
    qs = np.array([root_from_markers(hum, marker_names, y) for y in Y])
    rho = 1.0

    def unpack(theta):
        d = theta[: len(names)]
        r = np.exp(theta[len(names)]) if fit_scale else 1.0
        off = theta[len(names) + 1:].reshape(M, 3)
        return d, r, off

    def apply(d, off):
        full = hum.site_offsets.copy()
        full[sids] = off
        hum.apply_dimensions(BodyDimensions.from_array(d), full)

    def residuals(theta):
        d, r, off = unpack(theta)
        apply(d, off)
        res = np.empty((K, M, 3))
        for k in range(K):
            data.qpos[:] = qs[k]
            mujoco.mj_kinematics(model, data)
            target = centers[k] + rho * r * (Y[k] - centers[k])
            res[k] = data.site_xpos[sids] - target
        res *= W[..., None]
        dmodel = BodyDimensions.from_array(d)
        prior_res = [(d - prior) / sig_dim,
                     np.array([(dmodel.standing_height() - height) / 0.002]),
                     (off / sig_off[:, None]).ravel()]
        for li, ri in pairs:  # symétrie : offset gauche = miroir (y -> -y) de l'offset droit
            prior_res.append((off[li] - off[ri] * np.array([1, -1, 1])) / 0.01)
        # résidus marqueurs en unités d'écart-type (sigma_marker) ; images corrélées -> ~10 images effectives
        return np.concatenate([res.ravel() / sigma_marker * np.sqrt(10.0 / K)] + prior_res)

    def marker_rmse():
        err = []
        for k in range(K):
            data.qpos[:] = qs[k]
            mujoco.mj_kinematics(model, data)
            target = centers[k] + rho * (Y[k] - centers[k])
            e = np.linalg.norm(data.site_xpos[sids] - target, axis=1)
            err.append(e[W[k] > 0])
        return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))

    rmse_before = None
    for it in range(rounds):
        # 1) IK sur les images de calibration (cibles à l'échelle courante)
        for k in range(K):
            target = centers[k] + rho * (Y[k] - centers[k])
            qs[k], _ = ik.solve(qs[k], target, W[k] ** 2)
        if rmse_before is None:
            rmse_before = marker_rmse()
        # 2) moindres carrés sur la géométrie
        theta0 = np.concatenate([hum.dims.as_array(), [0.0], offsets.ravel()])
        lo = np.concatenate([0.5 * prior, [np.log(0.7)], -0.15 * np.ones(M * 3)])
        hi = np.concatenate([1.6 * prior, [np.log(1.4)], 0.15 * np.ones(M * 3)])
        theta0 = np.clip(theta0, lo + 1e-9, hi - 1e-9)
        sol = least_squares(residuals, theta0, bounds=(lo, hi), x_scale="jac", max_nfev=30, method="trf")
        d, r, offsets = unpack(sol.x)
        rho *= r
        apply(d, offsets)
        if verbose:
            print(f"[calibration] tour {it}: rmse={marker_rmse() * 100:.2f} cm, échelle x{rho:.3f}, "
                  f"hauteur={BodyDimensions.from_array(d).standing_height():.3f} m")
    # dernière IK pour mesurer l'erreur finale
    for k in range(K):
        qs[k], _ = ik.solve(qs[k], centers[k] + rho * (Y[k] - centers[k]), W[k] ** 2)
    full = hum.site_offsets.copy()
    full[sids] = offsets
    return CalibrationResult(dims=hum.dims, site_offsets=full, scale_ratio=float(rho),
                             rmse_before=float(rmse_before), rmse_after=marker_rmse(), frames=frames)
