"""Cinématique inverse MuJoCo par moindres carrés (Levenberg-Marquardt).

Pour chaque image, on minimise sur qpos :
    sum_m w_m |x_m(q) - y_m|^2                      (marqueurs 3D)
  + sum_m w2_m |proj(x_m(q)) - u_m|^2 / sigma_px^2  (reprojection 2D, optionnelle)
  + w_s |q - q_prev|^2                               (continuité temporelle)
  + w_p |q - q0|^2 (articulations seulement)         (a priori de posture, faible)
  + contraintes de contact / non-pénétration du sol (cibles à fort poids)
Les jacobiennes viennent de `mj_jacSite` ; les mises à jour passent par `mj_integratePos`
(gère la rotule des épaules et la liberté du bassin) ; les limites articulaires sont imposées
par projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..models.humanoid import Humanoid


@dataclass
class CameraPose:
    """Passage monde -> caméra : p_cam = R_cw @ (p_world - center) / scale ; projection K."""

    R_cw: np.ndarray
    center: np.ndarray
    scale: float
    K: np.ndarray

    def to_cam(self, p_world: np.ndarray) -> np.ndarray:
        return (p_world - self.center) @ self.R_cw.T / self.scale

    def project(self, p_world: np.ndarray) -> np.ndarray:
        pc = self.to_cam(p_world)
        z = np.clip(pc[..., 2], 1e-6, None)
        return np.stack([self.K[0, 0] * pc[..., 0] / z + self.K[0, 2],
                         self.K[1, 1] * pc[..., 1] / z + self.K[1, 2]], axis=-1)


@dataclass
class IKConfig:
    max_iters: int = 30
    tol: float = 1e-7
    smooth_weight: float = 0.02      # continuité vers l'image précédente (par dof, en rad/m)
    posture_weight: float = 1e-3     # a priori vers la pose neutre (articulations)
    reproj_weight: float = 0.0       # poids relatif des résidus 2D (0 = désactivé)
    sigma_px: float = 15.0           # écart-type des keypoints 2D (pixels) ~ 1 cm en 3D
    ground_sites: list[str] = field(default_factory=lambda: [
        "left_heel", "right_heel", "left_big_toe_tip", "right_big_toe_tip",
        "left_small_toe_tip", "right_small_toe_tip"])
    ground_weight: float = 30.0


class MarkerIK:
    def __init__(self, hum: Humanoid, marker_sites: list[str], config: IKConfig | None = None):
        self.hum = hum
        self.m, self.d = hum.model, hum.data
        self.cfg = config or IKConfig()
        self.marker_ids = np.array([hum.site_id(n) for n in marker_sites])
        self.ground_ids = np.array([hum.site_id(n) for n in self.cfg.ground_sites])
        self.ground_offsets = np.zeros(len(self.ground_ids))  # hauteur du site quand le pied est à plat
        nv = self.m.nv
        self._jac = np.zeros((3, nv))
        # dofs articulaires (pas la liberté du bassin) pour l'a priori de posture
        self.joint_dof_mask = np.ones(nv, bool)
        self.joint_dof_mask[:6] = False
        self._hinge = [(int(self.m.jnt_qposadr[j]), self.m.jnt_range[j]) for j in range(self.m.njnt)
                       if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and self.m.jnt_limited[j]]

    def update_ground_offsets(self) -> None:
        """Hauteur de chaque site de pied au-dessus de la semelle, pose neutre."""
        self.hum.reset()
        sole = self.d.xpos[self.m.body("foot_l").id][2] - self.hum.dims.ankle_height
        self.ground_offsets = self.d.site_xpos[self.ground_ids, 2] - sole

    def _fk(self, q: np.ndarray) -> None:
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def _site_jac(self, sid: int) -> np.ndarray:
        mujoco.mj_jacSite(self.m, self.d, self._jac, None, int(sid))
        return self._jac.copy()

    def clip_limits(self, q: np.ndarray) -> np.ndarray:
        for adr, (lo, hi) in self._hinge:
            q[adr] = min(max(q[adr], lo), hi)
        return q

    def residuals(self, q, targets, weights, extra, q_prev, obs2d, w2d, cam):
        """Assemble (r, J) pour l'état courant."""
        self._fk(q)
        rows_r, rows_J = [], []
        xs = self.d.site_xpos
        for k, sid in enumerate(self.marker_ids):
            w = weights[k]
            if w <= 0:
                continue
            sw = np.sqrt(w)
            J = self._site_jac(sid)
            rows_r.append(sw * (xs[sid] - targets[k]))
            rows_J.append(sw * J)
            if cam is not None and w2d is not None and w2d[k] > 0:
                pw = xs[sid]
                pc = cam.R_cw @ (pw - cam.center) / cam.scale
                z = max(pc[2], 1e-3)
                fx, fy = cam.K[0, 0], cam.K[1, 1]
                u = np.array([fx * pc[0] / z + cam.K[0, 2], fy * pc[1] / z + cam.K[1, 2]])
                dproj = np.array([[fx / z, 0, -fx * pc[0] / z**2], [0, fy / z, -fy * pc[1] / z**2]])
                Jc = dproj @ (cam.R_cw / cam.scale) @ J
                s2 = np.sqrt(w2d[k]) / self.cfg.sigma_px * 0.01  # 1 sigma px ~ 1 cm
                rows_r.append(s2 * (u - obs2d[k]))
                rows_J.append(s2 * Jc)
        for sid, target, w, axes in extra:
            sw = np.sqrt(w)
            J = self._site_jac(sid)
            rows_r.append(sw * (xs[sid] - target)[axes])
            rows_J.append(sw * J[axes])
        # non-pénétration du sol (unilatéral)
        if self.cfg.ground_weight > 0:
            sw = np.sqrt(self.cfg.ground_weight)
            for sid, off in zip(self.ground_ids, self.ground_offsets, strict=True):
                h = xs[sid, 2] - off
                if h < 0:
                    rows_r.append(np.array([sw * h]))
                    rows_J.append(sw * self._site_jac(sid)[2:3])
        nv = self.m.nv
        if q_prev is not None and self.cfg.smooth_weight > 0:
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(self.m, dq, 1.0, q_prev, q)
            sw = np.sqrt(self.cfg.smooth_weight)
            rows_r.append(sw * dq)
            rows_J.append(sw * np.eye(nv))
        if self.cfg.posture_weight > 0:
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(self.m, dq, 1.0, self.m.qpos0, q)
            sw = np.sqrt(self.cfg.posture_weight)
            rows_r.append(sw * dq[self.joint_dof_mask])
            rows_J.append(sw * np.eye(nv)[self.joint_dof_mask])
        return np.concatenate(rows_r), np.vstack(rows_J)

    def solve(self, q_init, targets, weights, extra=(), q_prev=None, obs2d=None, w2d=None, cam=None):
        """Résout une image. `extra` : liste de (site_id, cible(3,), poids, axes) pour contacts."""
        q = self.clip_limits(np.array(q_init, float))
        if cam is not None and self.cfg.reproj_weight > 0 and w2d is not None:
            w2d = np.asarray(w2d) * self.cfg.reproj_weight
        else:
            w2d = None
        r, J = self.residuals(q, targets, weights, extra, q_prev, obs2d, w2d, cam)
        cost = r @ r
        mu = 1e-3
        for _ in range(self.cfg.max_iters):
            H = J.T @ J
            g = J.T @ r
            dq = -np.linalg.solve(H + mu * (np.diag(np.diag(H)) + 1e-6 * np.eye(len(g))), g)
            q_new = q.copy()
            mujoco.mj_integratePos(self.m, q_new, dq, 1.0)
            q_new = self.clip_limits(q_new)
            r_new, J_new = self.residuals(q_new, targets, weights, extra, q_prev, obs2d, w2d, cam)
            cost_new = r_new @ r_new
            if cost_new < cost:
                improvement = cost - cost_new
                q, r, J, cost = q_new, r_new, J_new, cost_new
                mu = max(mu / 3, 1e-7)
                if improvement < self.cfg.tol:
                    break
            else:
                mu *= 5
                if mu > 1e6:
                    break
        self._fk(q)
        return q, cost

    def solve_sequence(self, targets, weights, q_init=None, extras=None, obs2d=None, w2d=None, cam=None,
                       frames=None):
        """IK séquentielle (warm-start sur l'image précédente). Renvoie qpos (T, nq) et coûts (T,)."""
        T = len(targets)
        frames = range(T) if frames is None else frames
        q = self.m.qpos0.copy() if q_init is None else np.array(q_init, float)
        out = np.zeros((T, self.m.nq))
        costs = np.zeros(T)
        q_prev = None
        for t in frames:
            q, costs[t] = self.solve(
                q, targets[t], weights[t], extras[t] if extras is not None else (), q_prev,
                obs2d[t] if obs2d is not None else None, w2d[t] if w2d is not None else None, cam)
            out[t] = q
            q_prev = q
        return out, costs


def root_from_markers(hum: Humanoid, names: list[str], targets: np.ndarray) -> np.ndarray:
    """qpos initial : bassin positionné/orienté d'après hanches et cou, articulations neutres."""
    idx = {n: i for i, n in enumerate(names)}
    lh, rh, nk = targets[idx["left_hip"]], targets[idx["right_hip"]], targets[idx["neck"]]
    center = (lh + rh) / 2
    y = lh - rh
    y /= np.linalg.norm(y) + 1e-9
    z = nk - center
    z -= y * (z @ y)
    z /= np.linalg.norm(z) + 1e-9
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    q = hum.model.qpos0.copy()
    q[:3] = center
    q[3:7] = quat
    return q
