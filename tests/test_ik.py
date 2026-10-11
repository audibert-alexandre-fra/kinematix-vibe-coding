import mujoco
import numpy as np
import pytest

from kinematix_vibe_coding.inverse_kinematix import IKConfig, MarkerIK, root_from_markers
from kinematix_vibe_coding.models.humanoid import BodyDimensions, Humanoid

MARKERS = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
           "right_heel", "left_big_toe_tip", "right_big_toe_tip", "left_small_toe_tip", "right_small_toe_tip",
           "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
           "left_index_third_joint", "right_index_third_joint", "left_pinky_third_joint",
           "right_pinky_third_joint", "neck", "nose", "left_ear", "right_ear", "left_eye", "right_eye",
           "left_acromion", "right_acromion", "left_olecranon", "right_olecranon"]


def random_pose(hum, rng, amp=0.5):
    q = hum.model.qpos0.copy()
    q[:3] = [0.3, -0.2, 1.0]
    quat = np.array([np.cos(0.4), 0, 0, np.sin(0.4)])
    q[3:7] = quat
    for j in range(hum.model.njnt):
        if hum.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            lo, hi = hum.model.jnt_range[j]
            mid = np.clip(0.0, lo, hi)
            q[hum.model.jnt_qposadr[j]] = np.clip(mid + rng.uniform(-amp, amp) * (hi - lo) / 2, lo, hi)
    for s in ("l", "r"):
        a = hum.joint_qposadr(f"shoulder_{s}")
        v = rng.normal(size=3) * 0.6
        qq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(qq, v / np.linalg.norm(v), np.linalg.norm(v))
        q[a:a + 4] = qq
    return q


@pytest.fixture(scope="module")
def hum():
    return Humanoid(BodyDimensions.from_height(1.8))


def test_ik_recovers_synthetic_pose(hum):
    rng = np.random.default_rng(0)
    ik = MarkerIK(hum, MARKERS, IKConfig(posture_weight=0.0, ground_weight=0.0, max_iters=100))
    ids = [hum.site_id(n) for n in MARKERS]
    for _ in range(3):
        q_true = random_pose(hum, rng)
        targets = hum.sites_world(q_true)[0, ids]
        q0 = root_from_markers(hum, MARKERS, targets)
        q, cost = ik.solve(q0, targets, np.ones(len(MARKERS)))
        err = np.linalg.norm(hum.sites_world(q)[0, ids] - targets, axis=1)
        assert err.max() < 5e-3, err.max()
        knee = hum.joint_qposadr("knee_flexion_l")
        assert abs(q[knee] - q_true[knee]) < np.deg2rad(2)


def test_joint_limits_respected(hum):
    ik = MarkerIK(hum, MARKERS, IKConfig(ground_weight=0.0))
    ids = [hum.site_id(n) for n in MARKERS]
    targets = hum.sites_world(hum.model.qpos0)[0, ids]
    # genou en hyperextension impossible : on décale la cheville vers l'avant
    k = MARKERS.index("left_ankle")
    targets[k] += [0.4, 0, 0.1]
    q, _ = ik.solve(hum.model.qpos0, targets, np.ones(len(MARKERS)))
    knee = q[hum.joint_qposadr("knee_flexion_l")]
    assert knee >= np.deg2rad(-5) - 1e-9


def test_ground_constraint(hum):
    ik = MarkerIK(hum, MARKERS, IKConfig(ground_weight=1e4))
    ik.update_ground_offsets()
    ids = [hum.site_id(n) for n in MARKERS]
    targets = hum.sites_world(hum.model.qpos0)[0, ids] - [0, 0, 0.1]  # tout le corps 10 cm sous le sol
    q, _ = ik.solve(hum.model.qpos0, targets, np.ones(len(MARKERS)))
    feet = hum.sites_world(q)[0, ik.ground_ids, 2] - ik.ground_offsets
    assert feet.min() > -0.01


def test_sequence_tracking(hum):
    rng = np.random.default_rng(1)
    q_a, q_b = random_pose(hum, rng, 0.3), random_pose(hum, rng, 0.3)
    q_b[:7] = q_a[:7]
    ids = [hum.site_id(n) for n in MARKERS]
    qs = np.array([q_a + (q_b - q_a) * a for a in np.linspace(0, 1, 10)])
    for q in qs:  # renormalise les quaternions
        for a in (3, hum.joint_qposadr("shoulder_l"), hum.joint_qposadr("shoulder_r")):
            q[a:a + 4] /= np.linalg.norm(q[a:a + 4])
    targets = hum.sites_world(qs)[:, ids]
    ik = MarkerIK(hum, MARKERS, IKConfig(ground_weight=0.0, posture_weight=0.0))
    out, costs = ik.solve_sequence(targets, np.ones((10, len(MARKERS))),
                                   q_init=root_from_markers(hum, MARKERS, targets[0]))
    err = np.linalg.norm(hum.sites_world(out)[:, ids] - targets, axis=-1)
    assert np.median(err) < 0.01
