import mujoco
import numpy as np
import pytest

from kinematix_vibe_coding.models.humanoid import BodyDimensions, Humanoid


@pytest.fixture(scope="module")
def hum():
    return Humanoid(BodyDimensions.from_height(1.75))


def _site_after(hum, joint, deg, site):
    hum.reset()
    q = hum.model.qpos0.copy()
    q[hum.joint_qposadr(joint)] = np.deg2rad(deg)
    return hum.sites_world(q)[0, hum.site_id(site)], hum.sites_world(hum.model.qpos0)[0, hum.site_id(site)]


def test_neutral_height_matches_dimensions(hum):
    d = hum.dims
    assert d.standing_height() == pytest.approx(1.75, abs=0.01)
    assert hum.neutral_height() == pytest.approx(d.standing_height(), abs=1e-6)


def test_model_structure(hum):
    m = hum.model
    assert m.nq == 7 + len(hum.hinge_names) + 2 * 4  # libre + pivots + 2 rotules (épaules)
    assert {"left_heel", "right_big_toe_tip", "neck", "nose"} <= set(hum.site_names)


@pytest.mark.parametrize("joint,site,axis,sign", [
    ("hip_flexion_l", "left_knee", 0, +1),       # genou vers l'avant
    ("hip_flexion_r", "right_knee", 0, +1),
    ("knee_flexion_l", "left_ankle", 0, -1),     # cheville vers l'arrière
    ("ankle_flexion_r", "right_big_toe_tip", 2, +1),  # dorsiflexion : orteils vers le haut
    ("lumbar_flexion", "neck", 0, +1),
    ("elbow_flexion_l", "left_wrist", 0, +1),
    ("hip_adduction_l", "left_knee", 1, -1),     # adduction : vers la ligne médiane
    ("hip_adduction_r", "right_knee", 1, +1),
    ("hip_rotation_l", "left_big_toe_tip", 1, -1),  # rotation interne : orteils vers l'intérieur
    ("hip_rotation_r", "right_big_toe_tip", 1, +1),
    ("clavicle_elevation_l", "left_shoulder", 2, +1),
    ("lumbar_bending", "neck", 1, -1),           # inclinaison vers la droite
    ("subtalar_r", "right_heel", 1, +1),         # inversion : semelle vers l'intérieur
    ("subtalar_l", "left_heel", 1, -1),
    ("neck_flexion", "nose", 2, -1),
])
def test_sign_conventions(hum, joint, site, axis, sign):
    after, before = _site_after(hum, joint, 30, site)
    assert sign * (after[axis] - before[axis]) > 0.01


def test_apply_dimensions_changes_lengths(hum):
    d = BodyDimensions.from_height(1.75)
    d.thigh += 0.1
    hum.apply_dimensions(d)
    p = hum.sites_world(hum.model.qpos0)[0]
    knee, hip = p[hum.site_id("left_knee")], p[hum.site_id("left_hip")]
    assert np.linalg.norm(knee - hip) == pytest.approx(d.thigh, abs=1e-9)
    hum.apply_dimensions(BodyDimensions.from_height(1.75))


def test_jacobian_available(hum):
    hum.reset()
    mujoco.mj_comPos(hum.model, hum.data)
    jac = np.zeros((3, hum.model.nv))
    mujoco.mj_jacSite(hum.model, hum.data, jac, None, hum.site_id("left_wrist"))
    assert np.abs(jac).sum() > 0
