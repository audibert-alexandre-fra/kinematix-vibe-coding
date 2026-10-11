import mujoco
import numpy as np
import pytest

from kinematix_vibe_coding.inverse_kinematix.angles import joint_angles, load_angles_csv, save_angles_csv
from kinematix_vibe_coding.models.humanoid import BodyDimensions, Humanoid


@pytest.fixture(scope="module")
def hum():
    return Humanoid(BodyDimensions.from_height(1.7))


@pytest.mark.parametrize("side", ["l", "r"])
def test_shoulder_decomposition_matches_hinge_conventions(hum, side):
    sg = 1 if side == "l" else -1
    for axis, name in [((0, -1, 0), "flexion"), ((-sg, 0, 0), "adduction"), ((0, 0, -sg), "rotation")]:
        q = hum.model.qpos0.copy()
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array(axis, float), np.deg2rad(30))
        a = hum.joint_qposadr(f"shoulder_{side}")
        q[a:a + 4] = quat
        ang = joint_angles(hum, q[None])
        assert ang[f"shoulder_{name}_{side}"][0] == pytest.approx(30, abs=1e-6)


def test_hinge_angles_and_csv(hum, tmp_path):
    q = np.tile(hum.model.qpos0, (3, 1))
    q[:, hum.joint_qposadr("knee_flexion_l")] = np.deg2rad([0, 45, 90])
    ang = joint_angles(hum, q)
    np.testing.assert_allclose(ang["knee_flexion_l"], [0, 45, 90])
    save_angles_csv(tmp_path / "a.csv", np.arange(3) / 30, ang)
    t, back = load_angles_csv(tmp_path / "a.csv")
    np.testing.assert_allclose(back["knee_flexion_l"], [0, 45, 90], atol=1e-3)
