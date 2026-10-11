import numpy as np
import pytest

from kinematix_vibe_coding.inverse_kinematix.angles import save_angles_csv
from kinematix_vibe_coding.utils.evaluation import EVAL_ANGLES, SYNC_ANGLES, compare_views, evaluate_output


def make_angles(fps, T, shift_s, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    tt = t + shift_s
    ang = {}
    for i, n in enumerate(sorted(set(EVAL_ANGLES + SYNC_ANGLES))):
        ang[n] = 30 * np.sin(2 * np.pi * (0.7 + 0.1 * i) * tt + i) + 0.3 * rng.normal(size=T)
    return t, ang


def test_compare_views_recovers_lag_with_different_fps():
    a = make_angles(30.0, 200, 0.0)
    b = make_angles(25.0, 170, 0.4, seed=1)  # b commence 0.4 s plus tard dans le mouvement
    res = compare_views(a, b)
    assert res["lag_s"] == pytest.approx(0.4, abs=0.03)
    assert res["mean_rmse_deg"] < 2.5


def test_evaluate_output(tmp_path):
    for name, fps, shift in [("cam0", 30.0, 0.0), ("cam1", 25.0, 0.2)]:
        d = tmp_path / "mvt" / name
        d.mkdir(parents=True)
        t, ang = make_angles(fps, 150, shift)
        save_angles_csv(d / "joint_angles.csv", t, ang)
    rep = evaluate_output(tmp_path, verbose=False)
    assert rep["mvt"]["mean_rmse_deg"] < 3
    assert (tmp_path / "mvt" / "comparaison_multivues.png").exists()
