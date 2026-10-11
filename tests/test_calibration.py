import numpy as np
import pytest

from kinematix_vibe_coding.calibration import (
    MarkerSet,
    calibrate,
    estimate_world_transform,
    fit_ground,
    initial_dimensions,
    measure_segments,
    select_frames,
)
from kinematix_vibe_coding.calibration.markers import FOOT_MARKERS
from kinematix_vibe_coding.models.humanoid import BodyDimensions, Humanoid
from tests.test_ik import random_pose


def synthetic_motion(hum, names, T=60, seed=0):
    rng = np.random.default_rng(seed)
    ids = [hum.site_id(n) for n in names]
    poses = [random_pose(hum, rng, 0.25) for _ in range(T)]
    return hum.sites_world(np.array(poses))[:, ids]


@pytest.fixture(scope="module")
def truth():
    d = BodyDimensions.from_height(1.80)
    d.thigh += 0.03
    d.upper_arm -= 0.02
    d.torso += 0.02
    hum = Humanoid(d)
    names = MarkerSet.default().names
    return hum, names, synthetic_motion(hum, names)


def test_initial_dimensions_match_height(truth):
    hum, names, X = truth
    m = {n: X[:, i] for i, n in enumerate(names)}
    seg = measure_segments(m, np.ones(len(X), bool))
    d, s = initial_dimensions(seg, height=hum.dims.standing_height())
    assert d.standing_height() == pytest.approx(hum.dims.standing_height(), abs=1e-6)
    assert s == pytest.approx(1.0, abs=0.08)
    assert d.thigh == pytest.approx(hum.dims.thigh, abs=0.03)


def test_auto_calibration_recovers_geometry_and_scale(truth):
    hum_true, names, X = truth
    H = hum_true.dims.standing_height()
    Xs = X * 0.9  # données à la mauvaise échelle (ambiguïté monoculaire)
    m = {n: Xs[:, i] for i, n in enumerate(names)}
    d0, s = initial_dimensions(measure_segments(m, np.ones(len(X), bool)), H)
    hum = Humanoid(d0)
    w = np.tile(MarkerSet.default().weights, (len(X), 1))
    frames = select_frames(np.ones(len(X), bool), Xs, k=25)
    res = calibrate(hum, names, Xs * s, w, H, frames, rounds=3)
    assert res.dims.standing_height() == pytest.approx(H, abs=0.005)
    assert res.rmse_after < 0.01
    assert res.rmse_after <= res.rmse_before + 1e-9
    assert res.dims.thigh == pytest.approx(hum_true.dims.thigh, abs=0.015)
    assert res.dims.upper_arm == pytest.approx(hum_true.dims.upper_arm, abs=0.015)
    assert s * res.scale_ratio == pytest.approx(1 / 0.9, rel=0.02)


def test_fit_ground_tilted_plane():
    rng = np.random.default_rng(0)
    n_true = np.array([0.0, -np.cos(0.3), np.sin(0.3)])  # caméra inclinée (Y bas en OpenCV)
    e1 = np.array([1.0, 0, 0])
    e2 = np.cross(n_true, e1)
    T = 120
    pts = np.zeros((T, 2, 3))
    for t in range(T):
        base = 2.0 * n_true * -1 + e1 * rng.uniform(-1, 1) + e2 * rng.uniform(-1, 1)
        pts[t, 0] = base
        pts[t, 1] = base + n_true * rng.uniform(0.0, 0.3)  # l'autre pied éventuellement levé
    n, d = fit_ground(pts, n_true + np.array([0.1, 0.0, 0.05]))
    assert np.degrees(np.arccos(np.clip(n @ n_true, -1, 1))) < 2.0
    assert d == pytest.approx(-2.0, abs=0.02)


def test_world_transform_puts_feet_on_ground(truth):
    hum, names, X = truth
    # monde -> "caméra" : rotation quelconque + translation
    ang = 0.4
    R = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]]) @ np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    cam = (X - [0, 0, 1.2]) @ R.T + [0, 0, 4]
    m = {n: cam[:, i] for i, n in enumerate(names)}
    wt = estimate_world_transform(m, FOOT_MARKERS, np.eye(3), scale=1.0)
    W = wt.to_world(cam)
    up_err = np.degrees(np.arccos(wt.R_wc[2] @ (R @ [0, 0, 1])))
    assert up_err < 5
    feet = W[:, [names.index(f) for f in FOOT_MARKERS], 2]
    assert np.percentile(feet.min(1), 10) == pytest.approx(0.0, abs=0.05)
