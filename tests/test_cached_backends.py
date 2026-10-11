"""Backends relus depuis un cache (MediaPipe, RTMW3D, NLF) : conversions de squelettes et
reconstruction de la position 3D caméra."""

import numpy as np
import pytest

from kinematix_vibe_coding.models import available_backends, get_backend
from kinematix_vibe_coding.models.cached import (
    CACHED_MODELS,
    backproject,
    reference_bones,
    root_depth_from_bones,
    solve_translation,
    to_mhr70,
)
from kinematix_vibe_coding.utils.skeleton import MHR70_JOINT_NAMES
from kinematix_vibe_coding.video import VideoWriter, default_intrinsics


def test_registered():
    assert {"mediapipe", "rtmw3d", "nlf"} <= set(available_backends())


@pytest.mark.parametrize("name", list(CACHED_MODELS))
def test_mappings_reference_known_points(name):
    spec = CACHED_MODELS[name]
    for dst, (srcs, w) in spec.mapping.items():
        assert dst in MHR70_JOINT_NAMES
        assert all(s in spec.src_names for s in srcs)
        assert 0 < w <= 1
    # les marqueurs essentiels à l'IK sont tous fournis
    for n in ("left_hip", "right_knee", "left_ankle", "right_heel", "left_big_toe_tip", "neck", "right_wrist"):
        assert n in spec.mapping


def test_to_mhr70_average_and_confidence():
    spec = CACHED_MODELS["mediapipe"]
    J = len(spec.src_names)
    j3 = np.random.default_rng(0).normal(size=(2, J, 3))
    sc = np.ones((2, J))
    sc[:, spec.src_names.index("left_shoulder")] = 0.4
    m3, _, conf = to_mhr70(spec.src_names, spec.mapping, j3, j3[..., :2], sc)
    ls, rs = spec.src_names.index("left_shoulder"), spec.src_names.index("right_shoulder")
    neck = MHR70_JOINT_NAMES.index("neck")
    assert np.allclose(m3[:, neck], (j3[:, ls] + j3[:, rs]) / 2)
    assert np.allclose(conf[:, neck], 0.5 * 0.4)  # poids x confiance minimale des points source
    assert conf[:, MHR70_JOINT_NAMES.index("left_acromion")].max() == 0  # sans équivalent


def test_solve_translation_recovers_camera_position():
    K = default_intrinsics(1280, 720).K
    rng = np.random.default_rng(1)
    rel = rng.normal(scale=0.4, size=(20, 3))
    t_true = np.array([0.3, -0.2, 4.5])
    p = rel + t_true
    uv = p[:, :2] / p[:, 2:] * K[0, 0] + K[:2, 2]
    t = solve_translation(rel, uv, K, np.ones(20))
    assert np.allclose(t, t_true, atol=1e-6)


def test_root_depth_from_bones_recovers_depth():
    K = default_intrinsics(1280, 720).K
    bones = reference_bones()
    names = MHR70_JOINT_NAMES
    T = 5
    # squelette synthétique respectant exactement les longueurs de référence : modèle MuJoCo neutre
    from kinematix_vibe_coding.models.humanoid import BodyDimensions, Humanoid

    hum = Humanoid(BodyDimensions.from_height(1.70))
    sites = hum.sites_world(hum.model.qpos0.copy())[0]
    cam = np.zeros((70, 3))
    for i, n in enumerate(names):
        if n in hum.site_names:
            q = sites[hum.site_names.index(n)]
            cam[i] = [-q[1], -q[2], q[0]]  # monde (X avant, Y gauche, Z haut) -> caméra face au sujet
    root = (cam[names.index("left_hip")] + cam[names.index("right_hip")]) / 2
    depths = np.linspace(3.0, 4.0, T)  # trajectoire lisse (filtre médian temporel)
    P = np.stack([cam - root + [0.1, 0.2, d] for d in depths])
    uv = P[..., :2] / P[..., 2:] * K[0, 0] + K[:2, 2]
    conf = np.zeros((T, 70))
    for a, b in bones:
        conf[:, [names.index(a), names.index(b)]] = 1.0
    z_rel = P[..., 2] - depths[:, None]
    Z0 = root_depth_from_bones(uv, z_rel, conf, K, bones, names)
    assert np.allclose(Z0, depths, atol=0.02)
    rec = backproject(uv, Z0[:, None] + z_rel, K)
    ok = conf[0] > 0
    assert np.abs(np.sort(rec[:, ok, 2].ravel()) - np.sort(P[:, ok, 2].ravel())).max() < 0.1


def test_cached_backend_reads_cache(tmp_path):
    video = tmp_path / "grp" / "clip.mp4"
    video.parent.mkdir()
    with VideoWriter(video, 25.0, 64, 48) as w:
        for _ in range(4):
            w.write(np.zeros((48, 64, 3), np.uint8))
    K = default_intrinsics(64, 48).K
    spec = CACHED_MODELS["mediapipe"]
    J = len(spec.src_names)
    rel = np.random.default_rng(3).normal(scale=0.3, size=(4, J, 3))
    t = np.array([0.0, 0.1, 3.0])
    p = rel + t
    uv = p[..., :2] / p[..., 2:] * K[0, 0] + K[:2, 2]
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(cache / "grp__clip.npz", joints3d=rel, joints2d=uv, scores=np.ones((4, J)), boxes=np.zeros((4, 4)),
             frame_valid=np.ones(4, bool), coords="root_relative", fps=25.0, image_size=np.array([64, 48]),
             focal=float(K[0, 0]), inference_s=1.0, load_s=0.5)
    seq = get_backend("mediapipe", cache_dir=cache).process_video(video)
    assert seq.backend == "mediapipe" and len(seq) == 4
    i = MHR70_JOINT_NAMES.index("left_knee")
    assert np.allclose(seq.joints3d_cam[:, i], p[:, spec.src_names.index("left_knee")], atol=0.01)  # cou = milieu 2D des épaules
    assert seq.confidence[:, i].min() == 1.0
