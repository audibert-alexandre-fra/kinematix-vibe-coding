from pathlib import Path

import numpy as np
import pytest

from kinematix_vibe_coding.models import PoseSequence, available_backends, get_backend
from kinematix_vibe_coding.models.detector import MainPersonTracker, iou
from kinematix_vibe_coding.models.instanthmr import INPUT_SIZE, flip_permutation, preprocess
from kinematix_vibe_coding.utils.skeleton import MHR70_EDGES, MHR70_JOINT_NAMES, mhr70
from kinematix_vibe_coding.video import default_intrinsics


def test_skeleton_consistency():
    assert len(MHR70_JOINT_NAMES) == 70
    assert all(0 <= a < 70 and 0 <= b < 70 for a, b in MHR70_EDGES)
    assert mhr70("left_heel") == 17


def test_flip_permutation_is_involution():
    perm = flip_permutation(MHR70_JOINT_NAMES)
    np.testing.assert_array_equal(perm[perm], np.arange(70))
    assert perm[mhr70("left_knee")] == mhr70("right_knee")
    assert perm[mhr70("neck")] == mhr70("neck")


def test_preprocess_crop_shape_and_padding():
    img = np.full((100, 200, 3), 255, np.uint8)
    crop, sx, sy, size, cliff = preprocess(img, np.array([150, 10, 210, 90], np.float32))
    assert crop.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert size == pytest.approx(96.0)
    assert cliff[0] > 0  # bbox à droite du centre
    _, _, _, _, cliff_f = preprocess(img, np.array([150, 10, 210, 90], np.float32), flip=True)
    assert cliff_f[0] == pytest.approx(-cliff[0])


def test_tracker_keeps_identity():
    tr = MainPersonTracker((640, 480))
    big = np.array([200, 100, 400, 460.0])
    small = np.array([10, 10, 60, 100.0])
    box, ok = tr.update([(small, 0.9), (big, 0.8)])
    np.testing.assert_array_equal(box, big)
    box, ok = tr.update([(small, 0.99), (big + 5, 0.7)])
    np.testing.assert_array_equal(box, big + 5)
    box, ok = tr.update([(small, 0.99)])  # personne principale perdue -> on garde la boîte
    assert not ok and iou(box, big) > 0.9


def test_pose_sequence_roundtrip(tmp_path):
    T = 5
    seq = PoseSequence(
        joints3d_cam=np.random.rand(T, 70, 3), joints2d=np.random.rand(T, 70, 2), confidence=np.ones((T, 70)),
        valid=np.ones(T, bool), bbox=np.zeros((T, 4)), fps=30.0, image_size=(640, 480),
        intrinsics=default_intrinsics(640, 480), joint_names=list(MHR70_JOINT_NAMES), backend="test",
        timings={"pose_s": 1.5},
    )
    seq.save(tmp_path / "s.npz")
    seq2 = PoseSequence.load(tmp_path / "s.npz")
    np.testing.assert_allclose(seq2.joints3d_cam, seq.joints3d_cam)
    assert seq2.timings == {"pose_s": 1.5} and seq2.image_size == (640, 480)
    assert seq2.joint("neck").shape == (T, 3)


def test_registry():
    assert "instanthmr" in available_backends()
    with pytest.raises(KeyError):
        get_backend("inconnu")


@pytest.mark.slow
def test_instanthmr_on_video():
    video = Path("data/jumpingjack/raw-video-cam2.mp4")
    seq = get_backend("instanthmr").process_video(video, max_frames=8)
    assert len(seq) == 8 and seq.valid.all()
    reproj = seq.intrinsics.project(seq.joints3d_cam)
    assert np.median(np.linalg.norm(reproj - seq.joints2d, axis=-1)) < 25


def test_sam3d_parquet_adapter(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from kinematix_vibe_coding.models.sam3d import parquet_to_sequence
    from kinematix_vibe_coding.video import VideoWriter

    video = tmp_path / "v.mp4"
    with VideoWriter(video, 30, 64, 48) as w:
        for _ in range(4):
            w.write(np.zeros((48, 64, 3), np.uint8))
    rows = []
    for f in range(4):
        for pid in (0, 1) if f < 3 else (0,):
            rows.append({"frame_idx": f, "person_id": pid, "joints_3d": np.full(210, 0.1 * f).tolist(),
                         "joints_2d": np.full(140, 10.0).tolist(), "cam_translation": [0.0, 0.0, 3.0],
                         "cam_focal_length": [80.0], "cam_principal_point": [32.0, 24.0],
                         "bbox": [0, 0, 10, 10], "confidence": 0.9})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "v.parquet")
    seq = parquet_to_sequence(tmp_path / "v.parquet", video)
    assert len(seq) == 4 and seq.valid.all()
    np.testing.assert_allclose(seq.joints3d_cam[2, 0], [0.2, 0.2, 3.2])
    assert seq.intrinsics.fx == 80.0


def test_metrabs_mapping():
    from kinematix_vibe_coding.models.metrabs import MOVI87_NAMES, movi_to_mhr70

    assert len(MOVI87_NAMES) == 87
    p3 = np.random.rand(3, 87, 3)
    j3, j2, conf = movi_to_mhr70(p3, np.random.rand(3, 87, 2))
    np.testing.assert_allclose(j3[:, mhr70("left_knee")], p3[:, MOVI87_NAMES.index("LKnee")])
    assert conf[0, mhr70("left_index_tip")] == 0 and conf[0, mhr70("right_hip")] == 1
    assert "metrabs" in available_backends()
