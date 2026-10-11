import numpy as np
import pytest

from kinematix_vibe_coding.video import VideoWriter, default_intrinsics, iter_frames, read_video_info


def test_write_read_roundtrip(tmp_path):
    path = tmp_path / "v.mp4"
    with VideoWriter(path, 25.0, 64, 48) as w:
        for i in range(10):
            w.write(np.full((48, 64, 3), i * 20, np.uint8))
    info = read_video_info(path)
    assert (info.width, info.height, info.n_frames) == (64, 48, 10)
    assert info.fps == pytest.approx(25.0)
    frames = list(iter_frames(path, max_frames=4))
    assert len(frames) == 4 and frames[0].shape == (48, 64, 3)


def test_missing_video():
    with pytest.raises(FileNotFoundError):
        read_video_info("nope.mp4")


def test_intrinsics_projection_center():
    cam = default_intrinsics(800, 600)
    assert cam.fx == pytest.approx(1000.0)
    uv = cam.project(np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 2.0]]))
    np.testing.assert_allclose(uv[0], [400, 300])
    np.testing.assert_allclose(uv[1], [900, 300])
