import numpy as np

from kinematix_vibe_coding.utils.filters import interpolate_invalid, lowpass, segments, smooth_bool


def test_interpolate_invalid():
    x = np.array([0.0, 1.0, np.nan, 3.0, np.nan])
    v = ~np.isnan(x)
    np.testing.assert_allclose(interpolate_invalid(x, v), [0, 1, 2, 3, 3])


def test_lowpass_removes_noise_keeps_signal():
    fps = 30.0
    t = np.arange(300) / fps
    sig = np.sin(2 * np.pi * 1.0 * t)
    noisy = sig + 0.2 * np.sin(2 * np.pi * 12 * t)
    out = lowpass(noisy, fps, 4.0)
    assert np.abs(out - sig)[20:-20].max() < 0.03
    assert lowpass(noisy[:5], fps, 4.0).shape == (5,)  # trop court : inchangé


def test_smooth_bool_and_segments():
    m = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0], bool)
    out = smooth_bool(m, 3)
    np.testing.assert_array_equal(out, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    assert segments(out) == [(0, 6)]
