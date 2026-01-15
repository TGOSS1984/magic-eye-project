import numpy as np

from magic_eye.stereogram import smooth_depth


def test_smooth_depth_zero_radius_no_change():
    depth = np.random.default_rng(0).random((40, 60), dtype=np.float32)
    out = smooth_depth(depth, radius=0.0)
    assert np.allclose(out, depth)


def test_smooth_depth_reduces_variance():
    rng = np.random.default_rng(1)
    depth = rng.random((60, 80), dtype=np.float32)

    out = smooth_depth(depth, radius=1.2)

    # Smoothing should generally reduce variance
    assert float(out.var()) < float(depth.var())
    assert out.min() >= 0.0
    assert out.max() <= 1.0
