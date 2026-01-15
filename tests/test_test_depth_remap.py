import numpy as np

from magic_eye.stereogram import remap_depth


def test_remap_depth_identity_defaults():
    depth = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    out = remap_depth(depth, near=1.0, far=0.0, gamma=1.0)

    assert np.allclose(out, depth)


def test_remap_depth_inverts_when_near_less_than_far():
    depth = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    out = remap_depth(depth, near=0.0, far=1.0, gamma=1.0)

    # Inverted: 0 -> 1, 1 -> 0
    assert np.allclose(out, np.array([1.0, 0.5, 0.0], dtype=np.float32))


def test_remap_depth_gamma_changes_curve():
    depth = np.array([0.25, 0.5, 0.75], dtype=np.float32)

    out_linear = remap_depth(depth, near=1.0, far=0.0, gamma=1.0)
    out_gamma2 = remap_depth(depth, near=1.0, far=0.0, gamma=2.0)

    # gamma=2 should reduce mid values (x^2 < x for 0<x<1)
    assert np.all(out_gamma2 < out_linear)


def test_remap_depth_invalid_gamma_raises():
    depth = np.array([0.0, 1.0], dtype=np.float32)

    try:
        remap_depth(depth, near=1.0, far=0.0, gamma=0.0)
        assert False, "Expected ValueError for gamma <= 0"
    except ValueError:
        assert True
