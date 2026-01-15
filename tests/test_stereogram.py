import numpy as np

from magic_eye.stereogram import StereogramParams, generate_autostereogram


def test_generate_autostereogram_rgb_shape_and_dtype():
    depth = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    depth = np.tile(depth, (100, 1))  # (H=100, W=200)

    img = generate_autostereogram(depth, params=StereogramParams(eye_separation_px=40, max_shift_px=16))

    assert img.shape == (100, 200, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0
    assert img.max() <= 255


def test_generate_autostereogram_grayscale_shape_and_dtype():
    depth = np.zeros((60, 120), dtype=np.float32)

    img = generate_autostereogram(
        depth,
        params=StereogramParams(eye_separation_px=30, max_shift_px=10),
        output_mode="L",
    )

    assert img.shape == (60, 120)
    assert img.dtype == np.uint8


def test_eye_separation_must_be_less_than_width():
    depth = np.zeros((10, 20), dtype=np.float32)

    try:
        generate_autostereogram(depth, params=StereogramParams(eye_separation_px=25, max_shift_px=10))
        assert False, "Expected ValueError for invalid eye_separation_px"
    except ValueError:
        assert True

def test_deterministic_output_same_seed_matches():
    depth = np.tile(np.linspace(0.0, 1.0, 160, dtype=np.float32), (60, 1))
    params = StereogramParams(eye_separation_px=40, max_shift_px=16)

    img1 = generate_autostereogram(depth, params=params, seed=123)
    img2 = generate_autostereogram(depth, params=params, seed=123)

    assert np.array_equal(img1, img2)


def test_deterministic_output_different_seed_differs():
    depth = np.tile(np.linspace(0.0, 1.0, 160, dtype=np.float32), (60, 1))
    params = StereogramParams(eye_separation_px=40, max_shift_px=16)

    img1 = generate_autostereogram(depth, params=params, seed=1)
    img2 = generate_autostereogram(depth, params=params, seed=2)

    assert not np.array_equal(img1, img2)
