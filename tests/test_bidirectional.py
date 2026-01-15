import numpy as np

from magic_eye.stereogram import StereogramParams, generate_autostereogram


def test_bidirectional_output_shape_and_dtype_rgb():
    depth = np.tile(np.linspace(0.0, 1.0, 160, dtype=np.float32), (60, 1))
    params = StereogramParams(eye_separation_px=40, max_shift_px=20)

    img = generate_autostereogram(
        depth,
        params=params,
        output_mode="RGB",
        seed=123,
        bidirectional=True,
    )

    assert img.shape == (60, 160, 3)
    assert img.dtype == np.uint8


def test_bidirectional_is_deterministic_with_seed():
    depth = np.tile(np.linspace(0.0, 1.0, 160, dtype=np.float32), (60, 1))
    params = StereogramParams(eye_separation_px=40, max_shift_px=20)

    img1 = generate_autostereogram(depth, params=params, seed=42, bidirectional=True)
    img2 = generate_autostereogram(depth, params=params, seed=42, bidirectional=True)

    assert np.array_equal(img1, img2)
