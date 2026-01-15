import numpy as np

from magic_eye.stereogram import StereogramParams, generate_autostereogram


def test_pattern_is_tiled_and_used_rgb():
    depth = np.zeros((20, 30), dtype=np.float32)
    params = StereogramParams(eye_separation_px=10, max_shift_px=0)

    # 2x3 RGB pattern with distinct values
    pattern = np.zeros((2, 3, 3), dtype=np.uint8)
    pattern[0, 0] = [10, 20, 30]
    pattern[1, 2] = [200, 210, 220]

    img = generate_autostereogram(
        depth,
        params=params,
        output_mode="RGB",
        pattern=pattern,
        seed=123,  # should not matter when pattern is provided
    )

    assert img.shape == (20, 30, 3)
    # With max_shift=0 and constraints copying from a fixed partner,
    # the image should still originate from the tiled pattern.
    assert (img[0, 0] == np.array([10, 20, 30], dtype=np.uint8)).all()


def test_pattern_shape_validation_grayscale():
    depth = np.zeros((10, 20), dtype=np.float32)
    params = StereogramParams(eye_separation_px=5, max_shift_px=0)

    bad_pattern = np.zeros((4, 4, 3), dtype=np.uint8)

    try:
        generate_autostereogram(
            depth,
            params=params,
            output_mode="L",
            pattern=bad_pattern,
        )
        assert False, "Expected ValueError for wrong pattern shape"
    except ValueError:
        assert True
