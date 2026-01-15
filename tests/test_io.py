import numpy as np
from PIL import Image

from magic_eye.io import load_depth_map, save_image


def test_load_depth_map_normalises(tmp_path):
    img = np.array(
        [
            [0, 128],
            [192, 255],
        ],
        dtype=np.uint8,
    )

    path = tmp_path / "depth.png"
    Image.fromarray(img).save(path)

    depth = load_depth_map(path)

    assert depth.shape == (2, 2)
    assert depth.dtype == np.float32
    assert depth.min() == 0.0
    assert depth.max() == 1.0


def test_save_image_creates_file(tmp_path):
    image = np.zeros((10, 10), dtype=np.uint8)
    path = tmp_path / "out.png"

    save_image(image, path)

    assert path.exists()
