from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


PathLike = Union[str, Path]


def load_depth_map(path: PathLike) -> np.ndarray:
    """
    Load a depth map image and normalise it to float32 [0.0, 1.0].

    The input image is expected to be grayscale or RGB.
    Brighter values are treated as nearer depth.

    Parameters
    ----------
    path : str or Path
        Path to the depth map image.

    Returns
    -------
    np.ndarray
        2D array of shape (H, W) with values in [0.0, 1.0].
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Depth map not found: {path}")

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)

    min_val = arr.min()
    max_val = arr.max()

    if max_val == min_val:
        raise ValueError("Depth map has no variation (all pixels identical)")

    arr = (arr - min_val) / (max_val - min_val)
    return arr


def save_image(image: np.ndarray, path: PathLike) -> None:
    """
    Save an image array to disk as a PNG.

    Parameters
    ----------
    image : np.ndarray
        Image array. Expected shape (H, W) or (H, W, 3).
    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.clip(image, 0, 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save(path)

def load_pattern(path: PathLike, *, mode: str = "RGB") -> np.ndarray:
    """
    Load a pattern/texture image for stereogram background and return as uint8 array.

    Parameters
    ----------
    path : str or Path
        Path to the pattern image.
    mode : str
        "RGB" (default) or "L" (grayscale).

    Returns
    -------
    np.ndarray
        Pattern array:
        - RGB: shape (H, W, 3), dtype uint8
        - L:   shape (H, W), dtype uint8
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pattern image not found: {path}")

    mode_u = mode.upper()
    if mode_u == "RGB":
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    if mode_u in {"L", "GRAY", "GREY"}:
        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=np.uint8)

    raise ValueError('mode must be "RGB" or "L".')
