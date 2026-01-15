from __future__ import annotations
import numpy as np


def random_dots(h: int, w: int, *, channels: int, rng: np.random.Generator) -> np.ndarray:
    if channels == 3:
        return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def blue_noise(h: int, w: int, *, channels: int, rng: np.random.Generator) -> np.ndarray:
    # Approximate blue noise via shuffled low-discrepancy grid
    base = np.linspace(0, 255, num=h * w, dtype=np.uint8)
    rng.shuffle(base)
    img = base.reshape(h, w)

    if channels == 3:
        return np.stack([img, img, img], axis=2)
    return img


def vertical_stripes(h: int, w: int, *, channels: int, stripe_width: int = 6) -> np.ndarray:
    x = (np.arange(w) // stripe_width) % 2
    img = (x * 255).astype(np.uint8)
    img = np.tile(img, (h, 1))

    if channels == 3:
        return np.stack([img, img, img], axis=2)
    return img
