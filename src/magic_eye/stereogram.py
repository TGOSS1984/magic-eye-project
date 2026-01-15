from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StereogramParams:
    """
    Parameters controlling stereogram generation.

    eye_separation_px:
        The base horizontal separation (in pixels) used by the stereogram.
        Larger values generally increase perceived depth but can make fusion harder.

    max_shift_px:
        Maximum additional shift applied for 'near' depth pixels.
        Depth values are expected in [0.0, 1.0], so shift = depth * max_shift_px.
    """

    eye_separation_px: int = 80
    max_shift_px: int = 24


def _validate_depth(depth: np.ndarray) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError("Depth map must be a 2D array (H, W).")

    depth = depth.astype(np.float32, copy=False)

    dmin = float(depth.min())
    dmax = float(depth.max())
    if not (0.0 <= dmin <= 1.0 and 0.0 <= dmax <= 1.0):
        raise ValueError(
            "Depth map must be normalised to [0.0, 1.0]. "
            "Use load_depth_map() or normalise before calling."
        )

    return depth

def _tile_pattern(pattern: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Tile a pattern image to exactly (h, w) or (h, w, c).
    """
    if pattern.ndim == 2:
        ph, pw = pattern.shape
        reps_y = (h + ph - 1) // ph
        reps_x = (w + pw - 1) // pw
        tiled = np.tile(pattern, (reps_y, reps_x))
        return tiled[:h, :w]

    if pattern.ndim == 3:
        ph, pw, c = pattern.shape
        reps_y = (h + ph - 1) // ph
        reps_x = (w + pw - 1) // pw
        tiled = np.tile(pattern, (reps_y, reps_x, 1))
        return tiled[:h, :w, :]

    raise ValueError("Pattern must be a 2D (L) or 3D (RGB) array.")


def generate_autostereogram(
    depth: np.ndarray,
    *,
    params: Optional[StereogramParams] = None,
    output_mode: str = "RGB",
    pattern: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a single-image stereogram (Magic Eye / autostereogram) from a depth map.

    This MVP implementation uses a random-dot base and enforces horizontal matching
    constraints row-by-row.

    Parameters
    ----------
    depth:
        Normalised depth map array of shape (H, W), values in [0.0, 1.0].
        Brighter values are treated as nearer depth (more shift).
    params:
        Stereogram parameters (eye separation and max shift).
    output_mode:
        "RGB" (default) or "L" (grayscale).
        pattern:
        Optional texture/pattern image array to use instead of random dots.
        - For RGB output: provide shape (Hp, Wp, 3), dtype uint8
        - For grayscale output: provide shape (Hp, Wp), dtype uint8
        The pattern will be tiled to the output size.

    rng:
        Optional NumPy random generator. If provided, it is used directly.
    seed:
        Optional integer seed. Used only if rng is None. Enables deterministic output.

    Returns
    -------
    np.ndarray
        Output image array:
        - RGB: shape (H, W, 3), dtype uint8
        - L:   shape (H, W), dtype uint8
    """
    params = params or StereogramParams()
    depth = _validate_depth(depth)

    if params.eye_separation_px <= 0:
        raise ValueError("eye_separation_px must be > 0")
    if params.max_shift_px < 0:
        raise ValueError("max_shift_px must be >= 0")

    h, w = depth.shape

    if params.eye_separation_px >= w:
        raise ValueError(
            "eye_separation_px must be smaller than the image width "
            f"(got {params.eye_separation_px} >= {w})."
        )

    if rng is None:
        rng = np.random.default_rng(seed)

    mode_u = output_mode.upper()

    if mode_u == "RGB":
        channels = 3
        if pattern is None:
            out = rng.integers(0, 256, size=(h, w, channels), dtype=np.uint8)
        else:
            if pattern.ndim != 3 or pattern.shape[2] != 3:
                raise ValueError("RGB pattern must have shape (H, W, 3).")
            if pattern.dtype != np.uint8:
                pattern = pattern.astype(np.uint8, copy=False)
            out = _tile_pattern(pattern, h, w).copy()

    elif mode_u in {"L", "GRAY", "GREY"}:
        channels = 1
        if pattern is None:
            out = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        else:
            if pattern.ndim != 2:
                raise ValueError("Grayscale pattern must have shape (H, W).")
            if pattern.dtype != np.uint8:
                pattern = pattern.astype(np.uint8, copy=False)
            out = _tile_pattern(pattern, h, w).copy()

    else:
        raise ValueError('output_mode must be "RGB" or "L".')

    # Enforce matching constraints row-by-row.
    # For each pixel x, we look to a "partner" pixel to the left.
    # Partner depends on depth (near => larger shift).
    sep = params.eye_separation_px
    max_shift = params.max_shift_px

    for y in range(h):
        # Iterate left-to-right so that when we copy from partner,
        # the partner pixel is already stabilised.
        for x in range(sep, w):
            shift = int(round(depth[y, x] * max_shift))
            partner = x - sep + shift
            if partner >= 0:
                if channels == 3:
                    out[y, x, :] = out[y, partner, :]
                else:
                    out[y, x] = out[y, partner]

    return out
