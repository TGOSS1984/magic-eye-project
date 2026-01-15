from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter


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


def remap_depth(depth: np.ndarray, *, near: float, far: float, gamma: float) -> np.ndarray:
    """
    Remap a normalised depth map [0..1] into a usable depth signal.

    near:
        Depth value in the input considered "near" (mapped to 1.0).
    far:
        Depth value in the input considered "far" (mapped to 0.0).
    gamma:
        Curve applied after range remap. >1 makes near areas pop more,
        <1 boosts mid/far values.

    Returns float32 depth in [0..1].
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    denom = near - far
    if denom == 0:
        raise ValueError("near and far must not be equal")

    d = (depth - far) / denom
    d = np.clip(d, 0.0, 1.0).astype(np.float32, copy=False)

    if gamma != 1.0:
        d = np.power(d, gamma, dtype=np.float32)

    return d


def smooth_depth(depth: np.ndarray, *, radius: float) -> np.ndarray:
    """
    Smooth a normalised depth map [0..1] using a Gaussian blur.

    radius:
        Pillow GaussianBlur radius. 0 disables smoothing.

    Returns float32 depth in [0..1].
    """
    if radius <= 0:
        return depth.astype(np.float32, copy=False)

    depth_u8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(depth_u8, mode="L")
    img_blur = img.filter(ImageFilter.GaussianBlur(radius=float(radius)))

    out = np.asarray(img_blur, dtype=np.float32) / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


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
        ph, pw, _c = pattern.shape
        reps_y = (h + ph - 1) // ph
        reps_x = (w + pw - 1) // pw
        tiled = np.tile(pattern, (reps_y, reps_x, 1))
        return tiled[:h, :w, :]

    raise ValueError("Pattern must be a 2D (L) or 3D (RGB) array.")


def _apply_constraints(
    out: np.ndarray,
    depth: np.ndarray,
    *,
    sep: int,
    max_shift: int,
    direction: str,
) -> np.ndarray:
    """
    Apply stereogram matching constraints in one direction.

    direction:
        "lr" = left-to-right (reference tends to be on the left)
        "rl" = right-to-left (reference tends to be on the right)
    """
    h, w = depth.shape
    channels = 1 if out.ndim == 2 else out.shape[2]

    if direction == "lr":
        for y in range(h):
            for x in range(sep, w):
                shift = int(round(depth[y, x] * max_shift))
                partner = x - sep + shift
                if partner >= 0:
                    if channels == 3:
                        out[y, x, :] = out[y, partner, :]
                    else:
                        out[y, x] = out[y, partner]
        return out

    if direction == "rl":
        for y in range(h):
            for x in range(w - sep - 1, -1, -1):
                shift = int(round(depth[y, x] * max_shift))
                partner = x + sep - shift
                if partner < w:
                    if channels == 3:
                        out[y, x, :] = out[y, partner, :]
                    else:
                        out[y, x] = out[y, partner]
        return out

    raise ValueError('direction must be "lr" or "rl".')


def generate_autostereogram(
    depth: np.ndarray,
    *,
    params: Optional[StereogramParams] = None,
    output_mode: str = "RGB",
    pattern: Optional[np.ndarray] = None,
    base: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    near: float = 1.0,
    far: float = 0.0,
    gamma: float = 1.0,
    bidirectional: bool = False,
    depth_blur: float = 0.0,
) -> np.ndarray:
    """
    Generate a single-image stereogram (Magic Eye / autostereogram) from a depth map.

    Supports three base texture sources (in priority order):
    1) base: a full-size (H,W) or (H,W,3) texture (already generated/tiled)
    2) pattern: a smaller texture image that will be tiled to (H,W)
    3) random dots fallback (seeded via rng/seed)

    If bidirectional=True, constraints are applied left→right and right→left
    and results are blended to reduce directional bias.
    """
    params = params or StereogramParams()
    depth = _validate_depth(depth)
    depth = remap_depth(depth, near=near, far=far, gamma=gamma)
    depth = smooth_depth(depth, radius=depth_blur)

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
    sep = params.eye_separation_px
    max_shift = params.max_shift_px

    channels = 3 if mode_u == "RGB" else 1
    if mode_u not in {"RGB", "L", "GRAY", "GREY"}:
        raise ValueError('output_mode must be "RGB" or "L".')

    # ---- Build base texture (base > pattern > random) ----
    if base is not None:
        if base.shape[:2] != (h, w):
            raise ValueError("Provided base pattern must match depth size.")
        out_base = base.copy()

    elif pattern is not None:
        if mode_u == "RGB":
            if pattern.ndim != 3 or pattern.shape[2] != 3:
                raise ValueError("RGB pattern must have shape (H, W, 3).")
        else:
            if pattern.ndim != 2:
                raise ValueError("Grayscale pattern must have shape (H, W).")

        if pattern.dtype != np.uint8:
            pattern = pattern.astype(np.uint8, copy=False)

        out_base = _tile_pattern(pattern, h, w).copy()

    else:
        if channels == 3:
            out_base = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        else:
            out_base = rng.integers(0, 256, size=(h, w), dtype=np.uint8)

    # ---- Apply constraints ----
    if not bidirectional:
        out = out_base.copy()
        out = _apply_constraints(out, depth, sep=sep, max_shift=max_shift, direction="lr")
        return out

    out_lr = _apply_constraints(out_base.copy(), depth, sep=sep, max_shift=max_shift, direction="lr")
    out_rl = _apply_constraints(out_base.copy(), depth, sep=sep, max_shift=max_shift, direction="rl")

    blended = ((out_lr.astype(np.uint16) + out_rl.astype(np.uint16)) // 2).astype(np.uint8)
    return blended

