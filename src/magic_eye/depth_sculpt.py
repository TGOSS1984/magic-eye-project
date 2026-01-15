from __future__ import annotations

from typing import Tuple

import numpy as np


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def _distance_transform_binary(mask: np.ndarray) -> np.ndarray:
    """
    Fast-ish distance transform without SciPy.

    This uses a simple two-pass chamfer approximation (not perfect EDT,
    but good enough to sculpt depth inside silhouettes).
    """
    h, w = mask.shape
    inf = 1e9
    dist = np.full((h, w), inf, dtype=np.float32)
    dist[~mask] = 0.0  # background distance = 0, we measure inside the mask

    # forward pass
    for y in range(h):
        for x in range(w):
            if dist[y, x] == 0.0:
                continue
            best = dist[y, x]
            if y > 0:
                best = min(best, dist[y - 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y - 1, x - 1] + 1.4)
                if x < w - 1:
                    best = min(best, dist[y - 1, x + 1] + 1.4)
            if x > 0:
                best = min(best, dist[y, x - 1] + 1.0)
            dist[y, x] = best

    # backward pass
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if dist[y, x] == 0.0:
                continue
            best = dist[y, x]
            if y < h - 1:
                best = min(best, dist[y + 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y + 1, x - 1] + 1.4)
                if x < w - 1:
                    best = min(best, dist[y + 1, x + 1] + 1.4)
            if x < w - 1:
                best = min(best, dist[y, x + 1] + 1.0)
            dist[y, x] = best

    # dist is 0 on background, increases inward
    # we only care inside mask
    inside = dist[mask]
    if inside.size == 0:
        return np.zeros_like(dist, dtype=np.float32)

    maxv = float(inside.max())
    if maxv <= 0:
        return np.zeros_like(dist, dtype=np.float32)

    out = dist / maxv
    out[~mask] = 0.0
    return out.astype(np.float32)


def generate_synthetic_depth(
    depth_ai: np.ndarray,
    image_rgb: np.ndarray,
    *,
    strength: float = 0.6,
    bg_threshold: int = 240,
) -> np.ndarray:
    """
    Generate a depth-friendly 'sculpted' depth map by blending AI depth with
    a silhouette-based distance field.

    Parameters
    ----------
    depth_ai:
        AI-estimated depth map, shape (H, W), values ~[0..1]
    image_rgb:
        Original RGB image array, shape (H, W, 3), uint8
    strength:
        How much to weight the sculpted component (0..1). Typical: 0.4–0.8
    bg_threshold:
        Grayscale threshold used to guess background for high-contrast images.

    Returns
    -------
    np.ndarray
        Depth map float32 in [0..1]
    """
    strength = float(np.clip(strength, 0.0, 1.0))

    # Normalise AI depth to [0..1]
    depth_ai_n = _normalize01(depth_ai)

    # Build foreground mask from brightness (works well for silhouettes / simple backgrounds)
    gray = (
        0.299 * image_rgb[..., 0]
        + 0.587 * image_rgb[..., 1]
        + 0.114 * image_rgb[..., 2]
    ).astype(np.float32)

    # Foreground = darker than bg_threshold (works for white backgrounds / simple posters)
    mask = gray < float(bg_threshold)

    # Sculpted depth: distance from background into object
    sculpt = _distance_transform_binary(mask)
    sculpt = _normalize01(sculpt)

    # Blend: keep global structure from AI, add sculpted internal volume
    out = (1.0 - strength) * depth_ai_n + strength * sculpt
    out = _normalize01(out)

    return out.astype(np.float32)
