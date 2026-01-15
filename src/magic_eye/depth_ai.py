from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def estimate_depth(image_path: PathLike) -> np.ndarray:
    """
    Estimate a depth map from a single RGB image using a pretrained MiDaS model.

    Returns
    -------
    np.ndarray
        Normalised depth map of shape (H, W), dtype float32, values in [0..1].

    Raises
    ------
    ImportError
        If torch / cv2 are not installed.
    """
    try:
        import cv2
        import torch
    except ImportError as exc:
        raise ImportError(
            "AI depth estimation requires optional dependencies. "
            "Install with: pip install -e .[ai]"
        ) from exc

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use stable hub model name
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
    model.to(device)
    model.eval()

    # Use the transforms provided by the MiDaS repo (most compatible)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError("Could not read image (cv2.imread returned None).")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    depth = prediction.squeeze().cpu().numpy()

    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax == dmin:
        raise ValueError("Depth estimator returned constant output")

    depth = (depth - dmin) / (dmax - dmin)
    return depth.astype(np.float32)

