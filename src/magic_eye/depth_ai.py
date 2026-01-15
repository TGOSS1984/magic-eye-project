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
        If torch / torchvision / cv2 are not installed.
    """
    try:
        import cv2
        import torch
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError(
            "AI depth estimation requires optional dependencies. "
            "Install with: pip install -e .[ai]"
        ) from exc

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load(
        "intel-isl/MiDaS",
        "DPT_Small",
        pretrained=True,
    )
    model.to(device)
    model.eval()

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(256, antialias=True),
            T.Normalize(mean=0.5, std=0.5),
        ]
    )

    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    depth = prediction.squeeze().cpu().numpy()

    # Normalise to [0..1]
    dmin, dmax = depth.min(), depth.max()
    if dmax == dmin:
        raise ValueError("Depth estimator returned constant output")

    depth = (depth - dmin) / (dmax - dmin)
    return depth.astype(np.float32)
