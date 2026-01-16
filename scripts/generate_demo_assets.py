from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from magic_eye.stereogram import StereogramParams, generate_autostereogram


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
EXAMPLES.mkdir(exist_ok=True)


def make_depth_map(w: int = 900, h: int = 600) -> np.ndarray:
    """
    Create a synthetic depth map that's very stereogram-friendly:
    a sphere + a block + a smaller 'nose' bump (recognisable).
    """
    depth = np.zeros((h, w), dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Big sphere (center)
    cx, cy, r = w * 0.5, h * 0.55, min(w, h) * 0.28
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sphere = np.clip(1.0 - dist / r, 0.0, 1.0) ** 1.5
    depth = np.maximum(depth, sphere * 0.85)

    # Block (left)
    bx0, bx1 = int(w * 0.18), int(w * 0.35)
    by0, by1 = int(h * 0.30), int(h * 0.62)
    depth[by0:by1, bx0:bx1] = np.maximum(depth[by0:by1, bx0:bx1], 0.65)

    # Small bump (right) – helps recognisability
    cx2, cy2, r2 = w * 0.72, h * 0.42, min(w, h) * 0.10
    dist2 = np.sqrt((xx - cx2) ** 2 + (yy - cy2) ** 2)
    bump = np.clip(1.0 - dist2 / r2, 0.0, 1.0) ** 1.8
    depth = np.maximum(depth, bump * 0.95)

    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def save_depth_png(depth: np.ndarray, path: Path) -> None:
    img = Image.fromarray((depth * 255).astype(np.uint8), mode="L")
    img.save(path)


def main() -> None:
    depth = make_depth_map()

    depth_path = EXAMPLES / "demo_depth.png"
    save_depth_png(depth, depth_path)

    params = StereogramParams(eye_separation_px=95, max_shift_px=36)
    img = generate_autostereogram(
        depth,
        params=params,
        output_mode="RGB",
        seed=42,
        near=1.0,
        far=0.0,
        gamma=1.6,
        bidirectional=True,
        depth_blur=0.6,
    )

    out_path = EXAMPLES / "demo_magic_eye.png"
    Image.fromarray(img).save(out_path)

    # Optional: add a small “viewing guide” image (static)
    guide = Image.new("RGB", (900, 180), "black")
    draw = ImageDraw.Draw(guide)
    draw.text((20, 20), "Magic Eye tip: Relax your eyes and look 'through' the image.", fill="white")
    draw.text((20, 60), "If it's too hard: reduce size (browser zoom 80%) and try again.", fill="white")
    guide.save(EXAMPLES / "demo_viewing_tip.png")

    print(f"Saved: {depth_path}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
