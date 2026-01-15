from __future__ import annotations

import argparse
from pathlib import Path

from magic_eye.io import load_depth_map, save_image
from magic_eye.stereogram import StereogramParams, generate_autostereogram


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magic-eye",
        description="Generate a Magic Eye (autostereogram) image from a depth map.",
    )

    parser.add_argument(
        "--depth",
        type=Path,
        required=True,
        help="Path to a grayscale depth map image (bright = near).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the generated stereogram PNG.",
    )

    parser.add_argument(
        "--eye-sep",
        type=int,
        default=80,
        help="Eye separation in pixels (default: 80). Must be < image width.",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=24,
        help="Maximum depth shift in pixels (default: 24).",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="RGB",
        choices=["RGB", "L"],
        help='Output mode: "RGB" (default) or "L" (grayscale).',
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic output.",
    )

    parser.add_argument(
        "--pattern",
        type=Path,
        default=None,
        help="Optional texture/pattern image to tile across the output (PNG/JPG).",
    )

    parser.add_argument(
        "--near",
        type=float,
        default=1.0,
        help="Input depth value treated as near (mapped to 1.0). Default: 1.0",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=0.0,
        help="Input depth value treated as far (mapped to 0.0). Default: 0.0",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Depth curve shaping (>1 punchier, <1 flatter). Default: 1.0",
    )




    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    depth = load_depth_map(args.depth)

    params = StereogramParams(
        eye_separation_px=args.eye_sep,
        max_shift_px=args.max_shift,
    )

    pattern = None
    if args.pattern is not None:
        from magic_eye.io import load_pattern
        pattern = load_pattern(args.pattern, mode=args.mode)


    img = generate_autostereogram(
        depth,
        params=params,
        output_mode=args.mode,
        pattern=pattern,
        seed=args.seed,
        near=args.near,
        far=args.far,
        gamma=args.gamma,
    )

    save_image(img, args.out)

    print(f"✅ Saved stereogram to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
