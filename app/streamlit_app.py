import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


from magic_eye.depth_sculpt import generate_synthetic_depth

from magic_eye.stereogram import (
    StereogramParams,
    generate_autostereogram,
    remap_depth,
    smooth_depth,
)

from magic_eye.patterns import random_dots, blue_noise, vertical_stripes

DEMO_DEPTH_PATH = Path(__file__).resolve().parents[1] / "examples" / "demo_depth.png"

def to_u8(d: np.ndarray) -> np.ndarray:
    """Convert float32 depth [0..1] to uint8 image."""
    return (np.clip(d, 0.0, 1.0) * 255.0).astype(np.uint8)


PRESETS = {
    "Balanced (default)": {
        "near": 0.90,
        "far": 0.25,
        "gamma": 1.5,
        "eye_sep": 90,
        "max_shift": 32,
        "depth_blur": 0.6,
        "bidirectional": True,
    },
    "Character / Portrait": {
        "near": 0.92,
        "far": 0.20,
        "gamma": 1.7,
        "eye_sep": 85,
        "max_shift": 34,
        "depth_blur": 0.5,
        "bidirectional": False,
    },
    "Creature / Wide subject": {
        "near": 0.92,
        "far": 0.35,
        "gamma": 1.6,
        "eye_sep": 100,
        "max_shift": 38,
        "depth_blur": 0.9,
        "bidirectional": True,
    },
    "Landscape / Scene": {
        "near": 0.85,
        "far": 0.40,
        "gamma": 1.4,
        "eye_sep": 110,
        "max_shift": 28,
        "depth_blur": 1.2,
        "bidirectional": True,
    },
    "High detail / Noisy depth": {
        "near": 0.90,
        "far": 0.30,
        "gamma": 1.8,
        "eye_sep": 95,
        "max_shift": 30,
        "depth_blur": 1.4,
        "bidirectional": True,
    },
}

st.set_page_config(
    page_title="Magic Eye Generator",
    page_icon="👁️",
    layout="wide",
)

st.title("👁️ Magic Eye Generator")
st.markdown(
    """
Generate **Magic Eye / autostereogram images** from depth maps or ordinary photos.

- Upload a **depth map** *or* an **RGB photo**
- Optionally upload a **texture/pattern**
- Adjust depth controls live
"""
)

with st.expander("👁️ How to view Magic Eye images"):
    st.markdown("""
    1. Relax your eyes and look *through* the image
    2. Do not focus on the dots
    3. Slowly increase viewing distance
    4. Let the 3D shape pop forward

    Tip: It may take 10–30 seconds the first time.
    """)


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Input")

def ai_available() -> bool:
    try:
        import torch  # noqa: F401
        import cv2  # noqa: F401
        return True
    except Exception:
        return False

demo_mode = st.sidebar.checkbox("Use built-in demo depth", value=False)

preset_name = st.sidebar.selectbox(
    "Preset",
    list(PRESETS.keys()),
    index=0,
)
preset = PRESETS[preset_name]

if ai_available():
    use_ai = st.sidebar.checkbox("Use AI depth from photo", value=False)
else:
    use_ai = False
    st.sidebar.info("AI depth-from-photo is disabled on this deployment.")


uploaded_depth = None
uploaded_image = None

if demo_mode:
    # Demo mode uses built-in depth, no uploads allowed
    uploaded_image = None
    uploaded_depth = None

elif use_ai:
    uploaded_image = st.sidebar.file_uploader(
        "Upload image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )

else:
    uploaded_depth = st.sidebar.file_uploader(
        "Upload depth map (grayscale)",
        type=["png", "jpg", "jpeg"],
    )


uploaded_pattern = st.sidebar.file_uploader(
    "Optional texture / pattern",
    type=["png", "jpg", "jpeg"],
)

st.sidebar.header("Depth controls")
near = st.sidebar.slider("Near", 0.0, 1.0, preset["near"], 0.01)
far = st.sidebar.slider("Far", 0.0, 1.0, preset["far"], 0.01)
gamma = st.sidebar.slider("Gamma", 0.2, 3.0, preset["gamma"], 0.05)

depth_blur = st.sidebar.slider(
    "Depth smoothing (blur radius)",
    0.0,
    3.0,
    preset["depth_blur"],
    0.1,
)

show_depth_debug = st.sidebar.checkbox(
    "Show depth debug (raw/remapped/smoothed)", value=False
)

auto_sculpt_depth = st.sidebar.checkbox(
    "Enhance depth for Magic Eye (recommended)",
    value=True,
)

sculpt_strength = st.sidebar.slider(
    "Depth sculpt strength",
    0.0, 1.0, 0.6, 0.05,
)


st.sidebar.header("Stereogram")
pattern_source = st.sidebar.selectbox(
    "Pattern source",
    ["Built-in: Random dots", "Built-in: Blue noise", "Built-in: Stripes", "Upload image"],
)

eye_sep = st.sidebar.slider(
    "Eye separation (px)",
    20,
    150,
    preset["eye_sep"],
    5,
)

max_shift = st.sidebar.slider(
    "Max depth shift (px)",
    0,
    60,
    preset["max_shift"],
    2,
)

bidirectional = st.sidebar.checkbox(
    "Bidirectional pass (improves wide shapes)",
    value=preset["bidirectional"],
)

st.sidebar.caption(f"Preset tuned for: **{preset_name}**")

mode = st.sidebar.selectbox("Output mode", ["RGB", "L"])

seed = st.sidebar.number_input(
    "Random seed (optional)",
    value=42,
    step=1,
)

generate = st.sidebar.button("Generate Magic Eye")

# ----------------------------
# Main logic
# ----------------------------
if generate:
    try:
        # ---- Get depth ----
                # ---- Get depth ----
        if demo_mode:
            if not DEMO_DEPTH_PATH.exists():
                st.error(f"Demo depth not found at: {DEMO_DEPTH_PATH}")
                st.stop()

            depth_img = Image.open(DEMO_DEPTH_PATH).convert("L")
            depth = np.asarray(depth_img, dtype=np.float32)

            dmin = float(depth.min())
            dmax = float(depth.max())
            if dmax == dmin:
                st.error("Demo depth map has no variation (all pixels identical).")
                st.stop()

            depth = (depth - dmin) / (dmax - dmin)

        elif use_ai:
            if uploaded_image is None:
                st.error("Please upload an image.")
                st.stop()

            from magic_eye.depth_ai import estimate_depth

            # --- Read uploaded image bytes ONCE ---
            img_bytes = uploaded_image.read()
            suffix = Path(uploaded_image.name).suffix or ".png"

            # --- Decode image to RGB NumPy array (needed for sculpting) ---
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_rgb = np.asarray(img_pil, dtype=np.uint8)

            # --- Write temp file for AI depth model ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            # --- Estimate AI depth ---
            depth = estimate_depth(tmp_path)

            # --- OPTIONAL: synthetic depth sculpting ---
            if auto_sculpt_depth:
                depth = generate_synthetic_depth(
                    depth,
                    img_rgb,
                    strength=sculpt_strength,
                )

            # Best-effort cleanup of temp file
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

        else:
            if uploaded_depth is None:
                st.error("Please upload a depth map (or enable demo mode).")
                st.stop()

            depth_img = Image.open(uploaded_depth).convert("L")
            depth = np.asarray(depth_img, dtype=np.float32)

            dmin = float(depth.min())
            dmax = float(depth.max())
            if dmax == dmin:
                st.error("Depth map has no variation (all pixels identical).")
                st.stop()

            depth = (depth - dmin) / (dmax - dmin)


        # ---- Depth debug preview + exports ----
        if show_depth_debug:
            depth_raw = np.clip(depth, 0.0, 1.0).astype(np.float32)
            depth_remapped = remap_depth(depth_raw, near=near, far=far, gamma=gamma)
            depth_smoothed = smooth_depth(depth_remapped, radius=depth_blur)

            st.subheader("Depth debug preview")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.caption(
                    f"Raw depth (min={depth_raw.min():.3f}, max={depth_raw.max():.3f})"
                )
                st.image(to_u8(depth_raw), clamp=True)

            with c2:
                st.caption(
                    f"Remapped (near={near:.2f}, far={far:.2f}, gamma={gamma:.2f}) "
                    f"(min={depth_remapped.min():.3f}, max={depth_remapped.max():.3f})"
                )
                st.image(to_u8(depth_remapped), clamp=True)

            with c3:
                st.caption(
                    f"Smoothed (blur={depth_blur:.2f}) "
                    f"(min={depth_smoothed.min():.3f}, max={depth_smoothed.max():.3f})"
                )
                st.image(to_u8(depth_smoothed), clamp=True)

            # Download buttons
            buf_raw = io.BytesIO()
            Image.fromarray(to_u8(depth_raw), mode="L").save(buf_raw, format="PNG")
            buf_raw.seek(0)

            buf_remap = io.BytesIO()
            Image.fromarray(to_u8(depth_remapped), mode="L").save(
                buf_remap, format="PNG"
            )
            buf_remap.seek(0)

            buf_smooth = io.BytesIO()
            Image.fromarray(to_u8(depth_smoothed), mode="L").save(
                buf_smooth, format="PNG"
            )
            buf_smooth.seek(0)

            st.download_button(
                "Download raw depth",
                data=buf_raw,
                file_name="depth_raw.png",
                mime="image/png",
            )
            st.download_button(
                "Download remapped depth",
                data=buf_remap,
                file_name="depth_remapped.png",
                mime="image/png",
            )
            st.download_button(
                "Download smoothed depth",
                data=buf_smooth,
                file_name="depth_smoothed.png",
                mime="image/png",
            )

        # ---- Pattern source (built-in or upload) ----
        pattern = None
        base = None

        h, w = depth.shape
        channels = 3 if mode == "RGB" else 1
        rng = np.random.default_rng(int(seed))

        if pattern_source == "Built-in: Random dots":
            base = random_dots(h, w, channels=channels, rng=rng)

        elif pattern_source == "Built-in: Blue noise":
            base = blue_noise(h, w, channels=channels, rng=rng)

        elif pattern_source == "Built-in: Stripes":
            base = vertical_stripes(h, w, channels=channels)

        elif pattern_source == "Upload image":
            if uploaded_pattern is not None:
                img = Image.open(uploaded_pattern)
                img = img.convert("RGB" if mode == "RGB" else "L")
                pattern = np.asarray(img, dtype=np.uint8)

        # ---- Generate stereogram ----
        params = StereogramParams(
            eye_separation_px=int(eye_sep),
            max_shift_px=int(max_shift),
        )


        result = generate_autostereogram(
            depth,
            params=params,
            output_mode=mode,
            pattern=pattern,
            base=base,
            seed=int(seed),
            near=near,
            far=far,
            gamma=gamma,
            bidirectional=bidirectional,
            depth_blur=depth_blur,
        )

        # ---- Display + download ----
        st.subheader("Result")
        st.image(result, clamp=True)

        buf = io.BytesIO()
        Image.fromarray(result).save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="Download image",
            data=buf,
            file_name="magic_eye.png",
            mime="image/png",
        )

    except Exception as exc:
        st.exception(exc)
