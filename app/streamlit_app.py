import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from magic_eye.stereogram import StereogramParams, generate_autostereogram


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

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Input")

use_ai = st.sidebar.checkbox("Use AI depth from photo", value=False)

uploaded_depth = None
uploaded_image = None

if use_ai:
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
near = st.sidebar.slider("Near", 0.0, 1.0, 1.0, 0.01)
far = st.sidebar.slider("Far", 0.0, 1.0, 0.0, 0.01)
gamma = st.sidebar.slider("Gamma", 0.2, 3.0, 1.0, 0.05)

st.sidebar.header("Stereogram")
eye_sep = st.sidebar.slider("Eye separation (px)", 20, 150, 80, 5)
max_shift = st.sidebar.slider("Max depth shift (px)", 0, 60, 24, 2)

mode = st.sidebar.selectbox("Output mode", ["RGB", "L"])

bidirectional = st.sidebar.checkbox("Bidirectional pass (improves wide shapes)", value=True)

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
        if use_ai:
            if uploaded_image is None:
                st.error("Please upload an image.")
                st.stop()

            try:
                from magic_eye.depth_ai import estimate_depth
            except ImportError:
                st.error(
                    "AI depth estimation not available. "
                    "Install optional dependencies with `pip install -e .[ai]`."
                )
                st.stop()

            # Write uploaded bytes to a temporary file so estimate_depth() can use a path
            img_bytes = uploaded_image.read()
            suffix = Path(uploaded_image.name).suffix or ".png"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            depth = estimate_depth(tmp_path)

        else:
            if uploaded_depth is None:
                st.error("Please upload a depth map.")
                st.stop()

            depth_img = Image.open(uploaded_depth).convert("L")
            depth = np.asarray(depth_img, dtype=np.float32)

            dmin = float(depth.min())
            dmax = float(depth.max())
            if dmax == dmin:
                st.error("Depth map has no variation (all pixels identical).")
                st.stop()

            depth = (depth - dmin) / (dmax - dmin)

        # ---- Optional pattern ----
        pattern = None
        if uploaded_pattern is not None:
            if mode == "RGB":
                pattern_img = Image.open(uploaded_pattern).convert("RGB")
                pattern = np.asarray(pattern_img, dtype=np.uint8)
            else:
                pattern_img = Image.open(uploaded_pattern).convert("L")
                pattern = np.asarray(pattern_img, dtype=np.uint8)

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
            seed=int(seed),
            near=near,
            far=far,
            gamma=gamma,
            bidirectional=bidirectional,
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

