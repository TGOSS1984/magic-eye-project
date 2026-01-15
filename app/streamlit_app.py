import io
import numpy as np
import streamlit as st
from PIL import Image

from magic_eye.io import load_depth_map, load_pattern
from magic_eye.stereogram import generate_autostereogram, StereogramParams


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

            img_bytes = uploaded_image.read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tmp = io.BytesIO()
            image.save(tmp, format="PNG")
            tmp.seek(0)

            depth = estimate_depth(tmp)

        else:
            if uploaded_depth is None:
                st.error("Please upload a depth map.")
                st.stop()

            depth_img = Image.open(uploaded_depth).convert("L")
            depth = np.asarray(depth_img, dtype=np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min())

        pattern = None
        if uploaded_pattern is not None:
            pattern_img = Image.open(uploaded_pattern)
            pattern = load_pattern(uploaded_pattern, mode=mode)

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
        )

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
