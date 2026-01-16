# 👁️ Magic Eye Generator

**Live demo:**  
👉 https://magic-eye-project.streamlit.app/

A professional-grade **Magic Eye / autostereogram generator** that creates single-image stereograms from:

- Grayscale **depth maps**
- Ordinary **RGB photos** (via optional AI depth estimation)

The project combines classic stereogram algorithms with modern depth estimation and a polished Streamlit interface.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Project Architecture](#project-architecture)
- [Usage](#usage)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
  - [Web App (Streamlit)](#web-app-streamlit)
- [AI Depth Estimation (Optional)](#ai-depth-estimation-optional)
- [Depth Debug & Inspection Tools](#depth-debug--inspection-tools)
- [Presets & Visual Tuning](#presets--visual-tuning)
- [Deployment](#deployment)
- [Testing Philosophy](#testing-philosophy)
- [Demo Assets](#demo-assets)
- [Roadmap](#roadmap)

---

## 🧠 Overview

Magic Eye (autostereogram) images encode 3D depth information into a **single 2D image** that the human visual system can decode without glasses.

This project builds that pipeline end-to-end:

RGB photo (optional)
↓
Depth estimation
↓
Depth remapping & smoothing
↓
Stereogram constraint solver
↓
Magic Eye image

yaml
Copy code

The system is designed to be:
- Modular
- Inspectable
- Deterministic (when desired)
- Suitable for both CLI and web use

---

## ✨ Key Features

- 🎯 Classic random-dot and texture-based autostereograms
- 🧠 Optional AI depth-from-photo (MiDaS-style monocular depth)
- 🎚️ Full control over depth mapping, eye separation, and smoothing
- 🔁 Bidirectional constraint solving for wide subjects
- 🖼️ Built-in texture generators (random dots, blue noise, stripes)
- 🔍 Depth debug preview (raw → remapped → smoothed)
- 💾 Exportable intermediate depth maps
- 🌐 Deployed Streamlit web demo

---

## ⚙️ How It Works

At its core, the generator:

1. **Normalises depth** into `[0, 1]`
2. **Remaps depth** using near/far planes and gamma curves
3. **Optionally smooths** noisy depth maps
4. **Applies stereogram constraints** horizontally
5. **Blends passes** (optional bidirectional mode)
6. Produces a single stereogram image

This mirrors how classic Magic Eye images were created — with modern tooling.

---

## 🏗️ Project Architecture

magic-eye-project/
│
├── app/
│ └── streamlit_app.py # Web UI
│
├── src/
│ └── magic_eye/
│ ├── stereogram.py # Core algorithm
│ ├── depth_ai.py # AI depth estimation
│ ├── depth_sculpt.py # Synthetic depth enhancement
│ ├── patterns.py # Texture generators
│ └── cli.py # Command-line interface
│
├── examples/
│ └── demo_depth.png # Known-good demo depth map
│
├── docs/
│ ├── screenshots/ # UI screenshots (placeholder)
│ └── demo.gif # Demo animation (placeholder)
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## 🚀 Usage

### Command Line Interface (CLI)

Generate a stereogram from a depth map:

```bash
pip install -e .[dev]
python -m magic_eye.cli \
  --depth path/to/depth.png \
  --out output.png
Tweak depth perception:

bash
Copy code
python -m magic_eye.cli \
  --depth depth.png \
  --out output.png \
  --eye-sep 90 \
  --max-shift 30
🌐 Web App (Streamlit)
Run locally:

bash
Copy code
pip install -e .[web]
streamlit run app/streamlit_app.py
Features
Upload depth maps or RGB photos

Built-in demo depth (no uploads required)

Pattern selection

Live depth tuning

Depth debug previews

Downloadable results

📸 Screenshot placeholder:
docs/screenshots/streamlit_ui.png

🧠 AI Depth Estimation (Optional)
AI depth estimation is optional and disabled on some deployments.

To enable locally:

bash
Copy code
pip install -e .[web,ai]
streamlit run app/streamlit_app.py
The AI pipeline:

Uses a pretrained monocular depth model

Produces relative (not metric) depth

Can be enhanced with synthetic sculpting for stereograms

🔍 Depth Debug & Inspection Tools
When enabled, the app displays:

Raw depth

Remapped depth

Smoothed depth

Each stage can be exported for inspection.

This makes the system transparent and educational — not a black box.

🎛️ Presets & Visual Tuning
Included presets optimise for different subject types:

Balanced (default)

Character / Portrait

Creature / Wide subject

Landscape / Scene

High detail / Noisy depth

Presets adjust:

Near / far depth planes

Gamma curves

Blur radius

Eye separation

Bidirectional passes

☁️ Deployment
This project is deployed using Streamlit Community Cloud.

Deploy your own copy
Fork this repository

Create a new app in Streamlit Community Cloud

Set the entry file to:

bash
Copy code
app/streamlit_app.py
Dependencies are installed from requirements.txt

⚠️ Python version is configured via Streamlit Cloud settings.

🧪 Testing Philosophy
We do not unit-test the AI model, by design:

It downloads large pretrained weights

Outputs vary across hardware

It would break deterministic CI

Instead, testing focuses on:

Core stereogram algorithms

Deterministic depth pipelines

Manual visual verification (industry standard for this domain)

This is deliberate and professional practice.

🎥 Demo Assets
Demo Depth Map
A known-good depth map is included:

bash
Copy code
examples/demo_depth.png
Used for:

Demo mode

Regression testing

Visual tuning

Demo GIF (placeholder)


To create:

Run the app locally

Record a short interaction

Save as docs/demo.gif

🛣️ Roadmap
Completed
✅ Core stereogram algorithm

✅ CLI tool

✅ Streamlit UI

✅ AI depth integration

✅ Depth debug tooling

✅ Cloud deployment

Possible Extensions
Export side-by-side stereo pairs

Animated depth sweeps

VR-friendly output

Performance optimisations (Numba / Cython)

📌 Final Notes
This project is intentionally scoped to demonstrate:

Algorithmic thinking

Visual computing

Clean Python architecture

Responsible AI integration

It is designed to be understandable, extensible, and portfolio-ready.