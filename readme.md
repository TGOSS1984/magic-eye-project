# Magic Eye Generator 🧠👁️

This project generates **Magic Eye / autostereogram images** from depth maps and
(optionally) from ordinary photos using AI-based depth estimation.

The goal is to build the system incrementally, starting with a classic
depth-map-based stereogram generator, then extending it with a monocular depth
model so users can create Magic Eye images from any photo.

---

## Project Roadmap

### Phase 1 – Core Algorithm
- Load grayscale depth maps
- Generate single-image stereograms (Magic Eye images)
- Support random-dot and texture-based patterns
- Command-line interface

### Phase 2 – Quality & Control
- Adjustable depth scaling and eye separation
- Deterministic output via random seeds
- Improved depth-to-disparity mapping

### Phase 3 – AI Enhancement
- Generate depth maps from ordinary photos
- End-to-end pipeline: image → depth → stereogram
- Optional GPU acceleration

---

## Why This Project?

Magic Eye images combine:
- Image processing
- Human visual perception
- Algorithmic constraint solving

This makes them an excellent showcase project for:
- NumPy-based algorithms
- Clean architectural design
- Optional AI integration without overengineering

---

## Status

✅ Project scaffolding complete  
✅ Depth map loading / saving utilities complete  
🚧 MVP stereogram generation added (random-dot autostereogram)  
🚧 CLI coming next

## Usage (MVP)

Generate a stereogram from a depth map:

```bash
pip install -e .[dev]
python -m magic_eye.cli --depth path/to/depth.png --out output.png

Tweak the depth effect:

python -m magic_eye.cli --depth path/to/depth.png --out output.png --eye-sep 90 --max-shift 30