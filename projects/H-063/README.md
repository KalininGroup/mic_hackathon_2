## MicroSeg Lab (Streamlit)

Microscopy‑agnostic segmentation framework with classical methods, SAM2 refinement, reviewer‑driven selection, and optional human prompting. Upload any image (TIFF/JPG/PNG), tune settings, and export overlays/masks as a ZIP.

### Highlights
- Classical‑first pass with optional SAM2 refinement.
- Reviewer‑driven selection between classical vs SAM outputs.
- Human‑in‑loop click prompts when results look weak.
- Multi‑target support with per‑target summaries.

### Setup
1. Activate your `.venv` in this folder.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure SAM2 is available and set paths if they differ (defaults assume `./sam2` and `./models/sam2.1_hiera_base_plus.pt`):
   ```bash
   export SAM2_ROOT=/Users/you/path/to/sam2
   export SAM2_CKPT=/Users/you/path/to/models/sam2.1_hiera_base_plus.pt
   export SAM2_CFG=/Users/you/path/to/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml
   ```
   Or create a `.env` in repo root:
   ```
   OPENAI_API_KEY=sk-...
   SAM2_ROOT=/Users/you/path/to/sam2
   SAM2_CKPT=/Users/you/path/to/models/sam2.1_hiera_base_plus.pt
   SAM2_CFG=/Users/you/path/to/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml
   ```
4. Set `OPENAI_API_KEY` (env or `.env`) to enable automated planning/review. Without a key, the app falls back to default heuristics.

### Run
```bash
streamlit run app.py
```

Upload an image, choose settings (rounds, relax steps, thresholds, planning/reviewer usage), and run the pipeline. When a human prompt is needed, the UI shows click‑based point/box tools.

You can optionally provide a reference example image (and mask) to guide the planner and reviewer. If you supply a mask, the pipeline infers polarity and scale to adjust classical parameters.

You can also supply prompts as JSON arrays:
- Positive points: `[[x,y], ...]`
- Negative points: `[[x,y], ...]`
- Boxes: `[[x1,y1,x2,y2], ...]`

Start the run, view per-round overlays and summaries, and download masks/plan/history as a ZIP.
