# LPR Easy (Pro) — Modular YOLO + EasyOCR

A clean, **modular** Python project for license plate processing:

- **Detection** with Ultralytics YOLO.
- **Recognition** with an improved EasyOCR engine.
- Extensible pipeline architecture.
- Optional **video annotation** (boxes only) using the same detector.

---

## Demo

<video src="https://github.com/andresgago/lpr-easy-pro/blob/main/media/video.mp4" controls muted playsinline loop width="100%" poster="https://github.com/andresgago/lpr-easy-pro/blob/main/media/poster.jpg"></video>

<p><a href="https://github.com/andresgago/lpr-easy-pro/blob/main/media/video.mp4">▶️ Watch the demo (MP4)</a></p>

---

## Highlights

- Clear module boundaries: `detectors/`, `ocr/`, `pipelines/`, `utils/`
- Single image/batch CLI: `python -m lpr_easy ...` (or `lpr-easy` after install)
- Optional OCR (`--ocr easyocr-plus`) for CSV/JSON outputs
- Crop renaming with recognized plate (`--name_with_plate`)
- Brazil/Mercosur normalization in OCR layer
- Video pipeline (recommended): **boxes only, no labels/OCR in frames** for speed/stability

---

## Install

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Requirements (summary)**:
- `ultralytics>=8.2.0`
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `easyocr>=1.7.1`
- `torch>=2.0.0`

---

## Paths (default vs. external)

By default, examples assume everything is inside the repo (e.g., `models/`, `samples/`, `out/`).
If you keep models/data/outputs **outside** the repo, adjust paths accordingly (examples below use external paths).

---

## Usage — Batch (images)

### Detect only (no OCR) — inside the repo
```bash
python -m lpr_easy   --weights models/model05.pt   --input_dir samples   --pattern "**/*.jpg"   --save_vis out_vis   --save_crops out_crops   --csv lpr_batch.csv
```

### Detect only (no OCR) — external paths
```bash
python -m lpr_easy   --weights ../lpr-models/model05.pt   --input_dir ../lpr-data/samples01   --pattern "**/*.JPG"   --save_vis ../lpr-out/vis   --save_crops ../lpr-out/crops   --csv ../lpr-out/lpr_batch.csv
```

> For mixed cases: `--pattern "**/*.[jJ][pP][gG]"`.

### Detect + OCR (plate/conf merged into main CSV) — external paths
```bash
python -m lpr_easy   --weights ../lpr-models/model05.pt   --input_dir ../lpr-data/samples01   --pattern "**/*.JPG"   --save_vis ../lpr-out/vis   --save_crops ../lpr-out/crops   --csv ../lpr-out/lpr_batch.csv   --ocr easyocr-plus
```

### CPU-only OCR + OCR sidecar CSV — external paths
```bash
python -m lpr_easy   --weights ../lpr-models/model05.pt   --input_dir ../lpr-data/samples01   --pattern "**/*.JPG"   --save_vis ../lpr-out/vis   --save_crops ../lpr-out/crops   --csv ../lpr-out/lpr_batch.csv   --ocr easyocr-plus --ocr-gpu=false --ocr-out ../lpr-out/ocr_only.csv
```

### Rename crops with recognized plate
```bash
python -m lpr_easy ... --ocr easyocr-plus --name_with_plate
```

### JSON output instead of CSV
```bash
python -m lpr_easy   --weights ../lpr-models/model05.pt   --input_dir ../lpr-data/samples01   --pattern "**/*.JPG"   --save_crops ../lpr-out/crops   --out ../lpr-out/lpr_batch.json --format json   --ocr easyocr-plus
```

> After installing the package (see **pyproject.toml**), you can also run the CLI as:
> ```bash
> lpr-easy --weights ... (same parameters as above)
> ```

---

## Usage — Video (boxes only, recommended)

Run as module:
```bash
python -m lpr_easy.pipelines.video_detect_only   --weights ../lpr-models/model05.pt   --input_video ../lpr-data/video01.mp4   --output_video ../lpr-out/video01_detected.mp4   --conf 0.15 --square_size 640   --output-max-width 1280
```

- Draws **only the plate rectangles** (no label/score, no OCR).
- Keep `--output-max-width 1280` for smaller files; drop it for full-res.

---

## Troubleshooting

**No input images found**
- Ensure working directory is the project root.
- Quote the pattern: `--pattern "**/*.JPG"`.
- Use `--pattern "**/*.[jJ][pP][gG]"` for case-insensitive matches.

**No module named `lpr_easy.__main__`**
- Add `lpr_easy/__main__.py` (snippet above).

**OpenCV can’t read video**
- Verify the path (`../lpr-data` vs `../lrp-data`).
- Try an absolute path.
- If codec issues: re-encode with FFmpeg:
  ```bash
  ffmpeg -i input.mp4 -vcodec libx264 -acodec aac output_h264.mp4
  ```

**macOS/MPS warnings**
- `pin_memory not supported on MPS` and EasyOCR `overflow` warnings are harmless.

**Video looks slow**
- Don’t set `--fps-out` lower than source FPS. Omit it to keep original FPS.

---

## Why boxes-only for video?

OCR overlays and per-track memory sometimes mixed texts across cars. Final decision: **boxes-only** on video for robust visualization. OCR remains available for batch CSV/JSON.

---

## License & Contributions

- License: MIT.
- Contributions welcome.
