# lpr_easy/cli.py
# All comments/docstrings in English.

import argparse
from pathlib import Path
from .config import AppConfig
from .pipelines.detect_then_read import run_pipeline

def _str2bool(v: str):
    s = str(v).strip().lower()
    if s in ("yes","true","t","y","1","on"): return True
    if s in ("no","false","f","n","0","off"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LPR Easy (Pro): YOLO detection + EasyOCR recognition")
    p.add_argument("--input_dir", type=str, default=None, help="Directory with images (used with --pattern).")
    p.add_argument("--image", type=str, default=None, help="Run on a single image file.")
    p.add_argument("--pattern", type=str, default="**/*.*", help="Glob pattern (e.g., '**/*.JPG').")

    p.add_argument("--weights", type=str, required=True, help="YOLO .pt weights path.")
    p.add_argument("--square_size", type=int, default=640, help="YOLO inference size (imgsz).")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")

    p.add_argument("--save_pre", type=str, default=None, help="Folder for preprocessed (resized) images.")
    p.add_argument("--save_vis", type=str, default=None, help="Folder for detection visualizations.")
    p.add_argument("--save_crops", type=str, default=None, help="Folder to save plate crops.")

    p.add_argument("--csv", type=str, default=None, help="Main CSV output with detections (+OCR if enabled).")
    p.add_argument("--out", type=str, default=None, help="Main JSON output file (used if --format json).")
    p.add_argument("--format", dest="fmt", type=str, default="csv", choices=["csv","json"],
                   help="Main output format.")
    p.add_argument("--name_with_plate", action="store_true", help="Rename crop files to include recognized plate.")

    p.add_argument("--ocr", choices=["none","easyocr-plus"], default="none",
                   help="Apply OCR on saved crops (default: none).")
    p.add_argument("--ocr-gpu", type=str, default=None,
                   help="Force GPU for OCR: 'true'/'false'. If omitted, autodetect.")
    p.add_argument("--ocr-out", type=str, default=None,
                   help="Optional separate OCR-only file (CSV or JSON).")
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Single image convenience
    if args.image and not args.input_dir:
        args.input_dir = str(Path(args.image).parent)
        args.pattern = Path(args.image).name

    cfg = AppConfig(**vars(args))
    run_pipeline(cfg)
