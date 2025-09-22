# lpr_easy/pipelines/detect_then_read.py
# All comments/docstrings in English.

from typing import Dict, Any, List
from pathlib import Path
import cv2

from ..config import AppConfig
from ..detectors.yolo_detector import YoloPlateDetector
from ..ocr.easyocr_engine import EasyOCREngine
from ..utils.io_utils import (
    ensure_dir, collect_images, save_visualization, save_pre,
    save_crop, maybe_rename_crop_with_plate,
    write_main_csv, write_main_json, write_ocr_sidecar
)

def run_pipeline(cfg: AppConfig):
    # Collect images
    if cfg.image and not cfg.input_dir:
        cfg.input_dir = str(Path(cfg.image).parent)
        cfg.pattern = Path(cfg.image).name

    if not cfg.input_dir:
        print("[INFO] Nothing to do: provide --image or --input_dir.")
        return

    imgs = collect_images(cfg.input_dir, cfg.pattern)
    if not imgs:
        print("[INFO] No input images found.")
        return

    # Prepare output dirs
    if cfg.save_pre: ensure_dir(cfg.save_pre)
    if cfg.save_vis: ensure_dir(cfg.save_vis)
    if cfg.save_crops: ensure_dir(cfg.save_crops)

    # Init components
    detector = YoloPlateDetector(cfg.weights)
    ocr_engine = None
    if cfg.ocr == "easyocr-plus":
        gpu_flag = None
        if cfg.ocr_gpu is not None:
            gpu_flag = cfg.ocr_gpu.strip().lower() in ("true","1","yes","on")
        ocr_engine = EasyOCREngine(langs=["en"], gpu=gpu_flag)

    all_rows, all_entries = [], []

    for img_path in imgs:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        dets = detector.predict(img, conf=cfg.conf, imgsz=cfg.square_size)
        class_names = detector.class_names

        # Save visualization and pre if requested
        if cfg.save_vis:
            vis_path = str(Path(cfg.save_vis) / f"{Path(img_path).stem}_det.jpg")
            save_visualization(img, dets, vis_path, class_names)
        if cfg.save_pre:
            pre_path = str(Path(cfg.save_pre) / f"{Path(img_path).stem}_pre.jpg")
            save_pre(img, pre_path, cfg.square_size)

        # Iterate detections and optionally OCR
        for i, (x1,y1,x2,y2,score,cls) in enumerate(dets):
            crop_path = ""
            if cfg.save_crops:
                crop_path = save_crop(img, (x1,y1,x2,y2), cfg.save_crops, Path(img_path).name, i)

            plate_txt, plate_conf = "", 0.0
            if ocr_engine and crop_path:
                try:
                    plate_txt, plate_conf = ocr_engine.read_path(crop_path)
                except Exception:
                    plate_txt, plate_conf = "", 0.0

                if cfg.name_with_plate and plate_txt:
                    new_cp = maybe_rename_crop_with_plate(crop_path, plate_txt)
                    if new_cp != crop_path:
                        crop_path = new_cp

            # append CSV row
            row = [
                img_path, crop_path, x1, y1, x2, y2, f"{score:.4f}",
                class_names[cls] if 0 <= cls < len(class_names) else str(cls),
                plate_txt, f"{plate_conf:.4f}"
            ]
            all_rows.append(row)

            # append JSON entry
            all_entries.append({
                "image": img_path,
                "crop_path": crop_path,
                "bbox": [x1,y1,x2,y2],
                "score": float(score),
                "class": class_names[cls] if 0 <= cls < len(class_names) else str(cls),
                "plate": plate_txt,
                "plate_conf": float(plate_conf),
            })

    # Write outputs
    if cfg.csv:
        write_main_csv(cfg.csv, all_rows)
        print(f"[OK] CSV written: {cfg.csv} ({len(all_rows)} rows)")

    if cfg.out and cfg.fmt.lower() == "json":
        write_main_json(cfg.out, all_entries)
        print(f"[OK] JSON written: {cfg.out} ({len(all_entries)} entries)")

    # Optional OCR sidecar
    if cfg.ocr_out and ocr_engine:
        # Build a simple map { crop_path: {plate, conf} }
        ocr_map = {}
        for e in all_entries:
            if e["crop_path"]:
                ocr_map[e["crop_path"]] = {"plate": e["plate"], "conf": e["plate_conf"]}
        write_ocr_sidecar(cfg.ocr_out, ocr_map)
        print(f"[OK] OCR sidecar written: {cfg.ocr_out}")
