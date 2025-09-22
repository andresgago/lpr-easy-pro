# lpr_easy/utils/io_utils.py
# All comments/docstrings in English.

import csv
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

def ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def collect_images(input_dir: str, pattern: str) -> List[str]:
    """
    Return a list of image paths under input_dir that match a glob pattern.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(input_dir, pattern), recursive=True))
    valid = []
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    for p in paths:
        if Path(p).suffix.lower() in exts:
            valid.append(p)
    return valid

def save_visualization(img: np.ndarray, dets: List[Tuple[int,int,int,int,float,int]], out_path: str, class_names: List[str]):
    """
    Draw bounding boxes on the image and save it.
    dets: list of (x1,y1,x2,y2,score,cls_id)
    """
    vis = img.copy()
    for (x1, y1, x2, y2, score, cls) in dets:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[cls] if 0 <= cls < len(class_names) else 'plate'} {score:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(out_path, vis)

def save_pre(img: np.ndarray, out_path: str, square_size: int):
    resized = cv2.resize(img, (square_size, square_size))
    cv2.imwrite(out_path, resized)

def save_crop(img: np.ndarray, bbox, out_dir: str, base_name: str, idx: int) -> str:
    """
    Save a plate crop from the original image given a bbox (x1,y1,x2,y2).
    Returns the crop path.
    """
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    pad = int(0.05 * max(x2 - x1, y2 - y1))
    x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
    x2p = min(w - 1, x2 + pad); y2p = min(h - 1, y2 + pad)
    crop = img[y1p:y2p, x1p:x2p].copy()
    out_path = os.path.join(out_dir, f"{Path(base_name).stem}_crop{idx:02d}.jpg")
    cv2.imwrite(out_path, crop)
    return out_path

def maybe_rename_crop_with_plate(crop_path: str, plate_text: str) -> str:
    """
    Optionally rename the crop file by appending the plate string.
    """
    if not plate_text:
        return crop_path
    p = Path(crop_path)
    safe_plate = "".join([c for c in plate_text if c.isalnum()])
    new_name = f"{p.stem}__{safe_plate}{p.suffix}"
    new_path = str(p.with_name(new_name))
    try:
        os.replace(str(p), new_path)
        return new_path
    except Exception:
        return crop_path

def write_main_csv(csv_path: str, rows: List[list]):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path","crop_path","x1","y1","x2","y2","score","class","plate","plate_conf"])
        w.writerows(rows)

def write_main_json(json_path: str, entries: List[dict]):
    with open(json_path, "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

def write_ocr_sidecar(path: str, ocr_map: dict):
    if path.lower().endswith(".json"):
        with open(path, "w") as f:
            json.dump(ocr_map, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["crop_path","plate","conf"])
            for k,v in ocr_map.items():
                w.writerow([k, v.get("plate",""), f'{v.get("conf",0.0):.4f}'])
