# lpr_easy/ocr/easyocr_engine.py
# All comments/docstrings in English.

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import easyocr

from ..utils.text_utils import normalize_plate, plate_validity_score, ALLOWLIST

def _unsharp(img, ksize=(0,0), sigma=1.0, amount=1.5, thresh=0):
    import numpy as np, cv2
    blur = cv2.GaussianBlur(img, ksize, sigma)
    sharp = cv2.addWeighted(img, 1+amount, blur, -amount, 0)
    if thresh > 0:
        low_contrast_mask = np.absolute(img - blur) < thresh
        np.copyto(sharp, img, where=low_contrast_mask)
    return sharp

def build_variants(bgr: np.ndarray) -> List[np.ndarray]:
    """
    Build a small set of preprocessed grayscale variants for TTA-like OCR.
    """
    h, w = bgr.shape[:2]
    scale = 2 if max(h, w) < 120 else 1
    img = cv2.resize(bgr, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v1 = gray
    v2 = _unsharp(gray, sigma=1.0, amount=1.2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v3 = clahe.apply(gray)
    v4 = cv2.adaptiveThreshold(v2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    return [v1, v2, v3, v4]

class EasyOCREngine:
    """
    EasyOCR engine with tuned defaults for alphanumeric license plates.
    """
    def __init__(self, langs=None, gpu: Optional[bool] = None):
        import torch
        langs = langs or ["en"]
        gpu_flag = torch.cuda.is_available() if gpu is None else bool(gpu)
        self.reader = easyocr.Reader(langs, gpu=gpu_flag, recog_network="latin_g2", download_enabled=True)

    def read_path(self, path: str) -> Tuple[str, float]:
        img = cv2.imread(path)
        if img is None:
            return "", 0.0
        return self.read_img(img)

    def read_img(self, bgr: np.ndarray) -> Tuple[str, float]:
        variants = build_variants(bgr)
        best_txt, best_conf, best_score = "", 0.0, -999
        for v in variants:
            results = self.reader.readtext(
                v, detail=1, allowlist=ALLOWLIST, decoder="beamsearch",
                text_threshold=0.3, low_text=0.2, link_threshold=0.2,
                paragraph=False, min_size=5, contrast_ths=0.05, adjust_contrast=1.0,
                rotation_info=[0,-5,5],
            )
            for _, txt, conf in results:
                if not txt: continue
                norm = normalize_plate(txt)
                score = plate_validity_score(norm) + int(conf*100)*0.01
                if score > best_score:
                    best_txt, best_conf, best_score = norm, float(conf), score

        if not best_txt:
            flat = self.reader.readtext(variants[0], detail=0, allowlist=ALLOWLIST)
            if isinstance(flat, list) and flat:
                joined = "".join([t for t in flat if isinstance(t, str)])
                best_txt = normalize_plate(joined)
        return best_txt, best_conf
