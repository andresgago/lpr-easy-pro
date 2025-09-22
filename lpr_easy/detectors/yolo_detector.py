# lpr_easy/detectors/yolo_detector.py
# All comments/docstrings in English.

from typing import List, Tuple
import numpy as np
from ultralytics import YOLO

class YoloPlateDetector:
    """
    Thin wrapper around Ultralytics YOLO for plate detection.
    """
    def __init__(self, weights: str):
        self.model = YOLO(weights)
        # class names exposed by the model (if any)
        self.class_names = self.model.names if hasattr(self.model, "names") else ["plate"]

    def predict(self, img: np.ndarray, conf: float, imgsz: int) -> List[Tuple[int,int,int,int,float,int]]:
        """
        Run detection on a single image and return a list of detections:
        (x1, y1, x2, y2, score, cls_id)
        """
        results = self.model.predict(img, conf=conf, imgsz=imgsz, verbose=False)
        r = results[0]
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            h, w = img.shape[:2]
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                score = float(b.conf[0].cpu().numpy()) if b.conf is not None else 0.0
                cls_id = int(b.cls[0].cpu().numpy()) if b.cls is not None else 0
                x1, y1, x2, y2 = xyxy
                x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
                y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
                if x2 > x1 and y2 > y1:
                    dets.append((x1, y1, x2, y2, score, cls_id))
        return dets
