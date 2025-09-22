# lpr_easy/config.py
# All comments/docstrings in English.

from dataclasses import dataclass
from typing import Optional

@dataclass
class AppConfig:
    # Inputs
    input_dir: Optional[str] = None
    image: Optional[str] = None
    pattern: str = "**/*.*"

    # Detection
    weights: str = ""
    square_size: int = 640
    conf: float = 0.25

    # Outputs
    save_pre: Optional[str] = None
    save_vis: Optional[str] = None
    save_crops: Optional[str] = None
    csv: Optional[str] = None
    out: Optional[str] = None
    fmt: str = "csv"
    name_with_plate: bool = False

    # OCR
    ocr: str = "none"  # "none" | "easyocr-plus"
    ocr_gpu: Optional[str] = None  # "true" | "false" | None (autodetect)
    ocr_out: Optional[str] = None  # optional separate OCR-only CSV/JSON
