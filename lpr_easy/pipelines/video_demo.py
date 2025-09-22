# lpr_easy/pipelines/video_demo.py
# Annotate a driving video with YOLO plate detections only (no OCR, no labels).

from pathlib import Path
import os
import cv2
import numpy as np

# Allow running as module or as script
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from lpr_easy.detectors.yolo_detector import YoloPlateDetector
else:
    from ..detectors.yolo_detector import YoloPlateDetector


def draw_box(frame: np.ndarray, bbox):
    """Draw a green rectangle only."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def process_video(weights: str,
                  input_video: str,
                  output_video: str,
                  conf: float = 0.25,
                  square_size: int = 640,
                  output_max_width: int | None = None,
                  fps_out: float | None = None):
    """
    Run YOLO plate detection per frame and save an annotated MP4.
    Only rectangles are drawn (no text).
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional downscale for output
    if output_max_width and w_in > output_max_width:
        scale = output_max_width / float(w_in)
        w_out = output_max_width
        h_out = int(h_in * scale)
    else:
        w_out, h_out = w_in, h_in

    fps = fps_out or fps_in

    # Prefer H.264 if available; fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w_out, h_out))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (w_out, h_out))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {output_video}")

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    detector = YoloPlateDetector(weights)

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = detector.predict(frame, conf=conf, imgsz=square_size)

        # Resize for output if needed
        if (w_out, h_out) != (w_in, h_in):
            frame_draw = cv2.resize(frame, (w_out, h_out))
            sx_d = w_out / float(w_in)
            sy_d = h_out / float(h_in)
        else:
            frame_draw = frame
            sx_d = sy_d = 1.0

        # Draw only boxes
        for (x1, y1, x2, y2, _score, _cls_id) in dets:
            x1s = int(x1 * sx_d);
            y1s = int(y1 * sy_d)
            x2s = int(x2 * sx_d);
            y2s = int(y2 * sy_d)
            draw_box(frame_draw, (x1s, y1s, x2s, y2s))

        writer.write(frame_draw)
        frames += 1
        if frames % 50 == 0:
            print(f"[INFO] processed {frames} frames...")

    cap.release()
    writer.release()
    print(f"[OK] Annotated video written: {output_video} ({frames} frames)")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Annotate a video with YOLO plate detections only (no OCR)")
    p.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    p.add_argument("--input_video", required=True, help="Path to input video (mp4)")
    p.add_argument("--output_video", required=True, help="Path to output annotated video (mp4)")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--square_size", type=int, default=640, help="YOLO inference size (imgsz)")
    p.add_argument("--output-max-width", type=int, default=None, help="Optional output max width (keeps aspect).")
    p.add_argument("--fps-out", type=float, default=None, help="Optional output FPS override.")
    args = p.parse_args()

    process_video(
        weights=args.weights,
        input_video=args.input_video,
        output_video=args.output_video,
        conf=args.conf,
        square_size=args.square_size,
        output_max_width=args.output_max_width,
        fps_out=args.fps_out,
    )
