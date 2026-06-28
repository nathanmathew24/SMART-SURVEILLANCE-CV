"""
detection.py - Object Detection using YOLOv8 nano (yolov8n.pt).

YOLOv8 (You Only Look Once v8) is a single-stage, anchor-free object detector
from Ultralytics.  The nano variant trades a small accuracy reduction for very
fast inference, making it suitable for real-time surveillance at ≥10 FPS on
CPU-only hardware.

CV Technique #2a: Object Detection
"""

import os
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from utils import draw_box, get_color

# Default weight location – will auto-download on first run if absent
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"


class ObjectDetector:
    """YOLOv8-nano object detector.

    Parameters
    ----------
    model_path:   Path to the .pt weights file.  Auto-downloads if missing.
    conf:         Minimum confidence threshold (0–1).
    iou:          IoU threshold for Non-Maximum Suppression.
    device:       Inference device ('cpu', '0', 'cuda:0', etc.).
    classes:      Optional list of COCO class IDs to keep (None → all).
    """

    def __init__(
        self,
        model_path: str | Path = _MODEL_PATH,
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "cpu",
        classes: list[int] | None = None,
    ) -> None:
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed.  Run: pip install ultralytics")

        self.conf = conf
        self.iou = iou
        self.device = device
        self.classes = classes

        # Load model (Ultralytics auto-downloads if path doesn't exist)
        self.model = YOLO(str(model_path))
        self.model.to(device)

        # Cache class names from the model
        self.class_names: dict[int, str] = self.model.names  # type: ignore[assignment]

    def detect(self, frame: np.ndarray) -> dict:
        """Run inference on a single BGR frame.

        Returns
        -------
        dict with keys:
          - "annotated":   BGR frame with boxes and labels drawn.
          - "detections":  List of dicts, each with:
                             class_id, class_name, confidence, bbox (x1,y1,x2,y2)
          - "count":       Total number of detections.
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )

        annotated = frame.copy()
        detections: list[dict] = []

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for box in boxes:
                # Extract values from Ultralytics Boxes object
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, str(cls_id))

                color = get_color(cls_id)
                label = f"{cls_name} {conf_val:.0%}"
                draw_box(annotated, x1, y1, x2, y2, label, color)

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_val, 3),
                    "bbox": (x1, y1, x2, y2),
                })

        return {
            "annotated": annotated,
            "detections": detections,
            "count": len(detections),
        }
