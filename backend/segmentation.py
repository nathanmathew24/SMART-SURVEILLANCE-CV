"""
segmentation.py - Instance Segmentation using YOLOv8n-seg.

YOLOv8n-seg extends the base YOLOv8 detector with a lightweight mask head
that predicts a per-instance binary segmentation mask in addition to the
bounding box.  Each mask precisely outlines the detected object.

CV Technique #4: Image / Instance Segmentation
"""

from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from utils import get_color

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n-seg.pt"


class Segmenter:
    """YOLOv8-nano instance segmentation.

    Parameters
    ----------
    model_path:   Path to yolov8n-seg.pt weights.
    conf:         Detection confidence threshold.
    iou:          NMS IoU threshold.
    device:       Torch device string.
    mask_alpha:   Opacity of the filled mask overlay (0 = invisible, 1 = solid).
    """

    def __init__(
        self,
        model_path: str | Path = _MODEL_PATH,
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "cpu",
        mask_alpha: float = 0.4,
    ) -> None:
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed.")

        self.conf = conf
        self.iou = iou
        self.device = device
        self.mask_alpha = mask_alpha

        self.model = YOLO(str(model_path))
        self.model.to(device)
        self.class_names: dict[int, str] = self.model.names  # type: ignore[assignment]

    def segment(self, frame: np.ndarray) -> dict:
        """Run instance segmentation on a BGR frame.

        Returns
        -------
        dict with keys:
          - "annotated":   BGR frame with coloured mask overlays and boxes.
          - "segments":    List of dicts: class_id, class_name, confidence,
                           bbox, mask (bool 2-D array, same H×W as frame).
          - "count":       Number of segmented instances.
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        annotated = frame.copy()
        overlay = frame.copy()
        segments: list[dict] = []

        for result in results:
            if result.masks is None or result.boxes is None:
                continue

            masks_data = result.masks.data.cpu().numpy()   # (N, H, W) float32
            h, w = frame.shape[:2]

            boxes = result.boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = self.class_names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                color = get_color(cls_id)

                # Resize mask from model output resolution to frame resolution
                mask_raw = masks_data[i]
                mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_bool = mask_resized > 0.5

                # Fill mask region on overlay with object colour
                overlay[mask_bool] = color

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label above box
                label = f"{cls_name} {conf_val:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(
                    annotated, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )

                segments.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_val, 3),
                    "bbox": (x1, y1, x2, y2),
                    "mask": mask_bool,
                })

        # Blend the coloured mask overlay with the annotated frame
        cv2.addWeighted(overlay, self.mask_alpha, annotated, 1 - self.mask_alpha, 0, annotated)

        return {
            "annotated": annotated,
            "segments": segments,
            "count": len(segments),
        }
