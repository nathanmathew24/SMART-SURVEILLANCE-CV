"""
tracking.py - Object Tracking using YOLOv8 nano with ByteTrack.

Tracking extends detection by assigning a persistent integer ID to each object
across frames.  We use Ultralytics' built-in ByteTrack integration which runs
on top of YOLOv8 detections and maintains tracklets using a Kalman filter +
IoU matching strategy.

CV Technique #2b: Object Tracking
"""

from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from utils import draw_label, get_color

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"


class ObjectTracker:
    """YOLOv8-nano detector with ByteTrack multi-object tracking.

    Parameters
    ----------
    model_path:  Path to yolov8n.pt weights.
    conf:        Detection confidence threshold.
    iou:         NMS IoU threshold.
    device:      Torch device string.
    persist:     Persist track IDs across calls (set False to reset on each
                 new video stream).
    """

    def __init__(
        self,
        model_path: str | Path = _MODEL_PATH,
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "cpu",
        persist: bool = True,
    ) -> None:
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed.")

        self.conf = conf
        self.iou = iou
        self.device = device
        self.persist = persist

        self.model = YOLO(str(model_path))
        self.model.to(device)
        self.class_names: dict[int, str] = self.model.names  # type: ignore[assignment]

        # Track history: track_id → list of (cx, cy) centroids
        self._history: dict[int, list[tuple[int, int]]] = {}
        self._max_history = 30  # keep last N centroid positions for trail

    def reset(self) -> None:
        """Clear track history (call when switching to a new video source)."""
        self._history.clear()

    def track(self, frame: np.ndarray) -> dict:
        """Run detection + tracking on a single BGR frame.

        Returns
        -------
        dict with keys:
          - "annotated":  BGR frame with boxes, IDs, and motion trails.
          - "tracks":     List of dicts: track_id, class_id, class_name,
                          confidence, bbox (x1,y1,x2,y2).
          - "count":      Active track count.
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            persist=self.persist,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        annotated = frame.copy()
        tracks: list[dict] = []

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = self.class_names.get(cls_id, str(cls_id))

                color = get_color(track_id)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                draw_label(annotated, f"ID:{track_id} {cls_name} {conf_val:.0%}",
                           x1, y1, color)

                # Update centroid history and draw motion trail
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                history = self._history.setdefault(track_id, [])
                history.append((cx, cy))
                if len(history) > self._max_history:
                    history.pop(0)

                for i in range(1, len(history)):
                    thickness = max(1, int(3 * i / len(history)))
                    cv2.line(annotated, history[i - 1], history[i], color, thickness)

                tracks.append({
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_val, 3),
                    "bbox": (x1, y1, x2, y2),
                })

        # Prune history for IDs no longer active this frame
        active_ids = {t["track_id"] for t in tracks}
        for tid in list(self._history.keys()):
            if tid not in active_ids:
                # keep a few frames of history then remove
                if len(self._history[tid]) > self._max_history:
                    del self._history[tid]

        return {
            "annotated": annotated,
            "tracks": tracks,
            "count": len(tracks),
        }
