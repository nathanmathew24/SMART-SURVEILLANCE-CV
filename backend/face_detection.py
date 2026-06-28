"""
face_detection.py - Face Detection using MediaPipe Face Detection.

MediaPipe Face Detection is a lightweight, ML-based face detector developed by
Google that runs entirely on CPU at real-time speeds.  It outputs bounding
boxes and six facial key-point landmarks (eyes, nose tip, mouth corners, ears)
per detected face.

CV Technique #3: Face Detection
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


class FaceDetector:
    """MediaPipe-based face detector (compatible with MediaPipe 0.10+).

    Uses the new Tasks API (FaceDetector task) which replaced the deprecated
    mp.solutions.face_detection interface in recent MediaPipe versions.
    Falls back to the legacy solutions API if the Tasks API is unavailable.
    """

    _LANDMARK_NAMES = [
        "right_eye", "left_eye", "nose_tip",
        "mouth_center", "right_ear", "left_ear",
    ]

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
    ) -> None:
        if not MP_AVAILABLE:
            raise ImportError("mediapipe is not installed.  Run: pip install mediapipe")

        self._use_tasks = False
        self._detector = None

        # Try the new Tasks API first (MediaPipe 0.10.14+)
        try:
            import urllib.request, os, tempfile
            # Download the face detection model if needed
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            )
            model_path = os.path.join(tempfile.gettempdir(), "blaze_face_short_range.tflite")
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(model_url, model_path)

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=min_detection_confidence,
            )
            self._detector = mp_vision.FaceDetector.create_from_options(options)
            self._use_tasks = True
        except Exception:
            # Fall back to legacy solutions API
            try:
                self._mp_face = mp.solutions.face_detection  # type: ignore[attr-defined]
                self._detector = self._mp_face.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=min_detection_confidence,
                )
                self._use_tasks = False
            except Exception as exc:
                raise RuntimeError(f"Could not initialise MediaPipe face detector: {exc}")

    def detect(self, frame: np.ndarray) -> dict:
        """Detect faces in a BGR frame."""
        if self._use_tasks:
            return self._detect_tasks(frame)
        return self._detect_legacy(frame)

    def _detect_tasks(self, frame: np.ndarray) -> dict:
        """Detection using the new MediaPipe Tasks API."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        annotated = frame.copy()
        faces = []

        for detection in result.detections:
            score = float(detection.categories[0].score)
            bb = detection.bounding_box
            x1 = max(0, bb.origin_x)
            y1 = max(0, bb.origin_y)
            x2 = min(w - 1, bb.origin_x + bb.width)
            y2 = min(h - 1, bb.origin_y + bb.height)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(
                annotated, f"Face {score:.0%}", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1, cv2.LINE_AA,
            )

            landmarks = []
            if detection.keypoints:
                for i, kp in enumerate(detection.keypoints):
                    px, py = int(kp.x * w), int(kp.y * h)
                    cv2.circle(annotated, (px, py), 4, (0, 255, 255), -1)
                    name = self._LANDMARK_NAMES[i] if i < len(self._LANDMARK_NAMES) else f"kp{i}"
                    landmarks.append({"name": name, "x": px, "y": py})

            faces.append({"bbox": (x1, y1, x2, y2), "confidence": round(score, 3), "landmarks": landmarks})

        return {"annotated": annotated, "faces": faces, "count": len(faces)}

    def _detect_legacy(self, frame: np.ndarray) -> dict:
        """Detection using the legacy mp.solutions API."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)

        annotated = frame.copy()
        faces = []

        if not result.detections:
            return {"annotated": annotated, "faces": faces, "count": 0}

        for detection in result.detections:
            score = float(detection.score[0])
            bb = detection.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w - 1, int((bb.xmin + bb.width) * w))
            y2 = min(h - 1, int((bb.ymin + bb.height) * h))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(
                annotated, f"Face {score:.0%}", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1, cv2.LINE_AA,
            )

            landmarks = []
            for i, kp in enumerate(detection.location_data.relative_keypoints):
                px, py = int(kp.x * w), int(kp.y * h)
                cv2.circle(annotated, (px, py), 4, (0, 255, 255), -1)
                name = self._LANDMARK_NAMES[i] if i < len(self._LANDMARK_NAMES) else f"kp{i}"
                landmarks.append({"name": name, "x": px, "y": py})

            faces.append({"bbox": (x1, y1, x2, y2), "confidence": round(score, 3), "landmarks": landmarks})

        return {"annotated": annotated, "faces": faces, "count": len(faces)}

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._detector and hasattr(self._detector, "close"):
            self._detector.close()
