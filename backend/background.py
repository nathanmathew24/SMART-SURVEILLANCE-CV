"""
background.py - Background Subtraction / Motion Detection using OpenCV MOG2.

MOG2 (Mixture of Gaussians v2) is an adaptive background modelling algorithm
that represents each pixel as a mixture of Gaussian distributions.  Moving
foreground objects cause pixels to deviate from the learned background model,
producing a foreground mask that highlights motion.

CV Technique #1: Background Subtraction / Change Detection
"""

import cv2
import numpy as np

from utils import draw_label


class BackgroundSubtractor:
    """Wraps OpenCV's MOG2 background subtractor with noise filtering.

    Parameters
    ----------
    history:        Number of frames used to build the background model.
    var_threshold:  Mahalanobis distance threshold for foreground/background
                    classification.  Lower → more sensitive to motion.
    detect_shadows: Whether MOG2 marks shadows (gray pixels) in the mask.
    min_area:       Minimum contour area (px²) to report as a motion region.
    """

    def __init__(
        self,
        history: int = 200,
        var_threshold: float = 40.0,
        detect_shadows: bool = True,
        min_area: int = 500,
        learning_rate: float = -1,
    ) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.min_area = min_area
        self.learning_rate = learning_rate  # -1 → automatic

        # Morphological kernel for noise removal
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def reset(self) -> None:
        """Force the background model to restart learning from scratch."""
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=True
        )

    def process(self, frame: np.ndarray) -> dict:
        """Run background subtraction on one frame.

        Returns
        -------
        dict with keys:
          - "annotated":      BGR frame with motion regions drawn.
          - "mask":           Binary foreground mask (uint8, 0/255).
          - "motion_regions": List of (x, y, w, h) bounding rects for each
                              contour whose area exceeds *min_area*.
          - "motion_detected": True if any valid region was found.
        """
        # 1. Apply MOG2 to obtain the raw foreground mask
        fg_mask = self._subtractor.apply(frame, learningRate=self.learning_rate)

        # 2. Remove shadow pixels (value 127) – keep only definite foreground
        _, binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 3. Morphological open (erode then dilate) to remove small noise blobs
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self._kernel)

        # 4. Dilate to fill holes inside moving objects
        cleaned = cv2.dilate(cleaned, self._kernel, iterations=2)

        # 5. Find contours of foreground blobs
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_regions: list[tuple[int, int, int, int]] = []
        annotated = frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            motion_regions.append((x, y, w, h))

            # Draw motion bounding box in red
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            draw_label(annotated, f"Motion ({int(area)}px)", x, y, (0, 0, 255))

        # Overlay a semi-transparent green tint on motion pixels
        motion_colour = np.zeros_like(frame)
        motion_colour[cleaned > 0] = (0, 200, 0)
        cv2.addWeighted(motion_colour, 0.35, annotated, 1.0, 0, annotated)

        return {
            "annotated": annotated,
            "mask": cleaned,
            "motion_regions": motion_regions,
            "motion_detected": len(motion_regions) > 0,
        }
