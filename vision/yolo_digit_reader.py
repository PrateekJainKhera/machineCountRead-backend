"""
YOLOv8-based digit reader for machine counter displays.

How it works:
    1. Run YOLOv8 on the (preprocessed) ROI crop.
    2. Each detection = one digit (class 0–9) with a bounding box.
    3. Sort detections left-to-right by x-coordinate.
    4. Concatenate digit classes to form the full counter number.

Why this is more accurate than EasyOCR for LCD displays:
    - Trained specifically on digit images (not general text).
    - Faster on CPU (~20–50ms vs EasyOCR's 200–500ms).
    - No confusion between letters and digits (allowlist not needed).
    - Handles partial occlusion better (each digit detected independently).

Usage:
    reader = YOLODigitReader("vision/digit_model.pt")
    result = reader.read_counter(frame, roi=(x, y, w, h))
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
from ultralytics import YOLO

from vision.ocr_reader import OCRResult, ROI, ROI_PADDING

logger = logging.getLogger(__name__)


class YOLODigitReader:
    """
    Reads numeric counter values from camera frames using a trained YOLOv8 model.
    Drop-in replacement for OCRReader — same read_counter() interface.
    """

    # Per-digit detection threshold — deliberately low; the real acceptance
    # gate is min_confidence applied to the average over the whole number.
    DETECT_CONF = 0.40

    def __init__(self, model_path: str = "vision/digit_model.pt"):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YOLO digit model not found at '{model_path}'. "
                "Run backend/train_digit_model.py first."
            )
        logger.info(f"Loading YOLO digit model from {model_path}...")
        self.model = YOLO(model_path)
        logger.info("YOLO digit model ready.")

    def read_counter(
        self,
        frame: np.ndarray,
        roi: Optional[ROI] = None,
        sharpness_threshold: float = 40.0,
        min_confidence: float = 0.60,
        preprocess_mode: str = "otsu",  # accepted for OCRReader interface parity — YOLO works on raw frames
    ) -> OCRResult:
        """
        Extract the machine counter value from a frame using YOLO digit detection.

        Args:
            frame:               BGR image from OpenCV / camera.
            roi:                 (x, y, width, height) region. If None, uses full frame.
            sharpness_threshold: Minimum Laplacian variance. Lower than EasyOCR default
                                 because YOLO is more robust to mild blur.
            min_confidence:      Minimum per-digit detection confidence (0–1).
                                 Digits below this are discarded.

        Returns:
            OCRResult with the counter value formed by concatenating detected digits.
        """
        try:
            cropped = self._crop_roi(frame, roi)

            # Sharpness gate
            sharpness = cv2.Laplacian(
                cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()
            if sharpness < sharpness_threshold:
                return OCRResult(
                    success=False, value=None, raw_text="",
                    confidence=0.0,
                    error=f"Frame too blurry (sharpness={sharpness:.1f})",
                )

            # Upscale small crops — YOLO also works better with larger input
            h, w = cropped.shape[:2]
            if w < 320:
                scale = 320 / w
                cropped = cv2.resize(cropped, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)

            # Run YOLO inference at a LOW detection threshold.
            # min_confidence must NOT be the detection threshold: a weak digit
            # would silently vanish and "12565" becomes "2565" — a wrong number
            # with high confidence. Detect everything plausible, then gate the
            # AVERAGE confidence of the full number below.
            results = self.model(cropped, verbose=False, conf=self.DETECT_CONF)

            if not results or len(results[0].boxes) == 0:
                return OCRResult(
                    success=False, value=None, raw_text="",
                    confidence=0.0, error="No digits detected",
                )

            boxes = results[0].boxes

            # Each box: class = digit (0–9), xyxy = bounding box, conf = confidence
            detections = []
            for box in boxes:
                cls   = int(box.cls[0].item())
                conf  = float(box.conf[0].item())
                x1    = float(box.xyxy[0][0].item())
                detections.append((x1, cls, conf))

            if not detections:
                return OCRResult(
                    success=False, value=None, raw_text="",
                    confidence=0.0, error="No confident digit detections",
                )

            # Sort left to right by x-coordinate to form the number
            detections.sort(key=lambda d: d[0])

            digits    = [str(d[1]) for d in detections]
            confs     = [d[2] for d in detections]
            raw_text  = "".join(digits)
            avg_conf  = float(np.mean(confs))
            value     = int(raw_text)

            logger.debug(
                f"YOLO detected digits: {raw_text} "
                f"(avg_conf={avg_conf:.2f}, n={len(digits)})"
            )

            # Acceptance gate on the WHOLE number, not per digit
            if avg_conf < min_confidence:
                return OCRResult(
                    success=False, value=value, raw_text=raw_text,
                    confidence=avg_conf,
                    error=f"Average digit confidence {avg_conf:.2f} below {min_confidence}",
                )

            return OCRResult(
                success=True,
                value=value,
                raw_text=raw_text,
                confidence=avg_conf,
            )

        except Exception as e:
            logger.error(f"YOLODigitReader.read_counter failed: {e}")
            return OCRResult(
                success=False, value=None, raw_text="",
                confidence=0.0, error=str(e),
            )

    def read_counter_consensus(
        self,
        frames: list,
        roi: Optional[ROI] = None,
        min_confidence: float = 0.60,
        sharpness_threshold: float = 40.0,
        preprocess_mode: str = "otsu",  # interface parity with OCRReader — unused
    ) -> OCRResult:
        """
        Majority-vote across multiple frames. Same interface as OCRReader.read_counter_consensus().
        """
        from collections import defaultdict

        votes: dict = defaultdict(list)

        for frame in frames:
            result = self.read_counter(
                frame, roi=roi, min_confidence=min_confidence,
                sharpness_threshold=sharpness_threshold,
            )
            if result.success and result.value is not None:
                votes[result.value].append(result.confidence)

        if not votes:
            return OCRResult(
                success=False, value=None, raw_text="",
                confidence=0.0, error="No valid readings across all frames",
            )

        # Pick value with most votes; break ties by average confidence
        best_value = max(votes, key=lambda v: (len(votes[v]), float(np.mean(votes[v]))))
        best_conf  = float(np.mean(votes[best_value]))

        return OCRResult(
            success=True,
            value=best_value,
            raw_text=str(best_value),
            confidence=best_conf,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_roi(self, frame: np.ndarray, roi: Optional[ROI]) -> np.ndarray:
        if roi is None:
            return frame
        img_h, img_w = frame.shape[:2]
        x, y, w, h = roi
        x1 = max(0, x - ROI_PADDING)
        y1 = max(0, y - ROI_PADDING)
        x2 = min(img_w, x + w + ROI_PADDING)
        y2 = min(img_h, y + h + ROI_PADDING)
        return frame[y1:y2, x1:x2]
