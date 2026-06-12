import cv2
import numpy as np
import easyocr
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# ROI type: (x, y, width, height) in pixels
ROI = Tuple[int, int, int, int]

# Padding added around ROI crop to avoid cutting edge digits
ROI_PADDING = 12


@dataclass
class OCRResult:
    """Single counter value result from OCRReader.read_counter()"""
    success: bool
    value: Optional[int]       # Extracted counter number (e.g. 18917)
    raw_text: str              # Raw string from OCR before parsing
    confidence: float          # 0.0 – 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "value": self.value,
            "raw_text": self.raw_text,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


@dataclass
class JobCardResult:
    """Job card value read from below the machine counter (alphanumeric or QR)."""
    success: bool
    value: Optional[str]       # Job card number e.g. "JC-4521" (uppercased)
    confidence: float          # 0.0 – 1.0 (QR decode = 1.0)
    source: str                # "qr" | "ocr" | "none"
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


@dataclass
class DetectedNumber:
    """One number detected in the image by read_all_counters()"""
    value: int                 # Parsed integer value
    raw_text: str              # Raw OCR text
    confidence: float          # 0.0 – 1.0
    x: int                     # Bounding box left edge (pixels)
    y: int                     # Bounding box top edge (pixels)
    width: int                 # Bounding box width (pixels)
    height: int                # Bounding box height (pixels)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "raw_text": self.raw_text,
            "confidence": round(self.confidence, 4),
            "bbox": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
        }


class OCRReader:
    """
    Reads numeric machine counter values from camera frames.

    Pipeline:
        Frame -> ROI Crop (with padding) -> Preprocess (OpenCV) -> EasyOCR -> Parse

    Two modes:
        read_counter()      — Returns the single highest-confidence number (use with ROI)
        read_all_counters() — Returns ALL numbers found in the image (use on full display)

    Preprocessing strategies:
        LCD / touchscreen displays  → Otsu threshold  (clean background, uniform lighting)
        LED / 7-segment displays    → Adaptive threshold (uneven glow, high contrast)
        Default: Otsu (covers most industrial display monitors like RMGT)
    """

    # Characters allowed when reading an alphanumeric job card number
    JOBCARD_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/"

    def __init__(self, gpu: bool = False):
        logger.info("Initializing EasyOCR — first run downloads model (~40MB)...")
        self.reader = easyocr.Reader(["en"], gpu=gpu)
        self._qr = cv2.QRCodeDetector()
        logger.info("EasyOCR ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_counter(
        self,
        frame: np.ndarray,
        roi: Optional[ROI] = None,
        sharpness_threshold: float = 80.0,
        preprocess_mode: str = "otsu",
    ) -> OCRResult:
        """
        Extract the single best machine counter value from a frame.
        Use this when the ROI is already tightly cropped to one number.

        Args:
            frame:               BGR image from OpenCV / camera.
            roi:                 (x, y, width, height) region. If None, uses full frame.
            sharpness_threshold: Minimum Laplacian variance to accept a frame.
                                 Frames below this are blurry (digit mid-transition)
                                 and rejected before OCR runs.
                                 Default 80.0 — tuned for RMGT LCD displays.
                                 Lower if too many frames are rejected; raise if misreads persist.

        Returns:
            OCRResult with the highest-confidence integer found.
        """
        try:
            cropped = self._crop_roi_with_padding(frame, roi)

            # ── Sharpness gate ────────────────────────────────────────────
            # Laplacian variance measures edge intensity in the image.
            # A digit mid-transition (physically rolling) creates motion blur →
            # edges soften → variance drops sharply. Reject these frames early
            # rather than running expensive OCR on a blurry digit and getting
            # a wrong number with high confidence.
            sharpness = cv2.Laplacian(
                cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()

            if sharpness < sharpness_threshold:
                logger.debug(
                    f"Frame rejected — sharpness {sharpness:.1f} < threshold {sharpness_threshold}"
                )
                return OCRResult(
                    success=False, value=None, raw_text="",
                    confidence=0.0,
                    error=f"Frame too blurry (sharpness={sharpness:.1f}, threshold={sharpness_threshold}). Digit may be mid-transition.",
                )

            preprocessed = self._preprocess(cropped, mode=preprocess_mode)

            results = self.reader.readtext(
                preprocessed,
                allowlist="0123456789",
                detail=1,
                paragraph=False,
            )

            if not results:
                return OCRResult(
                    success=False, value=None, raw_text="",
                    confidence=0.0, error="No digits detected in region",
                )

            # Pick highest-confidence detection
            best = max(results, key=lambda r: r[2])
            raw_text = best[1].strip()
            confidence = float(best[2])
            digits_only = "".join(filter(str.isdigit, raw_text))
            value = int(digits_only) if digits_only else None

            return OCRResult(
                success=value is not None,
                value=value,
                raw_text=raw_text,
                confidence=confidence,
                error=None if value is not None else "Could not parse digits",
            )

        except Exception as e:
            logger.error(f"read_counter failed: {e}")
            return OCRResult(
                success=False, value=None, raw_text="",
                confidence=0.0, error=str(e),
            )

    def read_counter_consensus(
        self,
        frames: List[np.ndarray],
        roi: Optional[ROI] = None,
        min_confidence: float = 0.85,
        sharpness_threshold: float = 80.0,
        preprocess_mode: str = "otsu",
    ) -> OCRResult:
        """
        Read counter from multiple frames and return the majority-vote result.

        Why this matters:
            Machine counters change 1-3 times per second at high speed.
            Any single frame may catch the digit mid-transition (blurry/partial).
            Reading N frames and taking the most frequent valid result eliminates
            those transition frames automatically.

        How it works:
            1. Run OCR on each frame independently.
            2. Collect all successful readings above min_confidence.
            3. Count how many frames agree on each value (vote count).
            4. Return the value with the highest vote count.
            5. If votes are tied, prefer the value with higher average confidence.

        Args:
            frames:         List of BGR frames captured close together (e.g. 5 frames
                            captured over ~1 second from the live camera).
            roi:            Region of Interest applied to every frame.
            min_confidence: Only accept per-frame readings above this threshold.
                            Default 0.85 — stricter than single-frame reads.

        Returns:
            OCRResult where:
                value      = majority-voted counter value
                confidence = average confidence of all frames that agreed
                raw_text   = "{value} [{votes}/{total} frames agreed]"
        """
        from collections import Counter

        if not frames:
            return OCRResult(
                success=False, value=None, raw_text="",
                confidence=0.0, error="No frames provided",
            )

        votes: Counter = Counter()           # value → count of frames that read it
        confidences: dict = {}               # value → list of confidence scores
        total_attempted = len(frames)
        total_successful = 0

        for frame in frames:
            result = self.read_counter(
                frame, roi=roi, sharpness_threshold=sharpness_threshold,
                preprocess_mode=preprocess_mode,
            )
            if result.success and result.value is not None and result.confidence >= min_confidence:
                votes[result.value] += 1
                confidences.setdefault(result.value, []).append(result.confidence)
                total_successful += 1

        if not votes:
            return OCRResult(
                success=False, value=None, raw_text="",
                confidence=0.0,
                error=f"No confident readings from {total_attempted} frames "
                      f"(min_confidence={min_confidence})",
            )

        # Pick value with most votes; break ties by average confidence
        winner = max(votes, key=lambda v: (votes[v], sum(confidences[v]) / len(confidences[v])))
        avg_confidence = sum(confidences[winner]) / len(confidences[winner])
        vote_count = votes[winner]

        logger.info(
            f"Consensus: {winner} | {vote_count}/{total_attempted} frames agreed "
            f"| avg_conf={avg_confidence:.3f}"
        )

        return OCRResult(
            success=True,
            value=winner,
            raw_text=f"{winner} [{vote_count}/{total_attempted} frames agreed]",
            confidence=avg_confidence,
        )

    def read_jobcard(
        self,
        frame: np.ndarray,
        roi: Optional[ROI] = None,
        min_confidence: float = 0.40,
        sharpness_threshold: float = 20.0,
    ) -> JobCardResult:
        """
        Read the job card placed below the machine counter.

        The job card carries an ALPHANUMERIC job number (e.g. "JC-4521") printed
        large, plus optionally a QR code encoding the same number.

        Strategy (cheapest + most reliable first):
            1. Try QR decode on the raw crop — exact, confidence 1.0, ~1ms.
            2. Fall back to EasyOCR with an alphanumeric allowlist.
               Multiple text boxes on one line are joined left-to-right, but
               small boxes (< 50% of the tallest box) are dropped so the job
               NAME printed smaller under the number doesn't corrupt the value.

        The card is static (magnet-mounted), so the sharpness gate is set low —
        it only rejects truly garbage frames, not the card itself.

        Args:
            frame:           BGR image from OpenCV / camera.
            roi:             (x, y, width, height) of the job card slot below the counter.
            min_confidence:  Per-box OCR confidence threshold.

        Returns:
            JobCardResult with value=None when no card is visible (slot empty).
        """
        try:
            cropped = self._crop_roi_with_padding(frame, roi)

            # Upscale small crops — helps both QR detection and OCR.
            # 560px gives small prefix letters ("JC-") enough pixels to survive.
            h, w = cropped.shape[:2]
            if w < 560:
                scale = 560 / w
                cropped = cv2.resize(cropped, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)

            # ── 1. QR code attempt ────────────────────────────────────────
            try:
                qr_text, points, _ = self._qr.detectAndDecode(cropped)
                if qr_text:
                    return JobCardResult(
                        success=True,
                        value=qr_text.strip().upper(),
                        confidence=1.0,
                        source="qr",
                    )
            except cv2.error:
                pass  # QR detector can throw on degenerate crops — fall through to OCR

            # ── Sharpness gate (low bar — card is static) ─────────────────
            sharpness = cv2.Laplacian(
                cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()
            if sharpness < sharpness_threshold:
                return JobCardResult(
                    success=False, value=None, confidence=0.0, source="none",
                    error=f"Frame too blurry (sharpness={sharpness:.1f})",
                )

            # ── 2. Alphanumeric OCR fallback ──────────────────────────────
            preprocessed = self._preprocess(cropped)

            results = self.reader.readtext(
                preprocessed,
                allowlist=self.JOBCARD_ALLOWLIST,
                detail=1,
                paragraph=False,
            )

            candidates = [
                (bbox, text.strip(), float(conf))
                for bbox, text, conf in results
                if conf >= min_confidence and text.strip()
            ]
            if not candidates:
                return JobCardResult(
                    success=False, value=None, confidence=0.0, source="none",
                    error="No job card text detected (slot may be empty)",
                )

            # The job NUMBER is the tallest text LINE on the card. Group boxes
            # into lines by vertical position and keep only the tallest line —
            # this drops the "JOB CARD" header and "Qty" footer entirely, even
            # when their boxes pass a simple height filter (which caused junk
            # like "JQ8PB1088" / "PB1088IO000" from joined header/footer text).
            def box_height(bbox):
                ys = [p[1] for p in bbox]
                return max(ys) - min(ys)

            def y_center(bbox):
                ys = [p[1] for p in bbox]
                return (max(ys) + min(ys)) / 2

            lines: list = []  # each: {"yc": float, "h": float, "boxes": [...]}
            for cand in sorted(candidates, key=lambda c: y_center(c[0])):
                h = box_height(cand[0])
                yc = y_center(cand[0])
                placed = False
                for line in lines:
                    if abs(yc - line["yc"]) < 0.6 * max(h, line["h"]):
                        line["boxes"].append(cand)
                        line["h"] = max(line["h"], h)
                        placed = True
                        break
                if not placed:
                    lines.append({"yc": yc, "h": h, "boxes": [cand]})

            best_line = max(lines, key=lambda l: l["h"])
            big = best_line["boxes"]

            # Join left-to-right (handles "JC" + "4521" splitting into two boxes)
            big.sort(key=lambda c: min(p[0] for p in c[0]))
            value = "".join(t.replace(" ", "").upper() for _, t, _ in big)
            avg_conf = sum(c for _, _, c in big) / len(big)

            # A job number always contains digits. Pure-letter results are slot
            # labels / painted text ("JOB CARD", "EMPTY") — treat as no card,
            # so the removal miss-counter can run.
            if not any(ch.isdigit() for ch in value):
                return JobCardResult(
                    success=False, value=None, confidence=0.0, source="none",
                    error=f"Text without digits ('{value}') — label/empty slot, not a job card",
                )

            return JobCardResult(
                success=True,
                value=value,
                confidence=avg_conf,
                source="ocr",
            )

        except Exception as e:
            logger.error(f"read_jobcard failed: {e}")
            return JobCardResult(
                success=False, value=None, confidence=0.0, source="none", error=str(e),
            )

    def read_all_counters(
        self,
        frame: np.ndarray,
        roi: Optional[ROI] = None,
        min_confidence: float = 0.4,
        min_height: int = 35,
        max_digits: int = 8,
    ) -> List[DetectedNumber]:
        """
        Detect and return ALL numbers visible in the image.
        Use this on a full machine display to capture every counter at once
        (e.g. job counter=388, sheet count=2700, total=18917).

        Args:
            frame:          BGR image from OpenCV / camera.
            roi:            Optional crop region. If None, scans full frame.
            min_confidence: Discard detections below this threshold (default 0.4).
            min_height:     Minimum bounding box height in pixels (default 35).
                            Filters out small icon/button numbers — real counter
                            digits on industrial displays are always larger.
            max_digits:     Maximum digit count to accept (default 8).
                            Filters out garbled multi-digit strings from noise.

        Returns:
            List of DetectedNumber sorted by confidence descending.
            Duplicates (same value detected twice) are deduplicated — only
            the highest-confidence instance is kept.
        """
        try:
            cropped = self._crop_roi_with_padding(frame, roi)
            preprocessed = self._preprocess(cropped)

            # Scale factor: preprocessing upscales the image, so bbox coords
            # need to be scaled back to match the original (cropped) frame size
            orig_h, orig_w = cropped.shape[:2]
            proc_h, proc_w = preprocessed.shape[:2]
            scale_x = orig_w / proc_w
            scale_y = orig_h / proc_h

            # ROI offset to map coords back to original full frame
            offset_x = max(0, roi[0] - ROI_PADDING) if roi else 0
            offset_y = max(0, roi[1] - ROI_PADDING) if roi else 0

            results = self.reader.readtext(
                preprocessed,
                allowlist="0123456789",
                detail=1,
                paragraph=False,
            )

            detected: List[DetectedNumber] = []

            for bbox, raw_text, confidence in results:
                # Filter 1: confidence threshold
                if confidence < min_confidence:
                    continue

                # Filter 2: must contain digits
                digits_only = "".join(filter(str.isdigit, raw_text.strip()))
                if not digits_only:
                    continue

                # Filter 3: max digits — rejects garbage strings like "12719720254151"
                if len(digits_only) > max_digits:
                    continue

                value = int(digits_only)

                # EasyOCR bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                # Scale back from preprocessed size to original frame size
                xs = [int(p[0] * scale_x) for p in bbox]
                ys = [int(p[1] * scale_y) for p in bbox]
                bw = max(xs) - min(xs)
                bh = max(ys) - min(ys)

                # Filter 4: minimum height — removes small icon/button numbers
                if bh < min_height:
                    continue

                bx = min(xs) + offset_x
                by = min(ys) + offset_y

                detected.append(DetectedNumber(
                    value=value,
                    raw_text=raw_text.strip(),
                    confidence=float(confidence),
                    x=max(0, bx),
                    y=max(0, by),
                    width=bw,
                    height=bh,
                ))

            # Sort by confidence descending
            detected.sort(key=lambda d: d.confidence, reverse=True)

            # Deduplication: if same value appears multiple times, keep highest confidence
            seen: dict = {}
            for det in detected:
                if det.value not in seen:
                    seen[det.value] = det
            deduplicated = list(seen.values())
            deduplicated.sort(key=lambda d: d.confidence, reverse=True)

            return deduplicated

        except Exception as e:
            logger.error(f"read_all_counters failed: {e}")
            return []

    def get_debug_frame(
        self,
        frame: np.ndarray,
        roi: Optional[ROI] = None,
        result: Optional[OCRResult] = None,
        all_results: Optional[List[DetectedNumber]] = None,
        jobcard_roi: Optional[ROI] = None,
        jobcard_value: Optional[str] = None,
    ) -> np.ndarray:
        """
        Annotate frame with ROI box, detected number boxes and values.
        Open via /ocr/debug/{camera_id} in browser to visually verify accuracy.
        """
        debug = frame.copy()

        # Draw ROI region — red box matching the ROI picker UI
        if roi:
            x, y, w, h = roi
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(debug, f"ROI x:{x} y:{y} w:{w} h:{h}", (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Draw job card ROI — blue box below the counter
        if jobcard_roi:
            x, y, w, h = jobcard_roi
            cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 140, 0), 2)
            label = f"JOB: {jobcard_value}" if jobcard_value else "JOB CARD (empty)"
            cv2.putText(debug, label, (x, min(y + h + 18, debug.shape[0] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 2)

        # Draw single result
        if result:
            label = (
                f"Counter: {result.value}  conf:{result.confidence:.2f}"
                if result.success else f"Failed: {result.error}"
            )
            color = (0, 255, 0) if result.success else (0, 0, 255)
            cv2.putText(debug, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Draw all detected numbers with bounding boxes
        if all_results:
            for det in all_results:
                color = (255, 140, 0)
                cv2.rectangle(debug, (det.x, det.y), (det.x + det.width, det.y + det.height), color, 2)
                label = f"{det.value} ({det.confidence:.2f})"
                cv2.putText(debug, label, (det.x, det.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return debug

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_roi_with_padding(self, frame: np.ndarray, roi: Optional[ROI]) -> np.ndarray:
        """
        Crop to ROI with extra padding on all sides.
        Padding prevents edge digits from being clipped during cropping.
        """
        if roi is None:
            return frame

        img_h, img_w = frame.shape[:2]
        x, y, w, h = roi

        # Expand by padding, clamp to image bounds
        x1 = max(0, x - ROI_PADDING)
        y1 = max(0, y - ROI_PADDING)
        x2 = min(img_w, x + w + ROI_PADDING)
        y2 = min(img_h, y + h + ROI_PADDING)

        return frame[y1:y2, x1:x2]

    def _preprocess(self, image: np.ndarray, mode: str = "otsu") -> np.ndarray:
        """
        Preprocessing pipeline for industrial machine display images.

        Pipeline:
          1. Upscale to ≥1200px wide — ensures distant/small digits have enough pixels for OCR
          2. Grayscale / channel select:
               mode="otsu" → standard grayscale (LCD / touchscreen monitors)
               mode="led"  → pick the color channel with the highest contrast.
                             Red LED digits live almost entirely in the R channel;
                             using it suppresses the faint "ghost" (unlit) segments
                             that plain grayscale carries through.
          3. CLAHE (4×4 tile)        — fine-grained local contrast for small distant digits
          4. Gaussian denoise        — removes sensor noise without blurring digit edges
          5. Sharpen (aggressive)    — crisp edges critical for distant digits
          6. Otsu threshold (GLOBAL) — separates lit digits from background.
                             IMPORTANT: global, not adaptive. Adaptive thresholding
                             promotes the faint ghost segments of 7-segment displays
                             to "lit" — tested: it reads 888888 on a DRC-161.
          7. Invert if needed        — handles both light-on-dark and dark-on-light displays
        """
        # 1. Upscale
        _, w = image.shape[:2]
        if w < 600:
            scale = 600 / w
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2. Grayscale / channel select
        if mode == "led" and image.ndim == 3:
            # LED displays: lit segments dominate one color channel (red for DRC-161,
            # green/amber for others). The max-std channel has the best lit-vs-ghost
            # contrast — works for any LED color without configuration.
            b, g, r = cv2.split(image)
            gray = max((b, g, r), key=lambda ch: float(ch.std()))
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE — Contrast Limited Adaptive Histogram Equalization
        #    clipLimit=2.0 suppresses noise amplification; 8×8 tile matches digit size at 600px.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 4. Mild denoise
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # 5. Sharpen digit edges
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

        # 6. GLOBAL Otsu threshold — lit segments are globally much brighter than
        #    ghost segments, so Otsu filters ghosts out. (Adaptive thresholding
        #    must NOT be used here — it turns ghost segments into 8s.)
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 7. Invert if background is darker than text
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)

        return thresh
