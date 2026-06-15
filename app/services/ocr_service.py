import cv2
import time
import threading
import numpy as np
import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from vision.camera_manager import CameraManager
from vision.ocr_reader import OCRReader, OCRResult, DetectedNumber, JobCardResult
from app.models.ocr_model import ROIConfig
from app.utils.counter_validator import validate_reading, ValidationResult
from app.utils import jobcard as jc_util

logger = logging.getLogger(__name__)

# ROI tuple type used internally
ROI = Tuple[int, int, int, int]

# How many past valid readings to keep per camera for rate estimation
RATE_HISTORY_SIZE = 10

# How often the background poller reads from a camera (seconds)
# Video files use a short interval to catch every counter change across the clip.
# Live cameras (webcam/RTSP/screen) use a longer interval — the counter changes
# slowly and frequent OCR saturates the CPU, making the stream feel laggy.
POLL_INTERVAL_VIDEO = 0.2   # one-shot video files
POLL_INTERVAL_LIVE  = 0.15  # webcam / RTSP / screen capture
                            # (cycle time is dominated by EasyOCR CPU inference,
                            #  not this sleep — see LIVE_BURST_FRAMES below)

# Extra reads on the last frozen frame after a one-shot video ends
VIDEO_DRAIN_READS = 3

# Live camera accuracy: burst-read N frames spaced LIVE_BURST_GAP seconds apart,
# then use consensus voting (majority vote across frames) to eliminate transition misreads.
# A digit mid-roll blurs 1 frame; the other 2 frames catch it stable → vote wins.
# Total burst cost: (LIVE_BURST_FRAMES - 1) × LIVE_BURST_GAP = 0.1s extra per cycle.
LIVE_BURST_FRAMES = 3        # frames to collect per live poll cycle
                             # Counter reader is the YOLO digit model (~20-50ms/frame
                             # on CPU), so a 3-frame consensus burst costs ~0.25s
                             # total — full transition protection at ~3 readings/sec.
                             # (If the YOLO model is missing and EasyOCR fallback is
                             # active, each frame costs ~1s — drop this to 2.)
LIVE_BURST_GAP    = 0.05     # seconds between frame samples in a burst
LIVE_MIN_CONFIDENCE = 0.85   # acceptance bar for live cameras (video mode uses 0.40).
                             # Empirical (50-reading phone test on SIMATIC HMI):
                             # every misread scored 72-84%, every correct read
                             # clustered 89-100% — 0.85 cleanly separates them.
                             # Cost: low-conf CORRECT frames are skipped too;
                             # for long-hours running, skipped > wrong.

# Job card reading: the card is static (magnet-mounted), so it doesn't need to be
# read every cycle. Reading it every Nth counter poll keeps CPU free for the counter.
# When a change is suspected (pending value or misses), the poller switches to
# EVERY cycle until resolved, so card swaps/removals are detected fast.
JOBCARD_POLL_EVERY = 2       # read job card every Nth poll cycle (~5s on live)
JOBCARD_MISS_LIMIT = 3       # consecutive failed reads before card considered REMOVED

# A camera's FIRST counter reading has no baseline to validate against, so a
# partial read at startup would poison the baseline. Require this many
# consecutive consistent readings before establishing it.
FIRST_READING_CONFIRM = 2

# Rejection-streak recovery: if validation rejects the SAME value (±20) this many
# times in a row (with good confidence), the display genuinely changed — adopt it.
REJECT_STREAK_ADOPT = 5


class OCRService:
    """
    Application-level service that manages:
        - One shared OCRReader instance (EasyOCR loads once)
        - Multiple CameraManager instances (one per camera)
        - Per-camera ROI configuration
        - Per-camera background polling thread (reads continuously, caches latest valid result)
        - Per-camera rolling rate history (last N valid readings → estimates counts/sec)
        - Per-camera last-accepted reading (for rate validation)

    Background polling is the key design here:
        The poller thread runs OCR every POLL_INTERVAL seconds and caches the result.
        API calls return from cache instantly — no OCR latency on each HTTP request.
        This decouples API response time from EasyOCR inference time (~200–500ms).

    This is created once at app startup and shared across all requests.
    """

    def __init__(self):
        logger.info("Starting OCRService...")
        # EasyOCR — job card (alphanumeric), read-all scans, image uploads
        self._reader = OCRReader(gpu=False)

        # YOLOv8 digit model — THE counter reader. Trained on display digits
        # (7-segment native), ~20-50ms/frame on CPU vs EasyOCR's ~1s.
        # Falls back to EasyOCR only if the model file is missing.
        try:
            from vision.yolo_digit_reader import YOLODigitReader
            self._counter_reader = YOLODigitReader("vision/digit_model.pt")
            logger.info("Counter reader: YOLOv8 digit model (vision/digit_model.pt)")
        except Exception as e:
            logger.warning(f"YOLO digit model unavailable ({e}) — counter falls back to EasyOCR.")
            self._counter_reader = self._reader

        self._cameras: Dict[str, CameraManager] = {}
        self._rois: Dict[str, ROI] = {}

        # Job card slot below the counter (second ROI per camera)
        self._jobcard_rois: Dict[str, ROI] = {}
        # camera_id → {"value", "confidence", "source", "present", "since",
        #              "last_seen", "misses"}
        self._jobcard_state: Dict[str, dict] = {}

        # Per-camera preprocessing mode: "otsu" (LCD) or "adaptive" (LED/7-segment)
        self._preprocess_modes: Dict[str, str] = {}

        # Human machine identifier per camera (e.g. "FLEXO-1") — used on
        # downtime events and reports to say WHICH machine paused.
        self._machine_ids: Dict[str, str] = {}

        # Tracks last accepted (valid) reading per camera for rate validation
        self._last_accepted: Dict[str, dict] = {}  # camera_id → {value, timestamp}
        # Per-camera max counter rate (units/second) for jump validation.
        # Default 30/sec: flexo label presses hit ~13-15 labels/sec at 39 m/min
        # (measured on the GIDOE SIMATIC HMI) and run faster at full speed.
        self._max_rates: Dict[str, float] = {}
        self._default_max_rate: float = 30.0

        # Rolling history of last N valid readings for rate estimation
        # Each entry: {"value": int, "timestamp": datetime}
        self._rate_history: Dict[str, deque] = {}

        # Consecutive consistent rejections per camera (for baseline recovery)
        # camera_id → {"value": int, "count": int, "last": datetime}
        self._reject_streaks: Dict[str, dict] = {}

        # First-reading confirmation per camera (baseline must repeat)
        # camera_id → {"value": int, "count": int, "last": datetime}
        self._first_pending: Dict[str, dict] = {}

        # Latest valid cached result from the background poller
        # Each entry: {"result": OCRResult, "validation": ValidationResult|None,
        #              "polled_at": datetime, "rate_per_second": float|None}
        self._latest_valid: Dict[str, dict] = {}

        # Background poller threads
        self._poll_running: Dict[str, bool] = {}
        self._pollers: Dict[str, threading.Thread] = {}

        logger.info("OCRService ready.")

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def register_camera(
        self,
        camera_id: str,
        source: str,
        roi: Optional[ROIConfig] = None,
        loop: bool = True,
        jobcard_roi: Optional[ROIConfig] = None,
        display_type: str = "lcd",
        max_rate_per_second: Optional[float] = None,
        machine_id: Optional[str] = None,
    ) -> None:
        """
        Register a camera and start its capture thread + background OCR poller.

        Args:
            camera_id:    Unique identifier (e.g. 'machine_a_display').
            source:       RTSP URL, file path, or '0' for webcam.
            roi:          Optional region of interest for the counter.
            jobcard_roi:  Optional region of the job card slot below the counter.
            display_type: "lcd" (Otsu threshold) or "led" (best-channel mode,
                          for LED / 7-segment counter displays).
            max_rate_per_second: Machine's max counter speed for jump validation.
                          Defaults to 30/sec (flexo label press territory).
        """
        # Stop existing camera/poller with the same ID
        if camera_id in self._cameras:
            self._stop_poller(camera_id)
            self._cameras[camera_id].stop()

        parsed_source = int(source) if source.isdigit() else source

        cam = CameraManager(source=parsed_source, camera_id=camera_id, loop=loop)
        cam.start()
        self._cameras[camera_id] = cam

        if roi:
            self._rois[camera_id] = (roi.x, roi.y, roi.width, roi.height)
        if jobcard_roi:
            self._jobcard_rois[camera_id] = (
                jobcard_roi.x, jobcard_roi.y, jobcard_roi.width, jobcard_roi.height
            )
        self._preprocess_modes[camera_id] = "led" if display_type == "led" else "otsu"
        if max_rate_per_second is not None and max_rate_per_second > 0:
            self._max_rates[camera_id] = max_rate_per_second
        self._machine_ids[camera_id] = (machine_id or camera_id).strip() or camera_id

        # Initialise per-camera state (clear ALL history on re-registration)
        self._rate_history[camera_id] = deque(maxlen=RATE_HISTORY_SIZE)
        self._latest_valid.pop(camera_id, None)
        self._last_accepted.pop(camera_id, None)  # reset baseline so rate check starts fresh
        self._jobcard_state.pop(camera_id, None)
        self._reject_streaks.pop(camera_id, None)
        self._first_pending.pop(camera_id, None)

        # Start background OCR poller
        self._start_poller(camera_id)

        logger.info(
            f"Camera '{camera_id}' registered. ROI: {roi} | jobcard ROI: {jobcard_roi} "
            f"| display: {display_type}"
        )

    def set_jobcard_roi(self, camera_id: str, roi: ROIConfig) -> bool:
        """Set/update the job card slot ROI for an existing camera. Returns False if not found."""
        if camera_id not in self._cameras:
            return False
        self._jobcard_rois[camera_id] = (roi.x, roi.y, roi.width, roi.height)
        # New slot location — old card state is meaningless
        self._jobcard_state.pop(camera_id, None)
        logger.info(f"Job card ROI updated for camera '{camera_id}': {roi}")
        return True

    def clear_jobcard_roi(self, camera_id: str) -> bool:
        """Remove the job card ROI (stops job card reading). Returns False if not found."""
        if camera_id not in self._cameras:
            return False
        self._jobcard_rois.pop(camera_id, None)
        self._jobcard_state.pop(camera_id, None)
        logger.info(f"Job card ROI cleared for camera '{camera_id}'.")
        return True

    def stop_ocr(self, camera_id: str) -> bool:
        """
        Stop background OCR polling for a camera without unregistering it.
        The camera capture thread keeps running so the live stream stays alive.
        Returns False if camera not found.
        """
        if camera_id not in self._cameras:
            return False
        self._stop_poller(camera_id)
        # Clear the cached reading so stale data doesn't persist in the UI
        self._latest_valid.pop(camera_id, None)
        logger.info(f"OCR polling stopped for camera '{camera_id}' (stream still active).")
        return True

    def unregister_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera (including its poller). Returns False if not found."""
        if camera_id not in self._cameras:
            return False
        self._stop_poller(camera_id)
        self._cameras[camera_id].stop()
        del self._cameras[camera_id]
        self._rois.pop(camera_id, None)
        self._rate_history.pop(camera_id, None)
        self._latest_valid.pop(camera_id, None)
        self._last_accepted.pop(camera_id, None)
        self._jobcard_rois.pop(camera_id, None)
        self._jobcard_state.pop(camera_id, None)
        self._preprocess_modes.pop(camera_id, None)
        self._max_rates.pop(camera_id, None)
        self._reject_streaks.pop(camera_id, None)
        self._first_pending.pop(camera_id, None)
        self._machine_ids.pop(camera_id, None)
        logger.info(f"Camera '{camera_id}' unregistered.")
        return True

    def get_machine_id(self, camera_id: str) -> str:
        """Human machine name for a camera (falls back to the camera id)."""
        return self._machine_ids.get(camera_id, camera_id)

    def set_roi(self, camera_id: str, roi: ROIConfig) -> bool:
        """Update ROI for an existing camera. Returns False if not found."""
        if camera_id not in self._cameras:
            return False
        self._rois[camera_id] = (roi.x, roi.y, roi.width, roi.height)
        # Clear cached reading AND validation baseline — the new ROI may point at a
        # completely different part of the screen, so the old accepted value is meaningless.
        self._latest_valid.pop(camera_id, None)
        self._last_accepted.pop(camera_id, None)
        self._rate_history[camera_id] = deque(maxlen=RATE_HISTORY_SIZE)
        self._reject_streaks.pop(camera_id, None)
        self._first_pending.pop(camera_id, None)
        logger.info(f"ROI updated for camera '{camera_id}': {roi}")
        return True

    def get_camera_status(self, camera_id: str) -> Optional[dict]:
        """Return status dict for a camera, or None if not found."""
        cam = self._cameras.get(camera_id)
        if not cam:
            return None
        status = cam.get_status()
        roi = self._rois.get(camera_id)
        status["roi"] = (
            {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
            if roi else None
        )
        return status

    def list_cameras(self) -> list:
        """Return status for all registered cameras."""
        return [self.get_camera_status(cid) for cid in self._cameras]

    # ------------------------------------------------------------------
    # Background poller
    # ------------------------------------------------------------------

    def _start_poller(self, camera_id: str) -> None:
        """Start the background OCR polling thread for a camera."""
        # Stop any existing poller for this camera before starting a new one.
        # Prevents two threads running simultaneously after rapid re-registration.
        if camera_id in self._pollers:
            self._stop_poller(camera_id)
        self._poll_running[camera_id] = True
        t = threading.Thread(
            target=self._poll_loop,
            args=(camera_id,),
            daemon=True,
            name=f"ocr-poller-{camera_id}",
        )
        self._pollers[camera_id] = t
        t.start()
        logger.info(f"Background OCR poller started for '{camera_id}'")

    def _stop_poller(self, camera_id: str) -> None:
        """Stop the background polling thread for a camera."""
        self._poll_running[camera_id] = False
        t = self._pollers.pop(camera_id, None)
        if t and t.is_alive():
            t.join(timeout=3.0)

    def _poll_loop(self, camera_id: str) -> None:
        """
        Continuously reads OCR from the camera and caches the latest valid result.

        Why this works for a fast-changing counter:
            - OCR runs every POLL_INTERVAL seconds regardless of API activity
            - Transition frames (digit mid-roll) are filtered by confidence gate
            - Rate validation rejects physically impossible jumps
            - The cache always holds the last *valid* reading — even if the last
              few frames were transition frames, the cache stays reliable
        """
        drain_remaining = 0  # extra reads on frozen last frame after video ends
        cycle = 0            # poll cycle counter — job card is read every Nth cycle

        # Pick poll interval once — video files need fast polling to catch every
        # counter change; live cameras can poll slowly to keep CPU free for the stream.
        cam0 = self._cameras.get(camera_id)
        is_video = cam0._is_video_file if cam0 else False
        poll_interval = POLL_INTERVAL_VIDEO if is_video else POLL_INTERVAL_LIVE

        while self._poll_running.get(camera_id, False):
            try:
                cam = self._cameras.get(camera_id)
                if not cam:
                    break

                roi = self._rois.get(camera_id)
                mode = self._preprocess_modes.get(camera_id, "otsu")
                is_video_test = cam._is_video_file and not cam._loop

                if is_video:
                    # ── Video mode: single frame read ─────────────────────────────
                    # Video files use fast polling (0.2s) to catch every counter change.
                    # Single frame is fine — the clip is pre-recorded so no live transition noise.
                    frame = cam.get_frame()
                    if frame is None:
                        if cam.is_done():
                            logger.info(f"[{camera_id}] Video completed — OCR poller stopping.")
                            break
                        time.sleep(poll_interval)
                        continue
                    # Video frames have compression artifacts → lower sharpness threshold
                    result = self._counter_reader.read_counter(
                        frame, roi=roi, sharpness_threshold=20.0, preprocess_mode=mode
                    )
                    min_conf = 0.40

                else:
                    # ── Live camera mode: burst read + consensus voting ────────────
                    # Collect LIVE_BURST_FRAMES frames spaced LIVE_BURST_GAP apart.
                    # A digit mid-roll corrupts at most 1 frame; majority vote on the
                    # other 2 correct frames eliminates the misread automatically.
                    frames = []
                    for i in range(LIVE_BURST_FRAMES):
                        f = cam.get_frame()
                        if f is not None:
                            frames.append(f)
                        if i < LIVE_BURST_FRAMES - 1:
                            time.sleep(LIVE_BURST_GAP)

                    if not frames:
                        time.sleep(poll_interval)
                        continue

                    # Sharpness gate is PER SOURCE TYPE:
                    #   - local webcams (integer index): 35 — their frames are
                    #     honestly soft; 80 rejected everything.
                    #   - phone IP cams / RTSP / screen: 80 — frames are sharp,
                    #     so anything below 80 is a transition/refocus frame
                    #     whose partial reads ("1359" → "39") must be dropped.
                    is_local_webcam = isinstance(cam.source, int) or (
                        isinstance(cam.source, str) and str(cam.source).isdigit()
                    )
                    sharp_thr = 35.0 if is_local_webcam else 80.0
                    result = self._counter_reader.read_counter_consensus(
                        frames, roi=roi, min_confidence=LIVE_MIN_CONFIDENCE,
                        sharpness_threshold=sharp_thr, preprocess_mode=mode,
                    )
                    min_conf = LIVE_MIN_CONFIDENCE

                # ── Job card read ──────────────────────────────────────────────
                # Normally every Nth cycle (card is static). When a change is
                # in progress (pending value or misses counting), read EVERY
                # cycle so swaps and removals resolve in seconds.
                cycle += 1
                if camera_id in self._jobcard_rois:
                    jc_state = self._jobcard_state.get(camera_id)
                    change_in_progress = bool(jc_state) and (
                        jc_state.get("pending") is not None
                        or (jc_state.get("present") and jc_state.get("misses", 0) > 0)
                    )
                    if change_in_progress or cycle % JOBCARD_POLL_EVERY == 1:
                        jc_frame = cam.get_frame()
                        if jc_frame is not None:
                            self._update_jobcard_state(camera_id, jc_frame)

                if result.success and result.value is not None:
                    result, validation = self._run_validation(
                        camera_id, result, min_confidence=min_conf,
                        skip_rate_check=is_video_test,
                    )
                    if validation.is_valid:
                        rate = self._estimate_rate(camera_id)
                        # Looping video files only: don't cache near-zero values —
                        # garbled frames during loop transitions read as 1/5/25.
                        # Live cameras MUST cache small values: a real counter that
                        # was just reset counts 0, 1, 2... and the UI needs to show it.
                        min_cacheable = 100 if (is_video and cam._loop) else 0
                        if result.value is not None and result.value >= min_cacheable:
                            self._latest_valid[camera_id] = {
                                "result": result,
                                "validation": validation,
                                "polled_at": datetime.now(),
                                "rate_per_second": rate,
                            }
                            logger.info(
                                f"[{camera_id}] ✓ ACCEPTED: value={result.value} "
                                f"conf={result.confidence:.2f} → cached for UI"
                            )

            except Exception as e:
                logger.error(f"Poller error for '{camera_id}': {e}")

            # After processing a frame, check if the one-shot video has finished.
            # When it first ends: do VIDEO_DRAIN_READS more reads on the frozen last
            # frame so the frontend receives multiple readings even from a short clip.
            cam = self._cameras.get(camera_id)
            if cam and cam.is_done():
                if drain_remaining == 0:
                    drain_remaining = VIDEO_DRAIN_READS
                    logger.info(
                        f"[{camera_id}] Video ended — draining last frame "
                        f"({drain_remaining} more reads)."
                    )
                else:
                    drain_remaining -= 1
                    if drain_remaining == 0:
                        logger.info(f"[{camera_id}] Drain complete — OCR poller stopping.")
                        break
                time.sleep(0.2)  # short gap between drain reads
                continue

            time.sleep(poll_interval)

    def _update_jobcard_state(self, camera_id: str, frame) -> None:
        """
        Read the job card slot and update presence state.

        Card lifecycle (the job session signal from the product vision):
          - First successful read       → card PRESENT, job starts ("since" timestamp)
          - Value changes               → new job ("since" resets)
          - JOBCARD_MISS_LIMIT misses   → card REMOVED, job finished
            (a single miss is tolerated — glare/hand briefly covering the card)
        """
        roi = self._jobcard_rois.get(camera_id)
        if roi is None:
            return

        jc: JobCardResult = self._reader.read_jobcard(frame, roi=roi)
        now = datetime.now()
        state = self._jobcard_state.setdefault(camera_id, {
            "value": None, "confidence": 0.0, "source": "none",
            "present": False, "since": None, "last_seen": None, "misses": 0,
            "pending": None,
        })

        _core = jc_util.core  # digit-only identity (prefix letters flicker)

        if jc.success and jc.value:
            if state["present"] and _core(state["value"]) == _core(jc.value):
                # Same card still in place — refresh, keeping the CLEANEST form.
                # "JC-4521" (clean) beats a misread like "JBCA4521" even though
                # the misread is longer.
                if jc_util.score(jc.value) > jc_util.score(state["value"]):
                    state["value"] = jc.value
                state.update({
                    "confidence": jc.confidence, "source": jc.source,
                    "last_seen": now, "misses": 0, "pending": None,
                })
            elif jc.source == "qr" or _core(state.get("pending")) == _core(jc.value):
                # New value CONFIRMED: either an exact QR decode (deterministic,
                # no confirmation needed) or the same digit-core twice in a row.
                # Single-frame flickers ("JQ8PB1088") can never displace the
                # stable value — they don't repeat.
                is_change = state["value"] is not None
                logger.info(
                    f"[{camera_id}] JOB CARD {'changed' if is_change else 'placed'}: "
                    f"{state['value']} → {jc.value} (source={jc.source}, conf={jc.confidence:.2f})"
                )
                state.update({
                    "value": jc.value,
                    "confidence": jc.confidence,
                    "source": jc.source,
                    "present": True,
                    "since": now,
                    "last_seen": now,
                    "misses": 0,
                    "pending": None,
                })
                # The job card is the RESET AUTHORITY. A new job means the
                # per-job counter restarts at 0 — clear the counter baseline so
                # that reset is accepted immediately instead of being fought as
                # a backwards jump. (A reset WITHOUT a card change stays
                # suspicious — that path is unchanged.)
                self._reset_counter_baseline(camera_id)
            else:
                # First sighting of a different value — hold as pending until
                # the next read confirms it. A card IS visible, so no miss.
                state["pending"] = jc.value
                state["misses"] = 0
        else:
            state["pending"] = None
            state["misses"] += 1
            if state["present"] and state["misses"] >= JOBCARD_MISS_LIMIT:
                logger.info(
                    f"[{camera_id}] JOB CARD removed: {state['value']} "
                    f"(after {state['misses']} consecutive misses) — job finished."
                )
                state["present"] = False
                # Job ended — next job's counter starts fresh at 0.
                self._reset_counter_baseline(camera_id)

    def _reset_counter_baseline(self, camera_id: str) -> None:
        """
        Clear the counter validation baseline for a camera. The next reading is
        treated as a fresh first reading, so a per-job reset to 0 is accepted
        cleanly. Does NOT clear the cached UI value (_latest_valid) — the UI
        keeps showing the last number until the new job's first value confirms.
        """
        self._last_accepted.pop(camera_id, None)
        self._first_pending.pop(camera_id, None)
        self._reject_streaks.pop(camera_id, None)
        self._rate_history[camera_id] = deque(maxlen=RATE_HISTORY_SIZE)
        logger.info(f"[{camera_id}] Counter baseline reset (job boundary).")

    def get_jobcard_state(self, camera_id: str) -> Optional[dict]:
        """
        Return current job card state for a camera, or None if no job card ROI is set.
        Keys: value, confidence, source, present, since, last_seen.
        """
        if camera_id not in self._jobcard_rois:
            return None
        return self._jobcard_state.get(camera_id) or {
            "value": None, "confidence": 0.0, "source": "none",
            "present": False, "since": None, "last_seen": None, "misses": 0,
        }

    def get_latest_reading(self, camera_id: str) -> Optional[dict]:
        """
        Return the latest valid cached reading from the background poller.
        Returns None if camera not found or no valid reading has been captured yet.

        This is the recommended endpoint for live monitoring dashboards.
        It returns instantly from cache — no OCR wait on each API call.
        """
        if camera_id not in self._cameras:
            return None
        return self._latest_valid.get(camera_id)

    # ------------------------------------------------------------------
    # Rate estimation
    # ------------------------------------------------------------------

    def _estimate_rate(self, camera_id: str) -> Optional[float]:
        """
        Estimate current production rate (counts/second) from rolling history.

        Uses oldest and newest readings in the window:
            rate = (newest_value - oldest_value) / elapsed_seconds

        Returns None if fewer than 2 valid readings are available yet.
        """
        history = self._rate_history.get(camera_id)
        if not history or len(history) < 2:
            return None
        oldest = history[0]
        newest = history[-1]
        elapsed = (newest["timestamp"] - oldest["timestamp"]).total_seconds()
        # Need a meaningful window — right after a history reset (ROI change,
        # counter reset) two readings land < 1s apart and a tiny delta over a
        # near-zero window produces absurd rates (e.g. 400/sec).
        if elapsed < 1.0:
            return None
        delta = newest["value"] - oldest["value"]
        if delta < 0:
            return None  # counter was reset — rate not meaningful across a reset
        return round(delta / elapsed, 2)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(
        self,
        camera_id: str,
        result: OCRResult,
        min_confidence: float = 0.85,
        skip_rate_check: bool = False,
    ) -> Tuple[OCRResult, ValidationResult]:
        """
        Validate an OCR result against confidence threshold and rate limits.
        If valid, updates last-accepted reading and rate history for this camera.

        skip_rate_check: when True, bypasses the rate/direction check (used for
        one-shot video files where the recording may jump between non-sequential
        machine states).
        """
        now = datetime.now()
        last = self._last_accepted.get(camera_id)

        # ── First-reading confirmation ─────────────────────────────────
        # With no baseline, validate_reading accepts ANYTHING — so a partial
        # read at startup ("1") would poison the baseline. Require the first
        # value to repeat (within noise + machine-rate growth) before trusting it.
        if last is None and not skip_rate_check:
            pend = self._first_pending.get(camera_id)
            max_rate = self._max_rates.get(camera_id, self._default_max_rate)
            matched = False
            if pend:
                elapsed_s = (now - pend["last"]).total_seconds()
                forward_allow = 20 + max_rate * elapsed_s * 1.5
                matched = (pend["value"] - 20) <= result.value <= (pend["value"] + forward_allow)
            if matched:
                pend["count"] += 1
                pend["value"] = result.value
                pend["last"] = now
            else:
                pend = {"value": result.value, "count": 1, "last": now}
                self._first_pending[camera_id] = pend

            if pend["count"] < FIRST_READING_CONFIRM:
                validation = ValidationResult(
                    is_valid=False, confidence_ok=True, rate_ok=True,
                    direction_ok=True, digit_count_ok=True,
                    reason=(
                        f"First reading {result.value} awaiting confirmation "
                        f"({pend['count']}/{FIRST_READING_CONFIRM})."
                    ),
                )
                result = OCRResult(
                    success=False, value=result.value, raw_text=result.raw_text,
                    confidence=result.confidence, timestamp=result.timestamp,
                    error=f"VALIDATION: {validation.reason}",
                )
                return result, validation
            # Confirmed — clear pending and fall through to normal validation
            self._first_pending.pop(camera_id, None)

        validation = validate_reading(
            new_value=result.value,
            new_confidence=result.confidence,
            # Pass prev=None when skipping rate check — counter_validator treats
            # any first reading as valid regardless of value.
            prev_value=None if skip_rate_check else (last["value"] if last else None),
            prev_timestamp=None if skip_rate_check else (last["timestamp"] if last else None),
            current_timestamp=now,
            min_confidence=min_confidence,
            max_rate_per_second=self._max_rates.get(camera_id, self._default_max_rate),
        )

        # ── Rejection-streak recovery ──────────────────────────────────
        # A single bad baseline (full-frame read before ROI, partial digit) can
        # make validation reject every CORRECT reading that follows. If OCR keeps
        # returning the same value (±20) with good confidence and validation keeps
        # rejecting it, the display genuinely changed — adopt it as the new baseline.
        if not validation.is_valid and validation.confidence_ok and result.value is not None:
            streak = self._reject_streaks.get(camera_id)
            matched = False
            if streak:
                # A real reset counter is COUNTING UP while being rejected, so
                # allow forward growth at the machine's max rate between streak
                # readings (e.g. labels counter: reset → 5, 47, 89... is one
                # consistent streak). Backwards beyond noise breaks the streak.
                elapsed_s = (now - streak["last"]).total_seconds()
                max_rate = self._max_rates.get(camera_id, self._default_max_rate)
                forward_allow = 20 + max_rate * elapsed_s * 1.5
                matched = (streak["value"] - 20) <= result.value <= (streak["value"] + forward_allow)
            if matched:
                streak["count"] += 1
                streak["value"] = result.value
                streak["last"] = now
            else:
                streak = {"value": result.value, "count": 1, "last": now}
                self._reject_streaks[camera_id] = streak

            if streak["count"] >= REJECT_STREAK_ADOPT:
                logger.warning(
                    f"[{camera_id}] {streak['count']} consecutive consistent rejections "
                    f"around {result.value} — display changed. Adopting as new baseline."
                )
                self._reject_streaks.pop(camera_id, None)
                self._rate_history[camera_id] = deque(maxlen=RATE_HISTORY_SIZE)
                validation = ValidationResult(
                    is_valid=True, confidence_ok=True, rate_ok=True,
                    direction_ok=True, digit_count_ok=True,
                    reason=(
                        f"Adopted {result.value} as new baseline after "
                        f"{streak['count']} consecutive consistent readings."
                    ),
                )
        elif validation.is_valid:
            self._reject_streaks.pop(camera_id, None)

        if validation.is_valid:
            # On a video loop or counter reset (large backwards jump), clear rate history
            # so stale history doesn't poison the rate estimate after the reset.
            last = self._last_accepted.get(camera_id)
            if last and result.value < last["value"] - 50:
                self._rate_history[camera_id] = deque(maxlen=RATE_HISTORY_SIZE)
                logger.info(f"[{camera_id}] Counter reset/loop detected ({last['value']} → {result.value}). Rate history cleared.")
            self._last_accepted[camera_id] = {"value": result.value, "timestamp": now}
            history = self._rate_history.get(camera_id)
            if history is not None:
                history.append({"value": result.value, "timestamp": now})
        else:
            logger.warning(
                f"[{camera_id}] Reading REJECTED: value={result.value} | {validation.reason}"
            )
            result = OCRResult(
                success=False,
                value=result.value,
                raw_text=result.raw_text,
                confidence=result.confidence,
                timestamp=result.timestamp,
                error=f"VALIDATION FAILED: {validation.reason}",
            )

        return result, validation

    # ------------------------------------------------------------------
    # OCR operations (on-demand)
    # ------------------------------------------------------------------

    def read_from_camera(
        self,
        camera_id: str,
        validate: bool = True,
        min_confidence: float = 0.85,
    ) -> Tuple[OCRResult, Optional[ValidationResult]]:
        """
        Grab the latest frame from a registered camera and run OCR on demand.

        Prefer get_latest_reading() for live monitoring — it returns the background
        poller's cached result instantly with no OCR wait.
        Use this when you need a guaranteed fresh reading right now.
        """
        cam = self._cameras.get(camera_id)
        if not cam:
            return OCRResult(
                success=False, value=None, raw_text="",
                confidence=0.0, error=f"Camera '{camera_id}' is not registered",
            ), None

        frame = cam.get_frame()
        if frame is None:
            return OCRResult(
                success=False, value=None, raw_text="",
                confidence=0.0, error=f"Camera '{camera_id}' has no frame yet",
            ), None

        roi = self._rois.get(camera_id)
        mode = self._preprocess_modes.get(camera_id, "otsu")
        result = self._counter_reader.read_counter(frame, roi=roi, preprocess_mode=mode)

        if validate and result.success and result.value is not None:
            result, validation = self._run_validation(camera_id, result, min_confidence)
            return result, validation

        return result, None

    def read_consensus_from_camera(
        self,
        camera_id: str,
        num_frames: int = 5,
        interval_ms: int = 200,
        min_confidence: float = 0.85,
        validate: bool = True,
    ) -> Tuple[OCRResult, Optional[ValidationResult]]:
        """
        Capture N frames over a short window and return the majority-vote result.

        Best for high-accuracy one-shot reads (e.g. at end of job, operator check).
        For continuous live monitoring use GET /ocr/latest/{camera_id} instead.
        """
        cam = self._cameras.get(camera_id)
        if not cam:
            return OCRResult(
                success=False, value=None, raw_text="",
                confidence=0.0, error=f"Camera '{camera_id}' is not registered",
            ), None

        frames = []
        for _ in range(num_frames):
            frame = cam.get_frame()
            if frame is not None:
                frames.append(frame)
            time.sleep(interval_ms / 1000.0)

        if not frames:
            return OCRResult(
                success=False, value=None, raw_text="",
                confidence=0.0, error=f"Camera '{camera_id}' has no frames yet",
            ), None

        roi = self._rois.get(camera_id)
        mode = self._preprocess_modes.get(camera_id, "otsu")
        result = self._counter_reader.read_counter_consensus(
            frames, roi=roi, min_confidence=min_confidence, preprocess_mode=mode
        )

        if validate and result.success and result.value is not None:
            result, validation = self._run_validation(camera_id, result, min_confidence)
            return result, validation

        return result, None

    def read_from_image(
        self,
        image_bytes: bytes,
        roi: Optional[ROIConfig] = None,
    ) -> OCRResult:
        """Run OCR on an uploaded image file (for testing without a live camera)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return OCRResult(
                success=False, value=None, raw_text="",
                confidence=0.0, error="Could not decode uploaded image"
            )

        roi_tuple = (roi.x, roi.y, roi.width, roi.height) if roi else None
        return self._counter_reader.read_counter(frame, roi=roi_tuple)

    def read_all_from_image(
        self,
        image_bytes: bytes,
        roi: Optional[ROIConfig] = None,
        min_confidence: float = 0.4,
        min_height: int = 35,
        max_digits: int = 8,
    ) -> List[DetectedNumber]:
        """Detect every number visible in an uploaded image."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return []
        roi_tuple = (roi.x, roi.y, roi.width, roi.height) if roi else None
        return self._reader.read_all_counters(
            frame, roi=roi_tuple, min_confidence=min_confidence,
            min_height=min_height, max_digits=max_digits,
        )

    def read_all_from_camera(
        self,
        camera_id: str,
        min_confidence: float = 0.4,
    ) -> List[DetectedNumber]:
        """Detect every number from a live registered camera frame."""
        cam = self._cameras.get(camera_id)
        if not cam:
            return []
        frame = cam.get_frame()
        if frame is None:
            return []
        roi = self._rois.get(camera_id)
        return self._reader.read_all_counters(frame, roi=roi, min_confidence=min_confidence)

    def get_debug_image(self, camera_id: str) -> Optional[bytes]:
        """
        Return a JPEG debug image with ROI box and last OCR result drawn on the live frame.

        Uses the cached result from the background poller — does NOT run OCR here.
        Running OCR on every stream frame (~10/sec) would saturate the CPU and make
        the stream feel laggy. The poller already keeps the result fresh every 1-2s.
        """
        cam = self._cameras.get(camera_id)
        if not cam:
            return None
        frame = cam.get_frame()
        if frame is None:
            return None
        roi = self._rois.get(camera_id)
        cached = self._latest_valid.get(camera_id)
        cached_result = cached["result"] if cached else None
        jc_roi = self._jobcard_rois.get(camera_id)
        jc_state = self._jobcard_state.get(camera_id)
        jc_value = jc_state["value"] if (jc_state and jc_state["present"]) else None
        debug_frame = self._reader.get_debug_frame(
            frame, roi=roi, result=cached_result,
            jobcard_roi=jc_roi, jobcard_value=jc_value,
        )
        _, buffer = cv2.imencode(".jpg", debug_frame)
        return buffer.tobytes()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self):
        """Stop all pollers and cameras — called on app shutdown."""
        logger.info("Shutting down OCRService...")
        for camera_id in list(self._cameras.keys()):
            self._stop_poller(camera_id)
            self._cameras[camera_id].stop()
        self._cameras.clear()
        logger.info("OCRService shut down.")
