import threading
import time
import logging
from datetime import datetime
from itertools import count
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Counter unchanged for this long → machine considered idle (downtime event opens).
# Production default 300s (5 min); set lower per camera for testing via the API/UI.
DEFAULT_IDLE_THRESHOLD_S = 300.0

# Watcher tick interval
TICK_S = 2.0

# If the OCR poller hasn't refreshed a camera's reading for this long, the
# CAMERA is considered offline — that is NOT machine downtime (we just can't
# see), so no event is opened and idle time doesn't accumulate.
READING_STALE_S = 60.0

# Operator-selectable downtime reasons (from the product transcript)
DOWNTIME_REASONS = [
    "makeready",
    "changeover",
    "tea_break",
    "lunch_break",
    "maintenance",
    "breakdown",
    "idle",
    "planned_stop",
    "other",
]


class DowntimeService:
    """
    Watches every OCR camera's counter stream. When a counter stops changing
    for the camera's idle threshold, a downtime event opens automatically
    (started_at = the moment the counter last changed — auto time entry).
    When the counter moves again, the event resolves automatically.

    The operator fills in the WHY via the reason form (REST API); start/end
    times can also be corrected manually (time_source flips to "manual").

    All data is in-memory (consistent with current project phase — DB later).
    """

    def __init__(self, ocr_service):
        self._ocr = ocr_service
        self._ids = count(1)
        self._events: List[dict] = []
        # camera_id → {"last_value", "last_change", "active_event_id"}
        self._track: Dict[str, dict] = {}
        self._thresholds: Dict[str, float] = {}
        self._lock = threading.Lock()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="downtime-watch")
        self._thread.start()
        logger.info("DowntimeService started (auto-watching all OCR cameras).")

    # ------------------------------------------------------------------
    # Watcher loop
    # ------------------------------------------------------------------

    def _loop(self):
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Downtime watcher error: {e}")
            time.sleep(TICK_S)

    def _tick(self):
        now = datetime.now()
        with self._lock:
            # Drop tracking for unregistered cameras (history events are kept)
            for cid in list(self._track.keys()):
                if cid not in self._ocr._cameras:
                    self._track.pop(cid, None)

            for camera_id, cached in list(self._ocr._latest_valid.items()):
                result = cached.get("result")
                polled_at = cached.get("polled_at")
                if result is None or result.value is None:
                    continue

                # Camera/OCR offline ≠ machine idle. Don't accumulate idle time
                # while we can't actually see the counter.
                if polled_at and (now - polled_at).total_seconds() > READING_STALE_S:
                    tr = self._track.get(camera_id)
                    if tr:
                        tr["last_change"] = now  # freeze the idle clock
                    continue

                value = result.value
                tr = self._track.get(camera_id)
                if tr is None:
                    self._track[camera_id] = {
                        "last_value": value, "last_change": now, "active_event_id": None,
                    }
                    continue

                if value != tr["last_value"]:
                    # ── Counter moved — machine is producing ──
                    tr["last_value"] = value
                    tr["last_change"] = now
                    if tr["active_event_id"] is not None:
                        ev = self._get_event(tr["active_event_id"])
                        if ev and ev["status"] == "active":
                            ev["ended_at"] = now
                            ev["duration_s"] = round((now - ev["started_at"]).total_seconds(), 1)
                            ev["status"] = "resolved"
                            logger.info(
                                f"[{camera_id}] DOWNTIME ENDED: event #{ev['event_id']} "
                                f"({ev['duration_s']:.0f}s, reason={ev['reason'] or 'PENDING'})"
                            )
                        tr["active_event_id"] = None
                else:
                    # ── Counter unchanged — accumulate idle ──
                    idle_for = (now - tr["last_change"]).total_seconds()
                    threshold = self._thresholds.get(camera_id, DEFAULT_IDLE_THRESHOLD_S)
                    if idle_for >= threshold and tr["active_event_id"] is None:
                        jc = self._ocr.get_jobcard_state(camera_id) or {}
                        ev = {
                            "event_id": next(self._ids),
                            "camera_id": camera_id,
                            "machine_id": self._ocr.get_machine_id(camera_id),
                            # AUTO time entry: idle started when the counter
                            # last changed, not when we noticed.
                            "started_at": tr["last_change"],
                            "ended_at": None,
                            "duration_s": None,
                            "status": "active",
                            "reason": None,
                            "note": "",
                            "job_card": jc.get("value") if jc.get("present") else None,
                            "counter_value": value,
                            "time_source": "auto",
                        }
                        self._events.append(ev)
                        tr["active_event_id"] = ev["event_id"]
                        logger.warning(
                            f"[{ev['machine_id']}] DOWNTIME STARTED: counter stuck at {value} "
                            f"since {tr['last_change']:%H:%M:%S} (threshold {threshold:.0f}s) "
                            f"— event #{ev['event_id']}, reason PENDING"
                        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_status(self, camera_id: str) -> dict:
        """Live idle status for a camera."""
        now = datetime.now()
        with self._lock:
            tr = self._track.get(camera_id)
            threshold = self._thresholds.get(camera_id, DEFAULT_IDLE_THRESHOLD_S)
            if tr is None:
                return {
                    "camera_id": camera_id, "idle": False, "idle_seconds": 0.0,
                    "threshold_s": threshold, "active_event": None,
                }
            idle_seconds = (now - tr["last_change"]).total_seconds()
            active = self._get_event(tr["active_event_id"]) if tr["active_event_id"] else None
            return {
                "camera_id": camera_id,
                "idle": active is not None,
                "idle_seconds": round(idle_seconds, 1),
                "threshold_s": threshold,
                "active_event": dict(active) if active else None,
            }

    def get_events(self, camera_id: Optional[str] = None) -> List[dict]:
        """Downtime log, newest first."""
        with self._lock:
            evs = [
                dict(e) for e in self._events
                if camera_id is None or e["camera_id"] == camera_id
            ]
        return sorted(evs, key=lambda e: e["event_id"], reverse=True)

    def set_reason(self, event_id: int, reason: str, note: str = "") -> Optional[dict]:
        """Operator fills the downtime form. Returns the updated event."""
        if reason not in DOWNTIME_REASONS:
            raise ValueError(f"Unknown reason '{reason}'. Allowed: {DOWNTIME_REASONS}")
        with self._lock:
            ev = self._get_event(event_id)
            if ev is None:
                return None
            ev["reason"] = reason
            ev["note"] = note
            logger.info(f"Downtime event #{event_id}: reason set to '{reason}' ({note or 'no note'})")
            return dict(ev)

    def update_times(
        self, event_id: int,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> Optional[dict]:
        """Manual time correction by the operator/admin."""
        with self._lock:
            ev = self._get_event(event_id)
            if ev is None:
                return None
            if started_at:
                ev["started_at"] = started_at
            if ended_at:
                ev["ended_at"] = ended_at
                if ev["status"] == "active":
                    ev["status"] = "resolved"
                    # the watcher's active pointer must be released
                    for tr in self._track.values():
                        if tr.get("active_event_id") == event_id:
                            tr["active_event_id"] = None
            if ev["ended_at"]:
                ev["duration_s"] = round((ev["ended_at"] - ev["started_at"]).total_seconds(), 1)
            ev["time_source"] = "manual"
            logger.info(f"Downtime event #{event_id}: times updated manually.")
            return dict(ev)

    def set_threshold(self, camera_id: str, seconds: float) -> None:
        with self._lock:
            self._thresholds[camera_id] = max(10.0, float(seconds))
        logger.info(f"[{camera_id}] Idle threshold set to {seconds:.0f}s")

    def _get_event(self, event_id) -> Optional[dict]:
        for e in self._events:
            if e["event_id"] == event_id:
                return e
        return None

    def shutdown(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("DowntimeService shut down.")
