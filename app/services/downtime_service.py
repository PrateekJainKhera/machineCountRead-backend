import threading
import time
import logging
from datetime import datetime
from itertools import count
from typing import Dict, List, Optional

from app.db.database import session_scope, is_enabled
from app.db.models import DowntimeEventRow

logger = logging.getLogger(__name__)

# Counter unchanged for this long → machine considered idle (downtime event opens).
# 60s default is demo-friendly; raise per machine to 300-600s in production so
# short legitimate pauses don't each spawn a reason form.
DEFAULT_IDLE_THRESHOLD_S = 60.0

# Watcher tick interval
TICK_S = 2.0

# If this much wall-clock passed between ticks, the loop was SUSPENDED (laptop
# sleep, process freeze, long stall) — the gap is not machine idle. On resume,
# baselines are rebased so the gap can't fabricate a downtime event.
SUSPEND_GAP_S = 30.0

# If the OCR poller hasn't refreshed a camera's reading for this long, the
# camera can't currently see the counter — skip the tick (but DON'T reset the
# idle clock, so a brief poll hiccup doesn't wipe accumulated idle time).
READING_STALE_S = 90.0

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
        # camera_id → {"peak", "last_change", "active_event_id"}
        self._track: Dict[str, dict] = {}
        self._thresholds: Dict[str, float] = {}
        self._last_tick = None  # for suspend/sleep detection
        self._lock = threading.Lock()

        self._load_history()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="downtime-watch")
        self._thread.start()
        logger.info("DowntimeService started (auto-watching all OCR cameras).")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_history(self):
        if not is_enabled():
            return
        try:
            with session_scope() as s:
                if s is None:
                    return
                rows = s.query(DowntimeEventRow).order_by(DowntimeEventRow.id.desc()).limit(200).all()
                for r in rows:
                    # Close orphaned 'active' events from a prior run (can't be
                    # tracked again) so they don't linger as ongoing.
                    if r.status == "active":
                        r.status = "resolved"
                        if r.ended_at is None:
                            r.ended_at = r.started_at
                            r.duration_s = 0.0
                    self._events.insert(0, {
                        "event_id": r.id, "db_id": r.id, "camera_id": r.camera_id,
                        "machine_id": r.machine_id, "started_at": r.started_at,
                        "ended_at": r.ended_at, "duration_s": r.duration_s,
                        "status": r.status,
                        "reason": r.reason, "note": r.note or "", "job_card": r.job_card,
                        "counter_value": r.counter_value, "time_source": r.time_source,
                    })
                if rows:
                    # continue id sequence past what's in the DB
                    self._ids = count(max(r.id for r in rows) + 1)
            logger.info(f"DowntimeService loaded {len(self._events)} events from DB.")
        except Exception as e:
            logger.warning(f"Downtime history load failed: {e}")

    def _persist_open(self, ev: dict):
        if not is_enabled():
            return
        try:
            with session_scope() as s:
                if s is None:
                    return
                row = DowntimeEventRow(
                    id=ev["event_id"], machine_id=ev["machine_id"], camera_id=ev["camera_id"],
                    started_at=ev["started_at"], status="active", reason=ev["reason"],
                    note=ev["note"], job_card=ev["job_card"], counter_value=ev["counter_value"],
                    time_source=ev["time_source"],
                )
                s.merge(row)
        except Exception as e:
            logger.warning(f"Downtime open persist failed: {e}")

    def _persist_update(self, ev: dict):
        if not is_enabled():
            return
        try:
            with session_scope() as s:
                if s is None:
                    return
                row = s.get(DowntimeEventRow, ev["event_id"])
                if row:
                    row.ended_at = ev["ended_at"]
                    row.duration_s = ev["duration_s"]
                    row.status = ev["status"]
                    row.reason = ev["reason"]
                    row.note = ev["note"]
                    row.started_at = ev["started_at"]
                    row.time_source = ev["time_source"]
        except Exception as e:
            logger.warning(f"Downtime update persist failed: {e}")

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
        # Detect a suspended loop (laptop sleep / process freeze / long stall):
        # if far more than one tick elapsed, the gap is NOT machine idle — we
        # simply weren't watching. Resume cleanly: reset every idle baseline so
        # the gap can't fabricate a downtime event.
        resumed = self._last_tick is not None and (now - self._last_tick).total_seconds() > SUSPEND_GAP_S
        self._last_tick = now
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

                # Reading is stale — the camera can't currently see the counter.
                # Skip this tick WITHOUT touching last_change, so a brief poll
                # hiccup doesn't reset accumulated idle time. (A truly static,
                # readable counter re-caches every cycle and never goes stale;
                # if OCR is stopped, the camera isn't in _latest_valid at all.)
                if polled_at and (now - polled_at).total_seconds() > READING_STALE_S:
                    continue

                value = result.value
                tr = self._track.get(camera_id)
                if tr is None:
                    self._track[camera_id] = {
                        "peak": value, "last_change": now, "active_event_id": None,
                    }
                    continue

                if resumed:
                    # Came back from a suspend — rebase the idle clock to NOW so
                    # the offline gap isn't counted as downtime.
                    tr["peak"] = value
                    tr["last_change"] = now
                    continue

                peak = tr["peak"]
                # A counter only truly moves FORWARD. "Movement" = a new peak.
                #   • new high (value > peak)        → real production
                #   • big drop (value < peak - 50)   → job reset to ~0 (new job)
                #   • otherwise (dip within noise, or recovery to peak) → NOT
                #     movement. This is the fix: a stuck counter that flickers
                #     1448→1440→1448 (8 misread as 0) no longer resets the idle
                #     clock, so downtime is detected.
                moved = value > peak or value < peak - 50

                if moved:
                    tr["peak"] = value
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
                            self._persist_update(ev)
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
                        self._persist_open(ev)
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
            self._persist_update(ev)
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
            self._persist_update(ev)
            return dict(ev)

    def resolve_active(self, camera_id: str) -> None:
        """Close any open downtime event for a camera — used when a machine is
        disabled/removed so it doesn't linger as an active event in the DB."""
        now = datetime.now()
        with self._lock:
            tr = self._track.get(camera_id)
            if tr and tr.get("active_event_id") is not None:
                ev = self._get_event(tr["active_event_id"])
                if ev and ev["status"] == "active":
                    ev["ended_at"] = now
                    ev["duration_s"] = round((now - ev["started_at"]).total_seconds(), 1)
                    ev["status"] = "resolved"
                    self._persist_update(ev)
                    logger.info(f"[{camera_id}] Active downtime resolved (machine disabled/removed).")
                tr["active_event_id"] = None

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
