import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

from app.db.database import session_scope, is_enabled
from app.db.models import JobSessionRow
from app.utils import jobcard as jc_util

logger = logging.getLogger(__name__)

TICK_S = 2.0

# The job card must be GONE this long before the job is considered finished.
# Tolerates brief QR/text read dropouts (glare, blur, a hand passing) so one
# continuous job doesn't split into multiple sessions. A real job change to a
# DIFFERENT card closes the old job immediately (no grace needed).
JOB_END_GRACE_S = 25.0

# Loop-suspend detection (laptop sleep / freeze). On resume the gap is ignored
# so the job doesn't fragment or backdate.
SUSPEND_GAP_S = 30.0

_core = jc_util.core


class JobService:
    """
    Tracks production per JOB. The per-job counter starts at ~0 when the job
    card is placed and climbs to the job total; production = end - start.

    Driven by the job card (the job boundary authority):
        card placed              → open session (start_counter = first reading)
        card value changes       → close current session, open new one
        card removed             → close current session

    Sessions persist to the DB on open (start recorded) and update on close
    (end + production). Survives restarts; loads recent history on startup.
    """

    def __init__(self, ocr_service):
        self._ocr = ocr_service
        self._lock = threading.Lock()
        # camera_id → active session dict
        self._active: Dict[str, dict] = {}
        # completed + active sessions, newest first (in-memory mirror of DB)
        self._sessions: List[dict] = []
        self._next_mem_id = 1
        self._last_tick = None  # for suspend/sleep detection

        self._load_history()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="job-watch")
        self._thread.start()
        logger.info("JobService started (tracking production per job).")

    # ------------------------------------------------------------------
    # History load (restart survival)
    # ------------------------------------------------------------------

    def _load_history(self):
        if not is_enabled():
            return
        try:
            with session_scope() as s:
                if s is None:
                    return
                rows = s.query(JobSessionRow).order_by(JobSessionRow.id.desc()).limit(200).all()
                orphans = 0
                for r in rows:
                    # A session left "active" belongs to a previous run that
                    # didn't shut down cleanly. Close it now so it doesn't show
                    # as RUNNING forever (it can never be tracked again).
                    if r.status == "active":
                        r.status = "completed"
                        if r.ended_at is None:
                            r.ended_at = r.started_at
                        orphans += 1
                    self._sessions.append(self._row_to_dict(r))
                if rows:
                    self._next_mem_id = max(r.id for r in rows) + 1
            logger.info(
                f"JobService loaded {len(self._sessions)} job sessions from DB "
                f"({orphans} stale 'active' closed)."
            )
        except Exception as e:
            logger.warning(f"JobService history load failed: {e}")

    @staticmethod
    def _row_to_dict(r: JobSessionRow) -> dict:
        return {
            "id": r.id, "db_id": r.id, "machine_id": r.machine_id,
            "camera_id": r.camera_id, "job_card": r.job_card,
            "started_at": r.started_at, "ended_at": r.ended_at,
            "start_counter": r.start_counter, "end_counter": r.end_counter,
            "production": r.production, "status": r.status,
        }

    # ------------------------------------------------------------------
    # Watcher
    # ------------------------------------------------------------------

    def _loop(self):
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Job watcher error: {e}")
            time.sleep(TICK_S)

    def _tick(self):
        now = datetime.now()
        # Suspend detection: a big gap between ticks means the loop was frozen
        # (laptop sleep). Skip one tick so the gap can't expire the job-end grace
        # or otherwise fragment a running job — just clear absence and continue.
        resumed = self._last_tick is not None and (now - self._last_tick).total_seconds() > SUSPEND_GAP_S
        self._last_tick = now
        with self._lock:
            if resumed:
                for sess in self._active.values():
                    sess["absent_since"] = None
                return
            for camera_id in list(self._ocr._jobcard_rois.keys()):
                jc = self._ocr.get_jobcard_state(camera_id)
                if jc is None:
                    continue
                cached = self._ocr.get_latest_reading(camera_id)
                counter = cached["result"].value if cached and cached.get("result") else None

                active = self._active.get(camera_id)
                present = jc.get("present")
                value = jc.get("value")

                if present and value:
                    if active is None:
                        self._open(camera_id, value, now, counter)
                    elif _core(active["job_card"]) != _core(value):
                        # A genuinely DIFFERENT card → job changed: close + open.
                        self._close(camera_id, now)
                        self._open(camera_id, value, now, counter)
                    else:
                        # Same job — card present (or recovered from a dropout).
                        active["absent_since"] = None
                        if jc_util.score(value) > jc_util.score(active["job_card"]):
                            active["job_card"] = value  # keep cleanest form (JC-4521)
                        self._track_counter(active, counter)
                else:
                    # Card not visible. A brief dropout (QR glare/blur) must NOT
                    # end the job — only a SUSTAINED absence does. Otherwise one
                    # continuous job fragments into several sessions.
                    if active is not None:
                        if active.get("absent_since") is None:
                            active["absent_since"] = now
                        elif (now - active["absent_since"]).total_seconds() >= JOB_END_GRACE_S:
                            self._close(camera_id, now)

    def _track_counter(self, sess: dict, counter: Optional[int]):
        if counter is None:
            return
        # start = lowest value seen (captures the reset-to-0 point)
        if sess["start_counter"] is None or counter < sess["start_counter"]:
            sess["start_counter"] = counter
        # end = latest value seen
        sess["end_counter"] = counter
        sess["production"] = max(0, (sess["end_counter"] or 0) - (sess["start_counter"] or 0))

    def _open(self, camera_id: str, job_card: str, started_at: datetime, counter: Optional[int]):
        sess = {
            "id": self._next_mem_id,
            "db_id": None,
            "machine_id": self._ocr.get_machine_id(camera_id),
            "camera_id": camera_id,
            "job_card": job_card,
            "started_at": started_at,
            "ended_at": None,
            "start_counter": counter,
            "end_counter": counter,
            "production": 0,
            "status": "active",
            "absent_since": None,
        }
        self._next_mem_id += 1
        self._active[camera_id] = sess
        self._sessions.insert(0, sess)
        logger.info(f"[{sess['machine_id']}] JOB START: {job_card} (counter={counter})")
        self._persist_open(sess)

    def _close(self, camera_id: str, ended_at: datetime):
        sess = self._active.pop(camera_id, None)
        if sess is None:
            return
        sess["ended_at"] = ended_at
        sess["status"] = "completed"
        logger.info(
            f"[{sess['machine_id']}] JOB END: {sess['job_card']} — "
            f"production={sess['production']} "
            f"({sess['start_counter']}→{sess['end_counter']})"
        )
        self._persist_close(sess)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_open(self, sess: dict):
        if not is_enabled():
            return
        try:
            with session_scope() as s:
                if s is None:
                    return
                row = JobSessionRow(
                    machine_id=sess["machine_id"], camera_id=sess["camera_id"],
                    job_card=sess["job_card"], started_at=sess["started_at"],
                    start_counter=sess["start_counter"], status="active",
                )
                s.add(row)
                s.flush()
                sess["db_id"] = row.id
        except Exception as e:
            logger.warning(f"Job open persist failed: {e}")

    def _persist_close(self, sess: dict):
        if not is_enabled() or sess.get("db_id") is None:
            return
        try:
            with session_scope() as s:
                if s is None:
                    return
                row = s.get(JobSessionRow, sess["db_id"])
                if row:
                    row.ended_at = sess["ended_at"]
                    row.start_counter = sess["start_counter"]
                    row.end_counter = sess["end_counter"]
                    row.production = sess["production"]
                    row.status = "completed"
        except Exception as e:
            logger.warning(f"Job close persist failed: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current(self, camera_id: str) -> Optional[dict]:
        with self._lock:
            a = self._active.get(camera_id)
            return dict(a) if a else None

    def get_sessions(self, camera_id: Optional[str] = None, limit: int = 100) -> List[dict]:
        with self._lock:
            out = [
                dict(s) for s in self._sessions
                if camera_id is None or s["camera_id"] == camera_id
            ]
        return out[:limit]

    def end_current(self, camera_id: str) -> bool:
        """Manually close the active job for a camera (operator override)."""
        with self._lock:
            if camera_id in self._active:
                self._close(camera_id, datetime.now())
                return True
            return False

    def shutdown(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        # Close any open sessions so production isn't lost on shutdown
        now = datetime.now()
        with self._lock:
            for camera_id in list(self._active.keys()):
                self._close(camera_id, now)
        logger.info("JobService shut down.")
