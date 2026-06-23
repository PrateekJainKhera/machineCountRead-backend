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

# How often (in ticks) to flush a running job's live counter/production to the DB
# so an ACTIVE row reflects current figures instead of NULL until it closes.
# 5 ticks × 2s = every ~10s.
PROGRESS_PERSIST_TICKS = 5

# The job card must be GONE this long before the job is considered finished.
# Tolerates brief QR/text read dropouts (glare, blur, a hand passing) so one
# continuous job doesn't split into multiple sessions. A real job change to a
# DIFFERENT card closes the old job immediately (no grace needed).
JOB_END_GRACE_S = 25.0

# Loop-suspend detection (laptop sleep / freeze). On resume the gap is ignored
# so the job doesn't fragment or backdate.
SUSPEND_GAP_S = 30.0

# An "active" session in the DB is RESUMED on restart if it started within this
# many hours (a restart isn't a job boundary). Older ones are closed as stale.
RESUME_MAX_AGE_H = 18.0

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
        self._tick_count = 0    # for periodic progress persistence
        # camera_id → digit-core of a card that was just End-Job'd and is STILL
        # in the slot — don't auto-reopen it until the card is removed/changed.
        self._ended_card: Dict[str, str] = {}

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
            now = datetime.now()
            with session_scope() as s:
                if s is None:
                    return
                rows = s.query(JobSessionRow).order_by(JobSessionRow.id.desc()).limit(200).all()
                resumed = stale = 0
                for r in rows:
                    if r.status == "active":
                        age_h = (now - r.started_at).total_seconds() / 3600 if r.started_at else 999
                        if age_h <= RESUME_MAX_AGE_H:
                            # RESUME across restart — a restart is not a job
                            # boundary. Reattach so the SAME job continues (and
                            # can be closed normally) instead of fragmenting into
                            # a new row when the camera reconnects.
                            d = self._row_to_dict(r)
                            d["absent_since"] = None
                            self._sessions.append(d)
                            self._active[r.camera_id] = d
                            resumed += 1
                            continue
                        # Too old to resume (e.g. crashed days ago) → close stale.
                        r.status = "completed"
                        if r.ended_at is None:
                            r.ended_at = r.started_at
                        stale += 1
                    self._sessions.append(self._row_to_dict(r))
                if rows:
                    self._next_mem_id = max(r.id for r in rows) + 1
            logger.info(
                f"JobService loaded {len(self._sessions)} job sessions from DB "
                f"({resumed} resumed across restart, {stale} stale-closed)."
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
                        if self._ended_card.get(camera_id) == _core(value):
                            # This card was just End-Job'd and is still sitting in
                            # the slot — wait until it's removed (or a different
                            # card placed) before starting a fresh job.
                            continue
                        self._open(camera_id, value, now, counter)
                    elif _core(active["job_card"]) != _core(value):
                        # A genuinely DIFFERENT card → job changed: close + open.
                        self._close(camera_id, now)
                        self._open(camera_id, value, now, counter)
                    else:
                        # Same job — card present (or back after a pause/dropout).
                        active["absent_since"] = None
                        if jc_util.score(value) > jc_util.score(active["job_card"]):
                            active["job_card"] = value  # keep cleanest form (JC-4521)
                        self._track_counter(active, counter)
                else:
                    # Card not visible → the job STAYS ACTIVE (paused). A break,
                    # breakdown, or the card removed for a while does NOT end the
                    # job or create a new row — the counter just pauses/resumes.
                    # A job ends ONLY via "End Job" or a DIFFERENT card.
                    # The card is gone, so clear any End-Job hold.
                    self._ended_card.pop(camera_id, None)

            # Periodically flush running jobs so the DB row shows live
            # counter/production (and a late start_counter) instead of NULL
            # until the job finally closes.
            self._tick_count += 1
            if self._tick_count % PROGRESS_PERSIST_TICKS == 0:
                for sess in self._active.values():
                    self._persist_progress(sess)

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
        self._ended_card.pop(camera_id, None)
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

    def _persist_progress(self, sess: dict):
        """Save the running counter/production but keep the session ACTIVE —
        so a restart can resume it with the latest figures."""
        if not is_enabled() or sess.get("db_id") is None:
            return
        try:
            with session_scope() as s:
                if s is None:
                    return
                row = s.get(JobSessionRow, sess["db_id"])
                if row:
                    row.start_counter = sess["start_counter"]
                    row.end_counter = sess["end_counter"]
                    row.production = sess["production"]
        except Exception as e:
            logger.warning(f"Job progress persist failed: {e}")

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
        """Manually end the active job (the ONLY normal way a job ends, besides a
        different card being placed). Holds the just-ended card so it doesn't
        auto-reopen while still in the slot."""
        with self._lock:
            sess = self._active.get(camera_id)
            if sess is not None:
                self._ended_card[camera_id] = _core(sess["job_card"])
                self._close(camera_id, datetime.now())
                return True
            return False

    def start_job(self, camera_id: str) -> Optional[dict]:
        """Operator 'Start Job' — open a session for the card currently in the slot.

        Idempotent and coexists with the automatic card-driven start: if a job is
        already running it just returns it; otherwise it captures the current job
        card value + counter as the baseline and opens one. Returns None if no
        readable job card is present (nothing to identify the job with)."""
        with self._lock:
            active = self._active.get(camera_id)
            if active is not None:
                return dict(active)  # already running (auto-started or manual)
            jc = self._ocr.get_jobcard_state(camera_id) or {}
            value = jc.get("value") if jc.get("present") else None
            if not value:
                return None  # no card → can't identify the job
            cached = self._ocr.get_latest_reading(camera_id)
            counter = cached["result"].value if cached and cached.get("result") else None
            self._ended_card.pop(camera_id, None)  # clear any End-Job hold
            self._open(camera_id, value, datetime.now(), counter)
            return dict(self._active.get(camera_id))

    def close_active(self, camera_id: str) -> bool:
        """Finalise the active job when a machine is DISABLED or REMOVED.

        Unlike end_current, this does NOT set the End-Job hold — turning the
        machine off cleanly closes the job (recording its end_counter/production),
        and re-enabling it with a card present begins a fresh job. Prevents the
        stranded ACTIVE/NULL rows seen when a machine is switched off."""
        with self._lock:
            if camera_id in self._active:
                self._close(camera_id, datetime.now())
                self._ended_card.pop(camera_id, None)
                return True
            return False

    def shutdown(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        # Do NOT close active sessions — a restart is not a job boundary. Persist
        # their latest progress and leave them ACTIVE so they RESUME on restart
        # (one row per real job run, not one per restart).
        with self._lock:
            for sess in self._active.values():
                self._persist_progress(sess)
        logger.info("JobService shut down.")
