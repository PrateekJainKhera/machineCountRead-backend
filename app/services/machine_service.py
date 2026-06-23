"""
Machine Master service — the persistent camera↔machine registry.

One MachineRow = one machine = one camera + its fixed counter/job-card ROIs +
settings. Set up once; on every startup the enabled machines are auto-registered
into the OCR engine and start reading. This is what lets 10-50 machines run
without manual re-setup after a restart.

`machine_id` is also used as the engine camera_id (one camera per machine).
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from app.db.database import session_scope, is_enabled
from app.db.models import MachineRow
from app.models.ocr_model import ROIConfig

logger = logging.getLogger(__name__)


class MachineService:
    def __init__(self, ocr_service, downtime_service, job_service=None):
        self._ocr = ocr_service
        self._downtime = downtime_service
        self._job = job_service

    # ------------------------------------------------------------------
    # Row <-> dict
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(r: MachineRow) -> dict:
        counter_roi = (
            {"x": r.roi_x, "y": r.roi_y, "width": r.roi_w, "height": r.roi_h}
            if None not in (r.roi_x, r.roi_y, r.roi_w, r.roi_h) else None
        )
        jobcard_roi = (
            {"x": r.jc_x, "y": r.jc_y, "width": r.jc_w, "height": r.jc_h}
            if None not in (r.jc_x, r.jc_y, r.jc_w, r.jc_h) else None
        )
        return {
            "id": r.id,
            "machine_id": r.machine_id,
            "source": r.source,
            "display_type": r.display_type,
            "max_rate_per_second": r.max_rate,
            "idle_threshold_s": r.idle_threshold_s,
            "enabled": bool(r.enabled),
            "counter_roi": counter_roi,
            "jobcard_roi": jobcard_roi,
        }

    def _with_status(self, m: dict) -> dict:
        """Attach live engine status (connected / has_frame)."""
        status = self._ocr.get_camera_status(m["machine_id"])
        m = dict(m)
        m["connected"] = bool(status and status.get("connected"))
        m["has_frame"] = bool(status and status.get("has_frame"))
        m["ocr_running"] = bool(status and status.get("ocr_running"))
        return m

    # ------------------------------------------------------------------
    # Engine apply / remove
    # ------------------------------------------------------------------

    def _apply(self, m: dict) -> None:
        """Register (or re-register) this machine's camera into the OCR engine."""
        roi = ROIConfig(**m["counter_roi"]) if m.get("counter_roi") else None
        jc = ROIConfig(**m["jobcard_roi"]) if m.get("jobcard_roi") else None
        self._ocr.register_camera(
            camera_id=m["machine_id"],
            source=m["source"],
            roi=roi,
            loop=True,
            jobcard_roi=jc,
            display_type=m["display_type"],
            max_rate_per_second=m.get("max_rate_per_second"),
            machine_id=m["machine_id"],
        )
        self._downtime.set_threshold(m["machine_id"], m["idle_threshold_s"])
        logger.info(f"Machine '{m['machine_id']}' applied to engine (enabled).")

    def _remove(self, machine_id: str) -> None:
        """Stop reading this machine in the engine (config stays in the DB).

        Finalises the active job and any open downtime event FIRST, so turning a
        machine off doesn't leave stranded ACTIVE/NULL rows in the DB."""
        if self._job is not None:
            self._job.close_active(machine_id)
        self._downtime.resolve_active(machine_id)
        self._ocr.unregister_camera(machine_id)
        logger.info(f"Machine '{machine_id}' removed from engine.")

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def register_all_enabled(self) -> int:
        """On boot: register every enabled machine so it starts reading. Returns count."""
        if not is_enabled():
            logger.warning("Persistence disabled — no machines to auto-register.")
            return 0
        n = 0
        for m in self.list_machines(with_status=False):
            if m["enabled"]:
                try:
                    self._apply(m)
                    n += 1
                except Exception as e:
                    logger.error(f"Auto-register failed for '{m['machine_id']}': {e}")
        logger.info(f"Machine Master: auto-registered {n} enabled machine(s) on startup.")
        return n

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_machines(self, with_status: bool = True) -> List[dict]:
        if not is_enabled():
            return []
        out: List[dict] = []
        with session_scope() as s:
            if s is None:
                return []
            for r in s.query(MachineRow).order_by(MachineRow.machine_id.asc()).all():
                out.append(self._row_to_dict(r))
        return [self._with_status(m) for m in out] if with_status else out

    def get(self, machine_id: str) -> Optional[dict]:
        if not is_enabled():
            return None
        with session_scope() as s:
            if s is None:
                return None
            r = s.query(MachineRow).filter(MachineRow.machine_id == machine_id).first()
            if r is None:
                return None
            return self._row_to_dict(r)

    def create(self, cfg) -> dict:
        """Create a machine, persist it, and (if enabled) start it reading."""
        if not is_enabled():
            raise RuntimeError("Persistence is disabled — cannot save machines.")
        now = datetime.now()
        with session_scope() as s:
            if s is None:
                raise RuntimeError("No database session.")
            exists = s.query(MachineRow).filter(MachineRow.machine_id == cfg.machine_id).first()
            if exists:
                raise ValueError(f"Machine '{cfg.machine_id}' already exists.")
            r = MachineRow(
                machine_id=cfg.machine_id,
                source=cfg.source,
                display_type=cfg.display_type,
                max_rate=cfg.max_rate_per_second,
                idle_threshold_s=cfg.idle_threshold_s,
                enabled=cfg.enabled,
                created_at=now,
                updated_at=now,
            )
            _set_rois(r, cfg.counter_roi, cfg.jobcard_roi)
            s.add(r)
            s.flush()
            m = self._row_to_dict(r)
        if m["enabled"]:
            self._apply(m)
        return self._with_status(m)

    def update(self, machine_id: str, patch) -> Optional[dict]:
        """Patch a machine and re-apply it to the engine."""
        if not is_enabled():
            raise RuntimeError("Persistence is disabled — cannot update machines.")
        with session_scope() as s:
            if s is None:
                return None
            r = s.query(MachineRow).filter(MachineRow.machine_id == machine_id).first()
            if r is None:
                return None
            if patch.source is not None:
                r.source = patch.source
            if patch.display_type is not None:
                r.display_type = patch.display_type
            if patch.max_rate_per_second is not None:
                r.max_rate = patch.max_rate_per_second
            if patch.idle_threshold_s is not None:
                r.idle_threshold_s = patch.idle_threshold_s
            if patch.enabled is not None:
                r.enabled = patch.enabled
            if patch.counter_roi is not None:
                _set_rois(r, patch.counter_roi, None)
            if patch.jobcard_roi is not None:
                _set_rois(r, None, patch.jobcard_roi)
            r.updated_at = datetime.now()
            s.flush()
            m = self._row_to_dict(r)
        # Re-apply: enabled → (re)register with new config; disabled → stop reading.
        if m["enabled"]:
            self._apply(m)
        else:
            self._remove(machine_id)
        return self._with_status(m)

    def delete(self, machine_id: str) -> bool:
        if not is_enabled():
            raise RuntimeError("Persistence is disabled — cannot delete machines.")
        with session_scope() as s:
            if s is None:
                return False
            r = s.query(MachineRow).filter(MachineRow.machine_id == machine_id).first()
            if r is None:
                return False
            s.delete(r)
        self._remove(machine_id)
        return True


def _set_rois(r: MachineRow, counter_roi, jobcard_roi) -> None:
    """Copy ROIConfig values onto the row (None leaves that ROI untouched)."""
    if counter_roi is not None:
        r.roi_x, r.roi_y, r.roi_w, r.roi_h = (
            counter_roi.x, counter_roi.y, counter_roi.width, counter_roi.height
        )
    if jobcard_roi is not None:
        r.jc_x, r.jc_y, r.jc_w, r.jc_h = (
            jobcard_roi.x, jobcard_roi.y, jobcard_roi.width, jobcard_roi.height
        )
