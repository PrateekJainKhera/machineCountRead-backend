from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional

from app.models.downtime_model import (
    DowntimeEventModel,
    DowntimeStatusModel,
    DowntimeReasonRequest,
    DowntimeTimesRequest,
    IdleThresholdRequest,
)
from app.services.downtime_service import DOWNTIME_REASONS

router = APIRouter(prefix="/downtime", tags=["Downtime"])


def _svc(request: Request):
    return request.app.state.downtime_service


@router.get("/reasons")
def list_reasons():
    """Allowed downtime reasons for the operator form."""
    return {"reasons": DOWNTIME_REASONS}


@router.get("/status/{camera_id}", response_model=DowntimeStatusModel)
def get_status(camera_id: str, request: Request):
    """
    Live idle status. `idle=true` means an active downtime event is open —
    the frontend should show the reason form if `active_event.reason` is null.
    """
    return _svc(request).get_status(camera_id)


@router.get("/events", response_model=List[DowntimeEventModel])
def get_all_events(request: Request, camera_id: Optional[str] = None):
    """Downtime log (newest first). Optional camera filter."""
    return _svc(request).get_events(camera_id)


@router.put("/events/{event_id}/reason", response_model=DowntimeEventModel)
def set_reason(event_id: int, body: DowntimeReasonRequest, request: Request):
    """Operator submits the downtime form: WHY was the machine idle."""
    try:
        ev = _svc(request).set_reason(event_id, body.reason, body.note)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if ev is None:
        raise HTTPException(status_code=404, detail=f"Downtime event {event_id} not found.")
    return ev


@router.put("/events/{event_id}/times", response_model=DowntimeEventModel)
def update_times(event_id: int, body: DowntimeTimesRequest, request: Request):
    """Manual correction of start/end times (marks the event time_source=manual)."""
    ev = _svc(request).update_times(event_id, body.started_at, body.ended_at)
    if ev is None:
        raise HTTPException(status_code=404, detail=f"Downtime event {event_id} not found.")
    return ev


@router.put("/threshold/{camera_id}")
def set_threshold(camera_id: str, body: IdleThresholdRequest, request: Request):
    """Set how long the counter must be unchanged before a downtime event opens."""
    _svc(request).set_threshold(camera_id, body.seconds)
    return {"message": f"Idle threshold for '{camera_id}' set to {body.seconds:.0f}s."}
