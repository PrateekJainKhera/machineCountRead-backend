from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional

from app.models.job_model import JobSessionModel

router = APIRouter(prefix="/jobs", tags=["Jobs"])


def _svc(request: Request):
    return request.app.state.job_service


@router.post("/start/{camera_id}", response_model=JobSessionModel)
def start_job(camera_id: str, request: Request):
    """Operator 'Start Job' — open a session for the job card currently in the slot.

    Idempotent: returns the running job if one already auto-started. 400 if no
    readable job card is present (nothing to identify the job)."""
    sess = _svc(request).start_job(camera_id)
    if sess is None:
        raise HTTPException(
            status_code=400,
            detail="No job card detected — place a job card in the slot, then Start Job.",
        )
    return sess


@router.post("/end/{camera_id}")
def end_current(camera_id: str, request: Request):
    """Manually end the active job on a machine (operator override)."""
    ok = _svc(request).end_current(camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail="No active job on this camera.")
    return {"message": f"Active job on '{camera_id}' ended."}


@router.get("/current/{camera_id}", response_model=Optional[JobSessionModel])
def get_current(camera_id: str, request: Request):
    """The job running right now on this machine (null if none)."""
    return _svc(request).get_current(camera_id)


@router.get("/sessions", response_model=List[JobSessionModel])
def get_sessions(request: Request, camera_id: Optional[str] = None, limit: int = 100):
    """Production-per-job log, newest first."""
    return _svc(request).get_sessions(camera_id, limit)
