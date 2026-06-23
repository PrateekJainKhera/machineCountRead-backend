from fastapi import APIRouter, HTTPException, Request
from typing import List

from app.models.machine_model import MachineConfig, MachineUpdate, MachineResponse

router = APIRouter(prefix="/machines", tags=["Machines"])


def _svc(request: Request):
    return request.app.state.machine_service


@router.get("", response_model=List[MachineResponse])
def list_machines(request: Request):
    """All machine master records with live engine status (connected / has_frame)."""
    return _svc(request).list_machines()


@router.get("/{machine_id}", response_model=MachineResponse)
def get_machine(machine_id: str, request: Request):
    svc = _svc(request)
    m = svc.get(machine_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found.")
    return svc._with_status(m)


@router.post("", response_model=MachineResponse)
def create_machine(cfg: MachineConfig, request: Request):
    """Create a machine (camera + ROIs + settings). Starts reading if enabled."""
    try:
        return _svc(request).create(cfg)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.put("/{machine_id}", response_model=MachineResponse)
def update_machine(machine_id: str, patch: MachineUpdate, request: Request):
    """Patch a machine and re-apply it to the engine."""
    try:
        m = _svc(request).update(machine_id, patch)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if m is None:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found.")
    return m


@router.delete("/{machine_id}")
def delete_machine(machine_id: str, request: Request):
    """Remove a machine completely (config + live camera)."""
    try:
        ok = _svc(request).delete(machine_id)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found.")
    return {"message": f"Machine '{machine_id}' removed."}
