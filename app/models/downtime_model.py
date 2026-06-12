from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DowntimeEventModel(BaseModel):
    """One machine-idle period detected from the counter stream."""
    event_id: int
    camera_id: str
    machine_id: str = Field(..., description="Which machine paused (set at camera registration)")
    started_at: datetime = Field(..., description="When the counter stopped changing (auto) or manual entry")
    ended_at: Optional[datetime] = Field(None, description="When the counter resumed (auto) or manual entry")
    duration_s: Optional[float] = None
    status: str = Field(..., description="'active' (machine still idle) | 'resolved'")
    reason: Optional[str] = Field(None, description="Operator-selected reason — null means form PENDING")
    note: str = ""
    job_card: Optional[str] = Field(None, description="Job card present when downtime started")
    counter_value: Optional[int] = Field(None, description="Counter value where it stalled")
    time_source: str = Field("auto", description="'auto' (detected) | 'manual' (operator corrected)")


class DowntimeStatusModel(BaseModel):
    """Live idle status of one camera/machine."""
    camera_id: str
    idle: bool = Field(..., description="True when an active downtime event is open")
    idle_seconds: float = Field(0.0, description="Seconds since the counter last changed")
    threshold_s: float = Field(..., description="Idle seconds required to open a downtime event")
    active_event: Optional[DowntimeEventModel] = None


class DowntimeReasonRequest(BaseModel):
    """Operator's downtime form submission."""
    reason: str = Field(..., description="makeready | changeover | tea_break | lunch_break | maintenance | breakdown | idle | planned_stop | other")
    note: str = Field("", description="Optional free-text detail")


class DowntimeTimesRequest(BaseModel):
    """Manual time correction for a downtime event."""
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


class IdleThresholdRequest(BaseModel):
    seconds: float = Field(..., ge=10, description="Counter-unchanged seconds before a downtime event opens (min 10)")
