from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class JobSessionModel(BaseModel):
    """One job run — production = end_counter - start_counter (start ≈ 0)."""
    id: int
    machine_id: str
    camera_id: str
    job_card: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    start_counter: Optional[int] = None
    end_counter: Optional[int] = None
    production: Optional[int] = Field(None, description="Sheets/labels produced this job")
    status: str = Field(..., description="'active' | 'completed'")
