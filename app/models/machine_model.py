from pydantic import BaseModel, Field
from typing import Optional

from app.models.ocr_model import ROIConfig


class MachineConfig(BaseModel):
    """Create a machine master record (camera + ROIs + settings)."""
    machine_id: str = Field(..., description="Unique machine name e.g. 'FLEXO-1' (also the camera id)")
    source: str = Field(..., description="RTSP URL, video file path, or webcam index as string")
    display_type: str = Field("lcd", description="'lcd' (Otsu) or 'led' (7-segment)")
    max_rate_per_second: Optional[float] = Field(None, gt=0, description="Max counter speed (units/sec)")
    idle_threshold_s: float = Field(300.0, ge=10, description="Counter frozen this long → downtime event")
    enabled: bool = Field(True, description="Off keeps the config but stops reading this machine")
    counter_roi: Optional[ROIConfig] = Field(None, description="Counter digits region")
    jobcard_roi: Optional[ROIConfig] = Field(None, description="Job card slot region")


class MachineUpdate(BaseModel):
    """Patch a machine master record — only provided fields change."""
    source: Optional[str] = None
    display_type: Optional[str] = None
    max_rate_per_second: Optional[float] = Field(None, gt=0)
    idle_threshold_s: Optional[float] = Field(None, ge=10)
    enabled: Optional[bool] = None
    counter_roi: Optional[ROIConfig] = None
    jobcard_roi: Optional[ROIConfig] = None


class MachineResponse(BaseModel):
    """A machine master record + its live engine status."""
    id: int
    machine_id: str
    source: str
    display_type: str
    max_rate_per_second: Optional[float] = None
    idle_threshold_s: float
    enabled: bool
    counter_roi: Optional[ROIConfig] = None
    jobcard_roi: Optional[ROIConfig] = None
    # Live status pulled from the OCR engine (false when disabled / not registered)
    connected: bool = False
    has_frame: bool = False
    ocr_running: bool = False
