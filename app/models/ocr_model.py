from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------

class ROIConfig(BaseModel):
    """Region of Interest coordinates for a camera."""
    x: int = Field(..., ge=0, description="Left edge (pixels)")
    y: int = Field(..., ge=0, description="Top edge (pixels)")
    width: int = Field(..., gt=0, description="Width (pixels)")
    height: int = Field(..., gt=0, description="Height (pixels)")


class CameraConfig(BaseModel):
    """Configuration for registering a camera source."""
    camera_id: str = Field(..., description="Unique camera identifier e.g. 'machine_a_display'")
    source: str = Field(..., description="RTSP URL, video file path, or camera index as string")
    machine_id: Optional[str] = Field(
        None, description="Human machine identifier e.g. 'FLEXO-1' — shown on downtime events and reports"
    )
    roi: Optional[ROIConfig] = Field(None, description="Region of Interest for the counter")
    loop: bool = Field(True, description="Loop video files. Set False to stop OCR when video ends.")
    jobcard_roi: Optional[ROIConfig] = Field(
        None, description="Region of the job card slot below the counter (alphanumeric/QR)"
    )
    display_type: str = Field(
        "lcd",
        description="Counter display type: 'lcd' (Otsu) or 'led' (best-channel mode for LED/7-segment displays)",
    )
    max_rate_per_second: Optional[float] = Field(
        None,
        gt=0,
        description="Machine's max counter speed (units/sec) for jump validation. "
                    "Default 30/sec — flexo label presses run ~13-15 labels/sec at moderate speed.",
    )


# ------------------------------------------------------------------
# Response schemas
# ------------------------------------------------------------------

class ValidationInfo(BaseModel):
    """Result of confidence + rate validation on an OCR reading."""
    is_valid: bool
    confidence_ok: bool
    rate_ok: bool
    direction_ok: bool
    reason: str


class OCRResponse(BaseModel):
    """Response from a single-value OCR read operation."""
    success: bool
    camera_id: str
    value: Optional[int] = Field(None, description="Extracted machine counter value")
    raw_text: str = Field("", description="Raw OCR text before parsing")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    timestamp: datetime
    error: Optional[str] = None
    validation: Optional[ValidationInfo] = Field(
        None,
        description="Validation result — present when validate=true was requested"
    )


class BBoxModel(BaseModel):
    """Bounding box of a detected number in the image."""
    x: int
    y: int
    width: int
    height: int


class DetectedNumberResponse(BaseModel):
    """One number detected by read_all_counters()."""
    value: int
    raw_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BBoxModel


class MultiOCRResponse(BaseModel):
    """Response from a read-all operation — returns every number found in the image."""
    camera_id: str
    count: int = Field(..., description="Total numbers detected")
    numbers: List[DetectedNumberResponse]
    timestamp: datetime


class CameraStatusResponse(BaseModel):
    """Live status of a registered camera."""
    camera_id: str
    source: str
    connected: bool
    has_frame: bool
    roi: Optional[ROIConfig] = None


class RegisterCameraResponse(BaseModel):
    """Confirmation response when a camera is registered."""
    message: str
    camera_id: str


class JobCardInfo(BaseModel):
    """
    Current job card state for a camera (read from the slot below the counter).

    present=False with a non-null value means the card WAS there and has been
    removed — i.e. the job identified by `value` just finished.
    """
    value: Optional[str] = Field(None, description="Job card number e.g. 'JC-4521'")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: str = Field("none", description="'qr' | 'ocr' | 'none'")
    present: bool = Field(False, description="Card currently visible in the slot")
    since: Optional[datetime] = Field(None, description="When this card was first seen (job start)")
    last_seen: Optional[datetime] = Field(None, description="Last successful read of the card")


class LatestReadingResponse(BaseModel):
    """
    Response from GET /ocr/latest/{camera_id}.

    Returns the background poller's cached result — instant, no OCR wait.
    Use this for live dashboards polling every few seconds.

    rate_per_second: estimated production speed derived from the last
                     10 valid readings (None until enough history exists).
    polled_at:       when the background poller last captured this value.
    stale_seconds:   how many seconds ago the value was last updated.
    """
    camera_id: str
    value: Optional[int] = None
    confidence: float = 0.0
    raw_text: str = ""
    rate_per_second: Optional[float] = Field(
        None, description="Estimated production rate — counts per second"
    )
    polled_at: Optional[datetime] = None
    stale_seconds: Optional[float] = Field(
        None, description="Seconds since last valid reading was captured"
    )
    validation: Optional[ValidationInfo] = None
    job_card: Optional[JobCardInfo] = Field(
        None, description="Job card state — null when no job card ROI is configured"
    )
