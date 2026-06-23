import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from typing import Optional
from datetime import datetime

from app.models.ocr_model import (
    CameraConfig,
    ROIConfig,
    OCRResponse,
    MultiOCRResponse,
    DetectedNumberResponse,
    BBoxModel,
    CameraStatusResponse,
    RegisterCameraResponse,
    ValidationInfo,
    LatestReadingResponse,
    JobCardInfo,
)

router = APIRouter(prefix="/ocr", tags=["OCR"])


def _get_service(request: Request):
    """Helper to pull OCRService from app state."""
    return request.app.state.ocr_service


# ------------------------------------------------------------------
# Camera management endpoints
# ------------------------------------------------------------------

@router.post("/cameras/register", response_model=RegisterCameraResponse)
def register_camera(config: CameraConfig, request: Request):
    """
    Register a camera source (RTSP / file / webcam index).
    The system starts capturing frames immediately.

    Example body:
        {
            "camera_id": "machine_a_display",
            "source": "rtsp://192.168.1.100:554/stream",
            "roi": { "x": 100, "y": 50, "width": 200, "height": 80 }
        }
    """
    service = _get_service(request)
    service.register_camera(
        camera_id=config.camera_id,
        source=config.source,
        roi=config.roi,
        loop=config.loop,
        jobcard_roi=config.jobcard_roi,
        display_type=config.display_type,
        max_rate_per_second=config.max_rate_per_second,
        machine_id=config.machine_id,
    )
    return RegisterCameraResponse(
        message=f"Camera '{config.camera_id}' registered successfully.",
        camera_id=config.camera_id,
    )


@router.delete("/cameras/{camera_id}")
def unregister_camera(camera_id: str, request: Request):
    """Stop and remove a registered camera."""
    service = _get_service(request)
    removed = service.unregister_camera(camera_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return {"message": f"Camera '{camera_id}' removed."}


@router.post("/cameras/{camera_id}/stop-ocr")
def stop_ocr(camera_id: str, request: Request):
    """Stop background OCR polling for a camera without unregistering it. Stream stays alive."""
    service = _get_service(request)
    stopped = service.stop_ocr(camera_id)
    if not stopped:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return {"message": f"OCR polling stopped for camera '{camera_id}'."}


@router.post("/cameras/{camera_id}/start-ocr")
def start_ocr(camera_id: str, request: Request):
    """(Re)start background OCR polling for an already-registered camera."""
    service = _get_service(request)
    started = service.start_ocr(camera_id)
    if not started:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return {"message": f"OCR polling started for camera '{camera_id}'."}


@router.put("/cameras/{camera_id}/roi", response_model=RegisterCameraResponse)
def update_roi(camera_id: str, roi: ROIConfig, request: Request):
    """Update the Region of Interest for a registered camera."""
    service = _get_service(request)
    updated = service.set_roi(camera_id, roi)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return RegisterCameraResponse(
        message=f"ROI updated for camera '{camera_id}'.",
        camera_id=camera_id,
    )


@router.put("/cameras/{camera_id}/jobcard-roi", response_model=RegisterCameraResponse)
def update_jobcard_roi(camera_id: str, roi: ROIConfig, request: Request):
    """
    Set/update the JOB CARD slot region — the fixed magnet spot below the counter
    where the operator places the printed job card (alphanumeric number + QR).

    Once set, the background poller reads the card alongside the counter and
    `GET /ocr/latest/{camera_id}` includes the `job_card` state.
    """
    service = _get_service(request)
    updated = service.set_jobcard_roi(camera_id, roi)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return RegisterCameraResponse(
        message=f"Job card ROI updated for camera '{camera_id}'.",
        camera_id=camera_id,
    )


@router.delete("/cameras/{camera_id}/jobcard-roi")
def clear_jobcard_roi(camera_id: str, request: Request):
    """Remove the job card ROI — stops job card reading for this camera."""
    service = _get_service(request)
    cleared = service.clear_jobcard_roi(camera_id)
    if not cleared:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return {"message": f"Job card ROI cleared for camera '{camera_id}'."}


@router.get("/cameras", response_model=list[CameraStatusResponse])
def list_cameras(request: Request):
    """List all registered cameras and their connection status."""
    service = _get_service(request)
    cameras = service.list_cameras()
    return [
        CameraStatusResponse(
            camera_id=c["camera_id"],
            source=c["source"],
            connected=c["connected"],
            has_frame=c["has_frame"],
            roi=ROIConfig(**c["roi"]) if c.get("roi") else None,
        )
        for c in cameras
    ]


@router.get("/cameras/{camera_id}/status", response_model=CameraStatusResponse)
def camera_status(camera_id: str, request: Request):
    """Get connection status for a specific camera."""
    service = _get_service(request)
    status = service.get_camera_status(camera_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return CameraStatusResponse(
        camera_id=status["camera_id"],
        source=status["source"],
        connected=status["connected"],
        has_frame=status["has_frame"],
        roi=ROIConfig(**status["roi"]) if status.get("roi") else None,
    )


# ------------------------------------------------------------------
# OCR read endpoints
# ------------------------------------------------------------------

@router.get("/latest/{camera_id}", response_model=LatestReadingResponse)
def get_latest_reading(camera_id: str, request: Request):
    """
    **Recommended for live dashboards.**

    Returns the background poller's cached result — responds instantly with no OCR wait.

    The poller runs OCR every 500ms in a background thread, filtering out:
      - Low-confidence readings (digit mid-transition / glare)
      - Physically impossible counter jumps (bad OCR misread)

    The cache always holds the last *validated* reading, so even if the last few
    frames caught a digit mid-roll, this endpoint still returns the correct value.

    Also returns `rate_per_second` — the estimated current production speed calculated
    from the last 10 valid readings. Useful for detecting machine idle vs running.

    `stale_seconds` tells you how long ago the last valid reading was captured.
    If this grows large (>10s), the display may be obscured or the camera disconnected.
    """
    service = _get_service(request)
    cached = service.get_latest_reading(camera_id)

    # Job card state — None when no job card ROI is configured for this camera
    jc = service.get_jobcard_state(camera_id)
    job_card = JobCardInfo(
        value=jc["value"],
        confidence=jc["confidence"],
        source=jc["source"],
        present=jc["present"],
        since=jc["since"],
        last_seen=jc["last_seen"],
    ) if jc else None

    if cached is None:
        if camera_id not in [c["camera_id"] for c in service.list_cameras()]:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
        # Camera exists but no valid reading yet (poller just started)
        return LatestReadingResponse(camera_id=camera_id, job_card=job_card)

    result = cached["result"]
    validation = cached["validation"]
    polled_at = cached["polled_at"]
    stale = (datetime.now() - polled_at).total_seconds() if polled_at else None

    return LatestReadingResponse(
        camera_id=camera_id,
        value=result.value,
        confidence=result.confidence,
        raw_text=result.raw_text,
        rate_per_second=cached.get("rate_per_second"),
        polled_at=polled_at,
        stale_seconds=round(stale, 1) if stale is not None else None,
        validation=ValidationInfo(**validation.to_dict()) if validation else None,
        job_card=job_card,
    )


@router.get("/read/{camera_id}", response_model=OCRResponse)
def read_from_camera(
    camera_id: str,
    request: Request,
    validate: bool = True,
    min_confidence: float = 0.85,
):
    """
    Read machine counter from a single live frame.
    Fast but may catch a digit mid-transition on fast machines.
    Use /read-consensus/{camera_id} for higher accuracy.

    Parameters:
        validate:       Run confidence + rate validation (default True).
        min_confidence: Confidence threshold used when validate=True (default 0.85).
    """
    service = _get_service(request)
    result, validation = service.read_from_camera(
        camera_id, validate=validate, min_confidence=min_confidence
    )
    return OCRResponse(
        success=result.success,
        camera_id=camera_id,
        value=result.value,
        raw_text=result.raw_text,
        confidence=result.confidence,
        timestamp=result.timestamp,
        error=result.error,
        validation=ValidationInfo(**validation.to_dict()) if validation else None,
    )


@router.get("/read-consensus/{camera_id}", response_model=OCRResponse)
def read_consensus_from_camera(
    camera_id: str,
    request: Request,
    num_frames: int = 5,
    interval_ms: int = 200,
    min_confidence: float = 0.85,
):
    """
    Read machine counter using multi-frame majority voting — recommended for fast machines.

    Captures N frames over a short window, runs OCR on each, and returns the
    value that appeared most consistently across frames.

    This eliminates transition-frame errors caused by digits changing mid-capture.

    Parameters:
        num_frames:     Frames to capture (default 5)
        interval_ms:    Gap between captures in ms (default 200ms → 1 sec total window)
        min_confidence: Per-frame confidence threshold (default 0.85)

    Example result raw_text: "1093 [4/5 frames agreed]"
    """
    service = _get_service(request)
    result, validation = service.read_consensus_from_camera(
        camera_id,
        num_frames=num_frames,
        interval_ms=interval_ms,
        min_confidence=min_confidence,
    )
    return OCRResponse(
        success=result.success,
        camera_id=camera_id,
        value=result.value,
        raw_text=result.raw_text,
        confidence=result.confidence,
        timestamp=result.timestamp,
        error=result.error,
        validation=ValidationInfo(**validation.to_dict()) if validation else None,
    )


@router.post("/read-image", response_model=OCRResponse)
async def read_from_image(
    request: Request,
    file: UploadFile = File(..., description="Image file of the machine display"),
    x: Optional[int] = None,
    y: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    """
    Upload an image and extract the machine counter value.
    Useful for testing OCR without a live camera.

    Optional query params for ROI: ?x=100&y=50&width=200&height=80
    """
    service = _get_service(request)
    image_bytes = await file.read()

    roi = None
    if all(v is not None for v in [x, y, width, height]):
        roi = ROIConfig(x=x, y=y, width=width, height=height)

    result = service.read_from_image(image_bytes, roi=roi)

    return OCRResponse(
        success=result.success,
        camera_id="uploaded_image",
        value=result.value,
        raw_text=result.raw_text,
        confidence=result.confidence,
        timestamp=result.timestamp,
        error=result.error,
    )


@router.post("/read-image-all", response_model=MultiOCRResponse)
async def read_all_from_image(
    request: Request,
    file: UploadFile = File(..., description="Image file of the machine display"),
    min_confidence: float = 0.4,
    min_height: int = 35,
    max_digits: int = 8,
):
    """
    Upload an image and detect ALL numbers visible on the display.
    Use this on a full machine screen (no ROI needed) to get every counter at once.

    Example response for the RMGT display:
        numbers: [ {value: 18917, ...}, {value: 2700, ...}, {value: 388, ...} ]
    """
    service = _get_service(request)
    image_bytes = await file.read()
    results = service.read_all_from_image(
        image_bytes, min_confidence=min_confidence, min_height=min_height, max_digits=max_digits
    )

    return MultiOCRResponse(
        camera_id="uploaded_image",
        count=len(results),
        numbers=[
            DetectedNumberResponse(
                value=d.value,
                raw_text=d.raw_text,
                confidence=d.confidence,
                bbox=BBoxModel(x=d.x, y=d.y, width=d.width, height=d.height),
            )
            for d in results
        ],
        timestamp=datetime.now(),
    )


@router.get("/read-all/{camera_id}", response_model=MultiOCRResponse)
def read_all_from_camera(
    camera_id: str,
    request: Request,
    min_confidence: float = 0.4,
):
    """
    Detect ALL numbers from a live registered camera frame.
    Returns every counter visible on the machine display.
    """
    service = _get_service(request)
    results = service.read_all_from_camera(camera_id, min_confidence=min_confidence)

    return MultiOCRResponse(
        camera_id=camera_id,
        count=len(results),
        numbers=[
            DetectedNumberResponse(
                value=d.value,
                raw_text=d.raw_text,
                confidence=d.confidence,
                bbox=BBoxModel(x=d.x, y=d.y, width=d.width, height=d.height),
            )
            for d in results
        ],
        timestamp=datetime.now(),
    )


@router.get("/snapshot/{camera_id}")
def snapshot_frame(camera_id: str, request: Request):
    """Returns a plain JPEG of the current frame — no annotations. Used by the ROI picker."""
    service = _get_service(request)
    cam = service._cameras.get(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    import cv2
    frame = cam.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet.")
    _, buf = cv2.imencode(".jpg", frame)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.get("/debug/{camera_id}")
def debug_frame(camera_id: str, request: Request):
    """
    Returns a JPEG image with the ROI box and OCR result drawn on the live frame.
    Open this URL in a browser to visually verify your ROI and OCR accuracy.
    """
    service = _get_service(request)
    jpeg_bytes = service.get_debug_image(camera_id)
    if jpeg_bytes is None:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' not found or no frame available yet.",
        )
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get("/stream/{camera_id}")
async def stream_camera(camera_id: str, request: Request):
    """
    MJPEG live stream with OCR bounding boxes drawn on every frame.
    Use as an <img> src in the frontend — browsers handle MJPEG natively.

    Example: <img src="http://localhost:8000/ocr/stream/machine_a_display" />
    """
    service = _get_service(request)

    # Validate camera exists before starting the stream
    if camera_id not in [c["camera_id"] for c in service.list_cameras()]:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    async def generate():
        while True:
            if await request.is_disconnected():
                break
            jpeg = service.get_debug_image(camera_id)
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.05)  # ~20 fps

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
