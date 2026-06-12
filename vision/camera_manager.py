import os

# Must be set BEFORE cv2 is imported. Without these, FFmpeg blocks forever on a
# dead/slow network stream (phone IP camera, RTSP) — read() and release() hang,
# which wedges the whole backend process during shutdown/reload.
# rw_timeout is in microseconds: 5s I/O timeout on network sources.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rw_timeout;5000000")

import cv2
import threading
import time
import logging
import numpy as np
from typing import Optional, Union

logger = logging.getLogger(__name__)


def _parse_screen_source(source: str):
    """
    Parse screen capture source string.
    Formats:
        "screen"            → full primary monitor
        "screen:0"          → monitor index 0
        "screen:x,y,w,h"   → region e.g. "screen:100,200,800,600"
    Returns (monitor_index_or_region_dict)
    """
    part = source[len("screen"):].lstrip(":")
    if not part:
        return {"monitor": 1}  # mss monitor 1 = primary
    if part.isdigit():
        return {"monitor": int(part) + 1}  # mss is 1-indexed
    try:
        x, y, w, h = [int(v.strip()) for v in part.split(",")]
        return {"top": y, "left": x, "width": w, "height": h}
    except ValueError:
        raise ValueError(f"Invalid screen source format: '{source}'. Use 'screen', 'screen:0', or 'screen:x,y,w,h'")


class CameraManager:
    """
    Thread-safe camera stream manager.
    Supports RTSP streams, local camera index, video files, and screen capture.

    Source formats:
        "0"                        → webcam index 0
        "rtsp://192.168.1.x/..."   → RTSP IP camera
        "C:/path/to/video.mp4"     → video file (loops)
        "screen"                   → full primary screen capture
        "screen:0"                 → monitor 0 screen capture
        "screen:x,y,w,h"          → region of screen e.g. "screen:100,200,800,400"
    """

    def __init__(
        self,
        source: Union[str, int],
        camera_id: str = "cam0",
        reconnect_delay: float = 5.0,
        loop: bool = True,
    ):
        self.source = source
        self.camera_id = camera_id
        self.reconnect_delay = reconnect_delay
        self._loop = loop

        src = str(source).strip('"\'')

        self._is_screen = src.startswith("screen")
        self._is_video_file = (
            not self._is_screen
            and isinstance(source, str)
            and not src.startswith("rtsp://")
            and not src.startswith("rtmp://")
            and not src.startswith("http")
            and src != "0"
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._sct = None          # mss instance for screen capture
        self._screen_region = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start background frame capture thread."""
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"[{self.camera_id}] Capture thread started | source: {self.source}")

    def stop(self):
        """Stop frame capture and release resources. Never blocks."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        # Release in a fire-and-forget daemon thread — cap.release() on a dead
        # network stream (phone IP camera, RTSP) can block indefinitely inside
        # FFmpeg and wedge the whole process during shutdown/reload.
        cap, sct = self._cap, self._sct
        self._cap = None
        self._sct = None

        def _release():
            try:
                if cap:
                    cap.release()
                if sct:
                    sct.close()
            except Exception:
                pass

        threading.Thread(target=_release, daemon=True, name=f"release-{self.camera_id}").start()
        self._connected = False
        logger.info(f"[{self.camera_id}] Stopped.")

    def get_frame(self) -> Optional[np.ndarray]:
        """Return latest frame as BGR numpy array, or None if not ready."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_connected(self) -> bool:
        return self._connected

    def is_done(self) -> bool:
        """True when a one-shot video file has finished playing (not applicable to live cameras)."""
        return self._is_video_file and not self._loop and not self._running

    def get_status(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "source": str(self.source),
            "connected": self._connected,
            "has_frame": self._frame is not None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self):
        """Continuously reads frames. Reconnects on failure."""
        if self._is_screen:
            self._capture_loop_screen()
        else:
            self._capture_loop_cv2()

    def _capture_loop_screen(self):
        """Screen capture loop using mss."""
        try:
            import mss
        except ImportError:
            logger.error(f"[{self.camera_id}] mss not installed. Run: pip install mss")
            return

        src = str(self.source).strip('"\'')
        try:
            region = _parse_screen_source(src)
        except ValueError as e:
            logger.error(f"[{self.camera_id}] {e}")
            return

        logger.info(f"[{self.camera_id}] Screen capture | region: {region}")
        with mss.mss() as sct:
            self._connected = True
            while self._running:
                try:
                    screenshot = sct.grab(region)
                    # BGRA → BGR
                    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
                    with self._lock:
                        self._frame = frame
                    time.sleep(0.033)  # ~30 fps screen capture
                except Exception as e:
                    logger.error(f"[{self.camera_id}] Screen capture error: {e}")
                    time.sleep(1)
        self._connected = False

    def _capture_loop_cv2(self):
        """Standard OpenCV capture loop for RTSP / webcam / video files."""
        while self._running:
            if not self._connect():
                logger.warning(f"[{self.camera_id}] Connection failed. Retrying in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                continue

            # For video files, pace playback to the file's native FPS so the OCR
            # poller sees frames at real-world speed (counters change at the right rate).
            # Live cameras / webcams: no pacing needed — they block on read() naturally.
            frame_delay = 0.0
            if self._is_video_file:
                fps = self._cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 0:
                    frame_delay = 1.0 / fps
                    logger.info(f"[{self.camera_id}] Video FPS={fps:.1f} → pacing at {frame_delay*1000:.0f}ms/frame")

            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    if self._is_video_file:
                        if self._loop:
                            # Loop mode: restart from beginning
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            # One-shot mode: video ended, stop cleanly.
                            # Keep self._frame so the poller can finish processing the last frame.
                            logger.info(f"[{self.camera_id}] Video file ended (one-shot mode). Stopping.")
                            self._running = False
                            self._connected = False
                            break
                    logger.warning(f"[{self.camera_id}] Frame read failed. Reconnecting...")
                    self._connected = False
                    self._cap.release()
                    time.sleep(self.reconnect_delay)
                    break

                with self._lock:
                    self._frame = frame
                self._connected = True

                if frame_delay > 0:
                    time.sleep(frame_delay)

    def _connect(self) -> bool:
        """Open the video/camera source. Returns True on success."""
        try:
            source = self.source.strip('"\'') if isinstance(self.source, str) else self.source
            # Convert webcam index string to int
            if isinstance(source, str) and source.isdigit():
                source = int(source)

            if isinstance(source, int) and os.name == "nt":
                # Windows webcams: the default MSMF backend can take 90+ seconds
                # to open (measured) — the UI gives up long before that.
                # DirectShow opens the same camera in under a second.
                self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                # MJPG first: in raw YUY2 mode most webcams silently cap at
                # 640x480; MJPG unlocks the sensor's full 1080p — 3x the pixels
                # on the counter digits.
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                got_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                got_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"[{self.camera_id}] Webcam capture resolution: {got_w}x{got_h}")
            else:
                self._cap = cv2.VideoCapture(source)

            if not self._cap.isOpened():
                logger.error(f"[{self.camera_id}] Could not open source: {self.source}")
                return False
            self._connected = True
            logger.info(f"[{self.camera_id}] Connected.")
            return True
        except Exception as e:
            logger.error(f"[{self.camera_id}] Connection error: {e}")
            return False
