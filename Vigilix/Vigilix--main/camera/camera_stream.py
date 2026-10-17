import os
import threading
import time
import cv2

# Ensure OpenCV uses TCP for RTSP to prevent packet drop artifacts
if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class CameraStream:
    """High-performance camera/video capture stream with low-latency buffering
    and optional threaded reading to prevent RTSP stale-frame accumulation."""

    def __init__(self, source=0, threaded=True):
        self.source = source
        self.threaded = threaded
        self.cap = None

        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._new_frame_event = threading.Event()
        self._last_read_time = 0.0

    def start(self):
        print(f"\n📡 Connecting to source: {self.source}")

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

        # Set minimal buffer size for RTSP / webcam live streams to avoid lag
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        print("✅ Stream connected successfully")

        if self.threaded:
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

        return self

    def _capture_loop(self):
        """Continuously pulls frames from the capture source so the internal
        OpenCV / FFmpeg buffer never accumulates stale frames."""
        while self._running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                with self._lock:
                    self._latest_frame = None
                time.sleep(0.01)
                continue

            with self._lock:
                self._latest_frame = frame
                self._last_read_time = time.time()
            self._new_frame_event.set()

    def read(self):
        if self.cap is None:
            raise RuntimeError("CameraStream has not been started.")

        if self.threaded:
            with self._lock:
                if self._latest_frame is None:
                    return None
                return self._latest_frame.copy()

        ret, frame = self.cap.read()
        return frame if ret else None

    def get_fps(self):
        if self.cap is None:
            return 30.0

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    def release(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        print("📴 Stream released.")