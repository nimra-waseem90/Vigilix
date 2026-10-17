import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from datetime import datetime
from ultralytics import YOLO

from camera.camera_stream import CameraStream
from alert.alert_manager import fire_alert
from fusion.event_format import create_event

MODEL_PATH = "model/detection/fire_smoke_model.pt"
CONFIDENCE_THRESHOLD = 0.60
REQUIRED_FRAMES = 3
FIRE_CLASS_ID = 1
RTSP_URL = "rtsp://127.0.0.1:8554/fire"


def detect_fire(camera_or_source=RTSP_URL):
    """Run fire detection on a CameraStream instance or a video/RTSP source string."""
    os.makedirs("alert/screenshots", exist_ok=True)
    print("Loading Fire YOLO model...")
    model = YOLO(MODEL_PATH)
    print("Fire model loaded successfully.")

    is_internal_camera = False
    if isinstance(camera_or_source, str):
        camera = CameraStream(source=camera_or_source, threaded=False)
        camera.start()
        is_internal_camera = True
    else:
        camera = camera_or_source

    print("✅ Fire detection stream started. Press Q to quit.")
    consecutive_fire_frames = 0
    alert_sent = False
    frame_number = 0
    fps = camera.get_fps() or 30.0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                if is_internal_camera:
                    print("⚠️ End of stream reached.")
                    break
                continue

            frame_number += 1
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            fire_detected = False
            highest_confidence = 0.0

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id == FIRE_CLASS_ID:
                        fire_detected = True
                        if confidence > highest_confidence:
                            highest_confidence = confidence

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            frame,
                            f"FIRE {confidence:.2f}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

            if fire_detected:
                consecutive_fire_frames += 1
                print(
                    f"🔥 FIRE DETECTED | Confirmation: "
                    f"{consecutive_fire_frames}/{REQUIRED_FRAMES} | "
                    f"Confidence: {highest_confidence:.2f}"
                )
            else:
                consecutive_fire_frames = 0

            if consecutive_fire_frames >= REQUIRED_FRAMES and not alert_sent:
                print("\n🚨 FIRE CONFIRMED!")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"alert/screenshots/fire_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"📸 Fire screenshot saved: {screenshot_path}")

                fire_alert(highest_confidence, screenshot_path=screenshot_path)
                print("🚨 Fire alarm triggered.")

                fire_event = create_event(
                    event="fire",
                    label="Fire",
                    confidence=highest_confidence,
                    timestamp=frame_number / fps,
                )
                alert_sent = True
                return fire_event

            status_text = (
                f"FIRE DETECTION {consecutive_fire_frames}/{REQUIRED_FRAMES}"
                if fire_detected
                else "Monitoring..."
            )
            color = (0, 0, 255) if fire_detected else (0, 255, 0)
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Vigilix - Fire Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nFire detection stopped by user.")
                break

    finally:
        if is_internal_camera:
            camera.release()
        cv2.destroyAllWindows()

    print("Fire detection completed.")
    return None


if __name__ == "__main__":
    detect_fire(RTSP_URL)
