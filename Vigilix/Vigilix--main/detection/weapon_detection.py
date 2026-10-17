import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from datetime import datetime
from ultralytics import YOLO

from camera.camera_stream import CameraStream
from alert.alert_manager import weapon_alert
from fusion.event_format import create_event

MODEL_PATH = "model/detection/weapon_yolo26x.pt"
RTSP_URL = "rtsp://127.0.0.1:8554/weapon"

CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMAGE_SIZE = int(os.environ.get("VIGILIX_WEAPON_IMGSZ", "1024"))
REQUIRED_FRAMES = 3


def _get_device():
    import torch
    return os.environ.get("VIGILIX_WEAPON_DEVICE", "0" if torch.cuda.is_available() else "cpu")


DEVICE = _get_device()
IGNORED_CLASSES = {"Person", "Fire Smoke", "Fire_Smoke"}


def detect_weapon(camera_or_source=RTSP_URL):
    """Run weapon detection on a CameraStream instance or a video/RTSP source string."""
    is_cuda = (DEVICE != "cpu")
    model = YOLO(MODEL_PATH)

    is_internal_camera = False
    if isinstance(camera_or_source, str):
        camera = CameraStream(source=camera_or_source, threaded=False)
        camera.start()
        is_internal_camera = True
    else:
        camera = camera_or_source

    consecutive_weapon_frames = 0
    alert_sent = False
    os.makedirs("alert/screenshots", exist_ok=True)
    frame_number = 0
    fps = camera.get_fps() or 30.0

    print("\n🔫 Weapon detector started. Press Q to quit.\n")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                if is_internal_camera:
                    print("⚠️ End of video / stream.")
                    break
                continue

            frame_number += 1

            pred_kwargs = {
                "source": frame,
                "conf": CONFIDENCE_THRESHOLD,
                "iou": IOU_THRESHOLD,
                "imgsz": IMAGE_SIZE,
                "device": DEVICE,
                "verbose": False,
            }
            if is_cuda:
                pred_kwargs["half"] = True
            results = model.predict(**pred_kwargs)

            result = results[0]
            weapon_detected = False
            highest_confidence = 0.0
            detected_weapon = None

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                weapon_type = result.names[class_id]

                if weapon_type in IGNORED_CLASSES:
                    continue

                if confidence > highest_confidence:
                    highest_confidence = confidence
                    detected_weapon = weapon_type

                weapon_detected = True

            if weapon_detected:
                consecutive_weapon_frames += 1
                print(
                    f"🔫 Weapon detected | Frame confirmation: "
                    f"{consecutive_weapon_frames}/{REQUIRED_FRAMES} | "
                    f"Type: {detected_weapon} | "
                    f"Confidence: {highest_confidence:.2f}"
                )
            else:
                consecutive_weapon_frames = 0

            annotated_frame = result.plot()
            cv2.imshow("Vigilix - Weapon Detection", annotated_frame)

            if consecutive_weapon_frames >= REQUIRED_FRAMES and not alert_sent:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"alert/screenshots/weapon_{detected_weapon}_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)

                weapon_alert(detected_weapon, highest_confidence, screenshot_path)

                weapon_event = create_event(
                    event="weapon",
                    label=detected_weapon,
                    confidence=highest_confidence,
                    timestamp=frame_number / fps,
                )

                print("\n🚨 STANDARDIZED WEAPON EVENT:")
                print(weapon_event)
                alert_sent = True
                return weapon_event

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nWeapon detection stopped by user.")
                break

    finally:
        if is_internal_camera:
            camera.release()
        cv2.destroyAllWindows()

    print("\nWeapon detection completed!")
    return None


if __name__ == "__main__":
    detect_weapon(RTSP_URL)