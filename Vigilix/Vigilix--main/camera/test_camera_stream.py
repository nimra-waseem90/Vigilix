import os
import sys
import argparse
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from camera.camera_stream import CameraStream

DEFAULT_SOURCE = "data/raw/videos/test.mp4"


def main():
    parser = argparse.ArgumentParser(description="Vigilix Camera Stream Test")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Video file path, RTSP URL, or webcam index (default: data/raw/videos/test.mp4)")
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        source = int(source)
    elif not source.startswith("rtsp://") and not os.path.exists(source):
        print(f"⚠️ Source file '{source}' not found. Please provide an existing video path or RTSP URL with --source.")
        return

    print("\n========================================")
    print("       VIGILIX CAMERA STREAM TEST")
    print("========================================")

    camera = CameraStream(source=source, threaded=True)

    try:
        camera.start()
        print("\n✅ Video stream opened successfully")
        fps = camera.get_fps()
        print(f"FPS: {fps}")
        print("\nPress Q to stop the stream.")

        frame_count = 0
        while True:
            frame = camera.read()
            if frame is None:
                print("\n⚠️ End of video / no frame received.")
                break

            frame_count += 1
            cv2.imshow("Vigilix Camera Stream", frame)

            if frame_count % 30 == 0:
                print(f"Frames received: {frame_count}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n🛑 Stream stopped by user.")
                break

    except Exception as e:
        print(f"\n❌ Camera stream error: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\n========================================")
        print("       CAMERA STREAM TEST COMPLETE")
        print("========================================")


if __name__ == "__main__":
    main()