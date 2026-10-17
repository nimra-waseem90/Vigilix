import os
import argparse
from fusion.fusion_engine import FusionEngine
from detection.fire_detection import detect_fire
from detection.weapon_detection import detect_weapon

DEFAULT_FIRE_SOURCE = os.environ.get("VIGILIX_FIRE_SOURCE", "data/raw/videos/fire_test.mp4")
DEFAULT_WEAPON_SOURCE = os.environ.get("VIGILIX_WEAPON_SOURCE", "data/raw/videos/weapon_test.mp4")


def process_event(fusion, event):
    if event is None:
        print("\nNo event detected.")
        return

    print("\n----------------------------------------")
    print("EVENT RECEIVED")
    print("----------------------------------------")
    print(event)

    fused_alert = fusion.add_event(event)
    if fused_alert:
        print("\n========================================")
        print("        VIGILIX FUSION ALERT")
        print("========================================")
        print(f"Event      : {fused_alert.get('event')}")
        print(f"Label      : {fused_alert.get('label')}")
        print(f"Confidence : {fused_alert.get('confidence')}")
        print(f"Severity   : {fused_alert.get('severity')}")
        print(f"Timestamp  : {fused_alert.get('timestamp')}")
        print("\nSource Events:")
        for source_event in fused_alert.get("source_events", []):
            print(source_event)
        print("========================================")


def main():
    parser = argparse.ArgumentParser(description="Vigilix Fusion & Detection CLI Test")
    parser.add_argument("--fire-source", default=DEFAULT_FIRE_SOURCE,
                        help="RTSP URL, video file, or camera index for fire detection")
    parser.add_argument("--weapon-source", default=DEFAULT_WEAPON_SOURCE,
                        help="RTSP URL, video file, or camera index for weapon detection")
    parser.add_argument("--skip-fire", action="store_true", help="Skip fire detection test")
    parser.add_argument("--skip-weapon", action="store_true", help="Skip weapon detection test")
    args = parser.parse_args()

    print("\n========================================")
    print("          VIGILIX FUSION SYSTEM")
    print("========================================")

    fusion = FusionEngine(time_window=5.0)

    if not args.skip_fire:
        if not args.fire_source.startswith("rtsp://") and not os.path.exists(args.fire_source):
            print(f"\n⚠️ Fire source '{args.fire_source}' not found. Provide a valid video path or RTSP stream with --fire-source.")
        else:
            print(f"\n🔥 Starting Fire Detection on {args.fire_source}...")
            fire_event = detect_fire(args.fire_source)
            process_event(fusion, fire_event)

    fusion.clear_events()

    if not args.skip_weapon:
        if not args.weapon_source.startswith("rtsp://") and not os.path.exists(args.weapon_source):
            print(f"\n⚠️ Weapon source '{args.weapon_source}' not found. Provide a valid video path or RTSP stream with --weapon-source.")
        else:
            print(f"\n🔫 Starting Weapon Detection on {args.weapon_source}...")
            weapon_event = detect_weapon(args.weapon_source)
            process_event(fusion, weapon_event)

    print("\n========================================")
    print("       VIGILIX FUSION TEST COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()