#!/usr/bin/env python3
import os
import time

import config
from gz_topic_tools import choose_topic, print_topic_help
from image_receivers import RgbReceiver
from perception import CombinedDetector

try:
    import cv2
except Exception:
    cv2 = None


def main():
    topic = choose_topic(
        config.RGB_TOPIC,
        config.PREFERRED_RGB_TOPICS,
        must_contain_any=["image", "camera"],
        avoid_contains_any=["depth", "points", "camera_info"],
    )
    if not topic:
        print("❌ Could not auto-find RGB camera topic.")
        print_topic_help()
        return

    receiver = RgbReceiver(topic, name="rgb")
    detector = CombinedDetector()

    print("Waiting for RGB frames...")
    pkt = receiver.wait_for_frame(timeout_s=10)
    if pkt is None:
        print("❌ No RGB frame received in 10s. Check topic in config.py")
        return
    print(f"✅ First RGB frame: {pkt.width}x{pkt.height}, encoding={pkt.encoding_guess}")
    print("Point the drone/camera at barrels. Ctrl+C to stop.\n")

    last_det = 0.0
    try:
        while True:
            pkt = receiver.get_latest()
            if pkt is None:
                time.sleep(0.05)
                continue
            frame = pkt.frame
            dets = detector.detect(frame)
            if dets and time.time() - last_det > config.DETECTION_COOLDOWN_S:
                last_det = time.time()
                path = detector.annotate_and_save(frame, dets, prefix="rgb_test")
                print("\n🎯 Detections:")
                for d in dets:
                    print(f"  {d.class_name:14s} conf={d.confidence:.2f} source={d.source} bbox={d.bbox_xyxy}")
                if path:
                    print(f"  saved: {path}")
            if config.ENABLE_DISPLAY and cv2 is not None:
                show = frame.copy()
                for d in dets:
                    x1, y1, x2, y2 = d.bbox_xyxy
                    cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(show, d.class_name, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                cv2.imshow("RGB barrel detector test", show)
                cv2.waitKey(1)
            time.sleep(0.08)
    except KeyboardInterrupt:
        print("\nStopped RGB detection test.")
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
