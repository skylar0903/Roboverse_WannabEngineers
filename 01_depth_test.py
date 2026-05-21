#!/usr/bin/env python3
import time

import config
from gz_topic_tools import choose_topic, print_topic_help
from image_receivers import DepthReceiver
from perception import DepthAnalyzer


def main():
    topic = choose_topic(
        config.DEPTH_TOPIC,
        config.PREFERRED_DEPTH_TOPICS,
        must_contain_any=["depth"],
        avoid_contains_any=["points", "pointcloud", "camera_info"],
    )
    if not topic:
        print("❌ Could not auto-find depth topic.")
        print_topic_help()
        return

    receiver = DepthReceiver(topic, name="depth")
    analyzer = DepthAnalyzer()
    print("Waiting for depth frames...")
    pkt = receiver.wait_for_frame(timeout_s=10)
    if pkt is None:
        print("❌ No depth frame received in 10s. Check topic in config.py")
        return
    print(f"✅ First depth frame: {pkt.width}x{pkt.height}, encoding={pkt.encoding_guess}")
    print("Printing L/C/R distances. Ctrl+C to stop.\n")

    try:
        while True:
            pkt = receiver.get_latest()
            if pkt is not None:
                s = analyzer.summarize(pkt.frame)
                print(
                    f"L={s.left_m:5.2f} m | C={s.center_m:5.2f} m | R={s.right_m:5.2f} m | "
                    f"min={s.min_m:5.2f} | {s.state:13s} -> {s.recommended:18s}",
                    end="\r",
                )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopped depth test.")


if __name__ == "__main__":
    main()
