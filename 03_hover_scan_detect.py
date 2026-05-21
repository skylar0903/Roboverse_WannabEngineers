#!/usr/bin/env python3
"""Safe flying perception test.

This test does NOT search the map yet. It:
1. connects and takes off like Phase 1.1,
2. starts RGB detection in the background,
3. hovers and scans yaw angles,
4. logs red/yellow detections once,
5. lands.

Run this only after 02_rgb_detection_test.py receives camera frames.
"""

import asyncio
import time

import config
from detection_memory import DetectionMemory
from flight_controller import FlightController
from gz_topic_tools import choose_topic, print_topic_help
from image_receivers import RgbReceiver
from perception import CombinedDetector

try:
    import cv2
except Exception:
    cv2 = None


async def detection_loop(receiver, detector, memory, controller, stop_event):
    last_processed = 0.0
    while not stop_event.is_set():
        pkt = receiver.get_latest()
        pos = controller.state.position
        if pkt is None or pos is None:
            await asyncio.sleep(0.05)
            continue
        # Do not process every frame; keep the VM responsive.
        if time.time() - last_processed < config.DETECTION_COOLDOWN_S:
            await asyncio.sleep(0.03)
            continue
        last_processed = time.time()
        dets = detector.detect(pkt.frame)
        if dets:
            path = detector.annotate_and_save(pkt.frame, dets, prefix="hover_scan")
            for d in dets:
                if "red" in d.class_name.lower() or "yellow" in d.class_name.lower():
                    memory.add_if_new(d.class_name, d.confidence, pos.north_m, pos.east_m, pos.down_m, d.source, path)
        await asyncio.sleep(0.02)


async def main():
    topic = choose_topic(
        config.RGB_TOPIC,
        config.PREFERRED_RGB_TOPICS,
        must_contain_any=["image", "camera"],
        avoid_contains_any=["depth", "points", "camera_info"],
    )
    if not topic:
        print("❌ Could not auto-find RGB topic.")
        print_topic_help()
        return

    receiver = RgbReceiver(topic, name="rgb")
    detector = CombinedDetector()
    memory = DetectionMemory(config.LOG_DIR)
    controller = FlightController()
    det_stop = asyncio.Event()
    det_task = None

    try:
        await controller.connect()
        await controller.wait_ready()
        await controller.arm_takeoff_start_offboard()

        det_task = asyncio.create_task(detection_loop(receiver, detector, memory, controller, det_stop))

        print("\n🔭 Hover yaw scan begins. Watch for detections.")
        yaw_list = [0, 45, 90, 135, 180, -135, -90, -45, 0]
        await controller.yaw_scan(yaw_list, hold_s=1.8)
        print(f"\nDetection summary: {memory.summary()}")
        await controller.hold_seconds(1.0)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt.")
    except Exception as exc:
        print(f"\n❌ hover scan error: {type(exc).__name__}: {exc}")
    finally:
        det_stop.set()
        if det_task:
            await asyncio.gather(det_task, return_exceptions=True)
        memory.close()
        await controller.land()
        await controller.shutdown()
        if cv2 is not None:
            cv2.destroyAllWindows()
        print(f"Logs saved in {config.LOG_DIR}/ and images in {config.DETECTION_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
