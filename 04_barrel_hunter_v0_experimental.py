#!/usr/bin/env python3
"""Experimental 5-minute barrel-hunter v0.

Run order:
1. 00_list_gz_topics.py
2. 01_depth_test.py
3. 02_rgb_detection_test.py
4. 03_hover_scan_detect.py
5. ONLY THEN this file.

This is a compact competition-style controller:
- take off to 2 m
- follow a lawnmower search pattern inside configurable N/E bounds
- run colour/YOLO detection continuously
- use depth left/centre/right as a simple obstacle guard
- log each red/yellow detection once

It is intentionally simpler than full SLAM because the Qualifier rewards fast reliable
barrel detection, not beautiful mapping.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import List, Tuple

import config
from detection_memory import DetectionMemory
from flight_controller import FlightController, yaw_from_vector, wrap_yaw_deg
from gz_topic_tools import choose_topic, print_topic_help
from image_receivers import DepthReceiver, RgbReceiver
from perception import CombinedDetector, DepthAnalyzer

try:
    import cv2
except Exception:
    cv2 = None


def build_lawnmower_waypoints() -> List[Tuple[float, float]]:
    """Return [(north, east), ...] lawnmower path.

    Uses config search bounds. Assumes start is somewhere near map centre. Tune config
    if your takeoff origin is near a corner.
    """
    waypoints: List[Tuple[float, float]] = []
    e = config.SEARCH_E_MIN
    row = 0
    while e <= config.SEARCH_E_MAX + 1e-6:
        if row % 2 == 0:
            waypoints.append((config.SEARCH_N_MIN, e))
            waypoints.append((config.SEARCH_N_MAX, e))
        else:
            waypoints.append((config.SEARCH_N_MAX, e))
            waypoints.append((config.SEARCH_N_MIN, e))
        e += config.LANE_SPACING_M
        row += 1
    return waypoints


def norm2(n: float, e: float) -> float:
    return math.hypot(n, e)


def unit2(n: float, e: float) -> Tuple[float, float]:
    mag = norm2(n, e)
    if mag < 1e-6:
        return 0.0, 0.0
    return n / mag, e / mag


async def detection_loop(receiver, detector, memory, controller, stop_event):
    last_processed = 0.0
    while not stop_event.is_set():
        pkt = receiver.get_latest()
        pos = controller.state.position
        if pkt is None or pos is None:
            await asyncio.sleep(0.04)
            continue
        if time.time() - last_processed < config.DETECTION_COOLDOWN_S:
            await asyncio.sleep(0.02)
            continue
        last_processed = time.time()
        dets = detector.detect(pkt.frame)
        if dets:
            path = detector.annotate_and_save(pkt.frame, dets, prefix="hunter")
            for d in dets:
                low = d.class_name.lower()
                if "red" in low or "yellow" in low:
                    memory.add_if_new(d.class_name, d.confidence, pos.north_m, pos.east_m, pos.down_m, d.source, path)
        await asyncio.sleep(0.01)


async def hunter_loop(controller, depth_receiver, memory, stop_event):
    analyzer = DepthAnalyzer()
    waypoints = build_lawnmower_waypoints()
    wp_idx = 0
    start = time.time()
    last_print = 0.0

    print("\n🚀 Barrel hunter v0 started.")
    print(f"Waypoints: {len(waypoints)} | time push: {config.MISSION_TIME_LIMIT_S:.0f}s")
    print("Press Ctrl+C to stop and land.\n")

    while not stop_event.is_set():
        elapsed = time.time() - start
        if elapsed > config.MISSION_TIME_LIMIT_S:
            print("\n⏱️ 5-minute scoring push complete.")
            return
        if wp_idx >= len(waypoints):
            print("\n✅ Lawnmower path complete.")
            return

        pos = controller.state.position
        if pos is None:
            await asyncio.sleep(0.05)
            continue

        target_n, target_e = waypoints[wp_idx]
        err_n = target_n - pos.north_m
        err_e = target_e - pos.east_m
        dist = norm2(err_n, err_e)
        if dist <= config.WAYPOINT_ACCEPT_RADIUS_M:
            print(f"✅ lane waypoint {wp_idx+1}/{len(waypoints)} reached: N={target_n:.1f} E={target_e:.1f}")
            wp_idx += 1
            # quick yaw scan at lane turn improves chance of seeing red/yellow on sides
            if wp_idx % 2 == 0:
                current_yaw = controller.state.yaw_deg or 0.0
                await controller.yaw_scan([current_yaw - 35, current_yaw, current_yaw + 35, current_yaw], hold_s=0.35)
            continue

        goal_n, goal_e = unit2(err_n, err_e)

        # Depth obstacle guard. If depth is missing, still fly the search path.
        depth_pkt = depth_receiver.get_latest() if depth_receiver is not None else None
        depth_state = "NO_DEPTH"
        avoid_n, avoid_e = 0.0, 0.0
        if depth_pkt is not None:
            s = analyzer.summarize(depth_pkt.frame)
            depth_state = f"{s.state} L={s.left_m:.1f} C={s.center_m:.1f} R={s.right_m:.1f}"
            # Convert body lateral avoidance to NED using current yaw.
            yaw_rad = math.radians(controller.state.yaw_deg or 0.0)
            # Body right vector in NED = [sin(yaw), cos(yaw)]
            right_n, right_e = math.sin(yaw_rad), math.cos(yaw_rad)
            if s.state == "DANGER_FRONT":
                # stronger sidestep away from closer side
                sign = -1.0 if s.left_m > s.right_m else 1.0  # - = left, + = right relative body
                avoid_n += sign * right_n * 1.5
                avoid_e += sign * right_e * 1.5
                goal_n *= 0.25
                goal_e *= 0.25
            elif s.state == "BLOCKED":
                # step backward / yaw scan if boxed in
                forward_n, forward_e = math.cos(yaw_rad), math.sin(yaw_rad)
                avoid_n += -forward_n * 1.2
                avoid_e += -forward_e * 1.2
                goal_n *= 0.0
                goal_e *= 0.0
            elif s.state == "CAUTION_FRONT":
                sign = -1.0 if s.left_m > s.right_m else 1.0
                avoid_n += sign * right_n * 0.55
                avoid_e += sign * right_e * 0.55
                goal_n *= 0.7
                goal_e *= 0.7

        move_n, move_e = unit2(goal_n + avoid_n, goal_e + avoid_e)
        if abs(move_n) + abs(move_e) < 1e-6:
            move_n, move_e = goal_n, goal_e
        lookahead = min(config.LOOKAHEAD_M, max(0.65, dist))
        cmd_n = pos.north_m + move_n * lookahead
        cmd_e = pos.east_m + move_e * lookahead
        # Clamp to configured map bounds so the virtual target does not run away.
        cmd_n = max(config.SEARCH_N_MIN - 1.0, min(config.SEARCH_N_MAX + 1.0, cmd_n))
        cmd_e = max(config.SEARCH_E_MIN - 1.0, min(config.SEARCH_E_MAX + 1.0, cmd_e))

        path_yaw = yaw_from_vector(move_n, move_e, controller.state.yaw_deg or 0.0)
        if config.YAW_MODE == "SLOW_SWEEP":
            sweep = config.YAW_SWEEP_DEG * math.sin(2 * math.pi * elapsed / max(config.YAW_SWEEP_PERIOD_S, 1.0))
            yaw = wrap_yaw_deg(path_yaw + sweep)
        else:
            yaw = path_yaw

        await controller.set_position(cmd_n, cmd_e, config.CRUISE_DOWN_M, yaw)

        if time.time() - last_print > 1.0:
            last_print = time.time()
            print(
                f"t={elapsed:5.1f}s | wp {wp_idx+1}/{len(waypoints)} dist={dist:4.1f} | "
                f"cmd N={cmd_n:5.1f} E={cmd_e:5.1f} yaw={yaw:6.1f} | {depth_state} | {memory.summary()}"
            )

        await asyncio.sleep(1.0 / config.CONTROL_LOOP_HZ)


async def main():
    rgb_topic = choose_topic(
        config.RGB_TOPIC,
        config.PREFERRED_RGB_TOPICS,
        must_contain_any=["image", "camera"],
        avoid_contains_any=["depth", "points", "camera_info"],
    )
    depth_topic = choose_topic(
        config.DEPTH_TOPIC,
        config.PREFERRED_DEPTH_TOPICS,
        must_contain_any=["depth"],
        avoid_contains_any=["points", "pointcloud", "camera_info"],
    )
    if not rgb_topic:
        print("❌ RGB topic not found; cannot run hunter.")
        print_topic_help()
        return
    if not depth_topic:
        print("⚠️ Depth topic not found. Hunter will run WITHOUT obstacle guard. Not recommended.")

    rgb = RgbReceiver(rgb_topic, name="rgb")
    depth = DepthReceiver(depth_topic, name="depth") if depth_topic else None
    detector = CombinedDetector()
    memory = DetectionMemory(config.LOG_DIR)
    controller = FlightController()
    stop_event = asyncio.Event()
    det_task = None

    try:
        await controller.connect()
        await controller.wait_ready()
        await controller.arm_takeoff_start_offboard()
        det_task = asyncio.create_task(detection_loop(rgb, detector, memory, controller, stop_event))
        await hunter_loop(controller, depth, memory, stop_event)
        print(f"\nFinal detection summary: {memory.summary()}")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt: landing.")
    except Exception as exc:
        print(f"\n❌ hunter error: {type(exc).__name__}: {exc}")
    finally:
        stop_event.set()
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
