from __future__ import annotations

import asyncio
import math
from typing import Optional

from mavsdk import System, telemetry
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, PositionNedYaw

import config
from telemetry_state import TelemetryState, telemetry_monitor


def wrap_yaw_deg(yaw: float) -> float:
    while yaw > 180:
        yaw -= 360
    while yaw < -180:
        yaw += 360
    return yaw


def yaw_from_vector(north: float, east: float, fallback: float = 0.0) -> float:
    if abs(north) + abs(east) < 1e-6:
        return fallback
    return wrap_yaw_deg(math.degrees(math.atan2(east, north)))


def distance_3d(pos, n: float, e: float, d: float) -> float:
    return math.sqrt((pos.north_m-n)**2 + (pos.east_m-e)**2 + (pos.down_m-d)**2)


class FlightController:
    def __init__(self) -> None:
        self.drone = System()
        self.state = TelemetryState()
        self.stop_event = asyncio.Event()
        self.monitor_task: Optional[asyncio.Task] = None
        self.streamer_task: Optional[asyncio.Task] = None
        self.current_setpoint = PositionNedYaw(0.0, 0.0, config.CRUISE_DOWN_M, 0.0)
        self.offboard_started = False

    async def connect(self) -> None:
        print(f"Connecting to PX4 at {config.SYSTEM_ADDRESS} ...")
        await self.drone.connect(system_address=config.SYSTEM_ADDRESS)
        async for s in self.drone.core.connection_state():
            if s.is_connected:
                print("✅ Connected to PX4.")
                break
        self.monitor_task = asyncio.create_task(telemetry_monitor(self.drone, self.state, self.stop_event))

    async def wait_ready(self) -> None:
        print("Waiting for local NED pose / readiness...")
        start = asyncio.get_running_loop().time()
        last_print = 0.0
        while True:
            now = asyncio.get_running_loop().time()
            if now - start > config.HEALTH_TIMEOUT_S:
                raise TimeoutError("PX4 local position not ready. Use x500_vision and set EKF origin if needed.")
            h = self.state.health
            pose_ok = self.state.has_local_pose()
            local_ok = bool(getattr(h, "is_local_position_ok", False)) if h else False
            armable = bool(getattr(h, "is_armable", False)) if h else False
            home_ok = bool(getattr(h, "is_home_position_ok", False)) if h else False
            if now - last_print > 1.0:
                print(f"  status: pose={pose_ok}, local_ok={local_ok}, armable={armable}, home_ok={home_ok}")
                last_print = now
            if pose_ok and (local_ok or armable or home_ok):
                print("✅ Ready enough for local-position offboard flight.")
                return
            await asyncio.sleep(0.2)

    async def arm_takeoff_start_offboard(self) -> None:
        try:
            await self.drone.action.set_takeoff_altitude(config.TAKEOFF_ALTITUDE_M)
        except ActionError:
            pass
        print("Arming...")
        await self.drone.action.arm()
        print(f"Sending takeoff command to about {config.TAKEOFF_ALTITUDE_M:.1f} m...")
        await self.drone.action.takeoff()

        start = asyncio.get_running_loop().time()
        while asyncio.get_running_loop().time() - start < config.TAKEOFF_TIMEOUT_S:
            if self.state.position and self.state.position.down_m < -0.10:
                print(f"✅ Lift-off detected: D={self.state.position.down_m:.2f} m")
                break
            await asyncio.sleep(0.2)
        else:
            print("⚠️ No clear lift-off detected before timeout; Offboard climb will take over carefully.")

        # start setpoint at current pose to avoid jumps
        if self.state.position:
            self.current_setpoint = PositionNedYaw(
                self.state.position.north_m,
                self.state.position.east_m,
                self.state.position.down_m,
                self.state.yaw_deg or 0.0,
            )

        async def stream() -> None:
            period = 1.0 / config.SETPOINT_RATE_HZ
            while not self.stop_event.is_set():
                try:
                    await self.drone.offboard.set_position_ned(self.current_setpoint)
                except Exception as exc:
                    if not self.stop_event.is_set():
                        print(f"⚠️ setpoint stream warning: {type(exc).__name__}: {exc}")
                    await asyncio.sleep(period)
                    continue
                await asyncio.sleep(period)

        print("Starting setpoint stream before Offboard...")
        self.streamer_task = asyncio.create_task(stream())
        await asyncio.sleep(1.2)
        print("Starting Offboard mode...")
        try:
            await self.drone.offboard.start()
            self.offboard_started = True
            print("✅ Offboard started.")
        except OffboardError as exc:
            raise RuntimeError(f"Offboard failed: {exc}") from exc

        await self.goto_position_current_alt_safe(config.CRUISE_DOWN_M, label="climb to cruise altitude")

    async def goto_position_current_alt_safe(self, down_m: float, label: str = "altitude") -> None:
        if not self.state.position:
            return
        pos = self.state.position
        yaw = self.state.yaw_deg or 0.0
        self.current_setpoint = PositionNedYaw(pos.north_m, pos.east_m, down_m, yaw)
        print(f"➡️ {label}: N={pos.north_m:.1f} E={pos.east_m:.1f} D={down_m:.1f}")
        start = asyncio.get_running_loop().time()
        while asyncio.get_running_loop().time() - start < 18.0:
            if self.state.position and abs(self.state.position.down_m - down_m) < config.ALTITUDE_ACCEPT_RADIUS_M:
                print("✅ altitude reached/close enough")
                return
            await asyncio.sleep(0.25)
        print("⚠️ altitude wait timeout, continuing")

    async def set_position(self, north_m: float, east_m: float, down_m: float, yaw_deg: float) -> None:
        self.current_setpoint = PositionNedYaw(north_m, east_m, down_m, wrap_yaw_deg(yaw_deg))

    async def hold_seconds(self, seconds: float, yaw_deg: Optional[float] = None) -> None:
        if self.state.position:
            yaw = self.state.yaw_deg if yaw_deg is None else yaw_deg
            self.current_setpoint = PositionNedYaw(self.state.position.north_m, self.state.position.east_m, self.state.position.down_m, yaw or 0.0)
        await asyncio.sleep(seconds)

    async def yaw_scan(self, yaw_list: list[float], hold_s: float = 1.5) -> None:
        if not self.state.position:
            return
        pos = self.state.position
        for y in yaw_list:
            print(f"🔭 yaw scan {y:.0f}°")
            self.current_setpoint = PositionNedYaw(pos.north_m, pos.east_m, config.CRUISE_DOWN_M, wrap_yaw_deg(y))
            await asyncio.sleep(hold_s)

    async def land(self) -> None:
        print("Preparing to land...")
        try:
            if self.offboard_started:
                await self.drone.offboard.stop()
                print("Offboard stopped.")
        except Exception as exc:
            print(f"⚠️ offboard stop warning: {type(exc).__name__}: {exc}")
        try:
            print("Landing...")
            await self.drone.action.land()
        except Exception as exc:
            print(f"⚠️ land command warning: {type(exc).__name__}: {exc}")
        start = asyncio.get_running_loop().time()
        while asyncio.get_running_loop().time() - start < config.LAND_TIMEOUT_S:
            pos = self.state.position
            near_ground = bool(pos is not None and pos.down_m >= -config.LAND_ALTITUDE_ACCEPT_M)
            if self.state.landed_state == telemetry.LandedState.ON_GROUND or near_ground:
                print("✅ Landed / near ground confirmed.")
                return
            await asyncio.sleep(0.5)
        print("⚠️ Landing wait timeout. Check simulator visually.")

    async def shutdown(self) -> None:
        self.stop_event.set()
        if self.streamer_task:
            self.streamer_task.cancel()
            await asyncio.gather(self.streamer_task, return_exceptions=True)
        if self.monitor_task:
            self.monitor_task.cancel()
            await asyncio.gather(self.monitor_task, return_exceptions=True)
