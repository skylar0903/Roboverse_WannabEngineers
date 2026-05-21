from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional


@dataclass
class NedPosition:
    north_m: float
    east_m: float
    down_m: float


@dataclass
class NedVelocity:
    north_m_s: float
    east_m_s: float
    down_m_s: float


class TelemetryState:
    def __init__(self) -> None:
        self.position: Optional[NedPosition] = None
        self.velocity: Optional[NedVelocity] = None
        self.yaw_deg: Optional[float] = None
        self.flight_mode: Optional[str] = None
        self.is_armed: Optional[bool] = None
        self.landed_state: Optional[object] = None
        self.health: Optional[object] = None

    def has_local_pose(self) -> bool:
        return self.position is not None and self.yaw_deg is not None


async def telemetry_monitor(drone, state: TelemetryState, stop_event: asyncio.Event) -> None:
    """Minimal telemetry monitor.

    Kept intentionally smaller than Phase 1.1 because the VM occasionally resets the
    MAVSDK gRPC server; fewer streams reduces moving parts during perception tests.
    """

    async def monitor_position_velocity() -> None:
        try:
            async for pv in drone.telemetry.position_velocity_ned():
                if stop_event.is_set():
                    break
                state.position = NedPosition(pv.position.north_m, pv.position.east_m, pv.position.down_m)
                state.velocity = NedVelocity(pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s)
        except Exception as exc:
            if not stop_event.is_set():
                print(f"⚠️ position telemetry stopped: {type(exc).__name__}: {exc}")

    async def monitor_yaw() -> None:
        try:
            async for att in drone.telemetry.attitude_euler():
                if stop_event.is_set():
                    break
                state.yaw_deg = att.yaw_deg
        except Exception as exc:
            if not stop_event.is_set():
                print(f"⚠️ yaw telemetry stopped: {type(exc).__name__}: {exc}")

    async def monitor_health() -> None:
        try:
            async for health in drone.telemetry.health():
                if stop_event.is_set():
                    break
                state.health = health
        except Exception as exc:
            if not stop_event.is_set():
                print(f"⚠️ health telemetry stopped: {type(exc).__name__}: {exc}")

    async def monitor_landed() -> None:
        try:
            async for landed in drone.telemetry.landed_state():
                if stop_event.is_set():
                    break
                state.landed_state = landed
        except Exception as exc:
            if not stop_event.is_set():
                print(f"⚠️ landed telemetry stopped: {type(exc).__name__}: {exc}")

    tasks = [
        asyncio.create_task(monitor_position_velocity()),
        asyncio.create_task(monitor_yaw()),
        asyncio.create_task(monitor_health()),
        asyncio.create_task(monitor_landed()),
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
