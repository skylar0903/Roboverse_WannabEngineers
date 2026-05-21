from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
except Exception as exc:  # pragma: no cover - depends on RoboVerse VM
    Node = None
    Image = None
    _GZ_IMPORT_ERROR = exc
else:
    _GZ_IMPORT_ERROR = None


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp_s: float
    width: int
    height: int
    encoding_guess: str


class BaseImageReceiver:
    def __init__(self, topic: str, name: str = "image") -> None:
        if Node is None or Image is None:
            raise RuntimeError(
                "Could not import Gazebo Python bindings: "
                f"{_GZ_IMPORT_ERROR}. Run inside the RoboVerse Ubuntu VM."
            )
        self.topic = topic
        self.name = name
        self.node = Node()
        self.lock = threading.Lock()
        self.latest: Optional[FramePacket] = None
        self.count = 0
        ok = self.node.subscribe(Image, topic, self._callback)
        if not ok:
            raise RuntimeError(f"Failed to subscribe to {topic}")
        print(f"✅ {name} receiver subscribed: {topic}")

    def _callback(self, msg: Image) -> None:
        raise NotImplementedError

    def get_latest(self) -> Optional[FramePacket]:
        with self.lock:
            if self.latest is None:
                return None
            # copy to prevent callback thread changing data while caller uses it
            return FramePacket(
                frame=self.latest.frame.copy(),
                timestamp_s=self.latest.timestamp_s,
                width=self.latest.width,
                height=self.latest.height,
                encoding_guess=self.latest.encoding_guess,
            )

    def wait_for_frame(self, timeout_s: float = 10.0) -> Optional[FramePacket]:
        start = time.time()
        while time.time() - start < timeout_s:
            pkt = self.get_latest()
            if pkt is not None:
                return pkt
            time.sleep(0.05)
        return None


class RgbReceiver(BaseImageReceiver):
    """Gazebo RGB image receiver.

    Returns frame as BGR uint8 because OpenCV expects BGR.
    """
    def _callback(self, msg: Image) -> None:
        h, w = int(msg.height), int(msg.width)
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        pixels = h * w
        encoding = "unknown"
        try:
            if raw.size == pixels * 3:
                frame = raw.reshape((h, w, 3))
                # Gazebo sample code treats default camera as RGB; convert RGB -> BGR.
                frame = frame[:, :, ::-1].copy()
                encoding = "rgb8_to_bgr"
            elif raw.size == pixels * 4:
                rgba = raw.reshape((h, w, 4))
                frame = rgba[:, :, [2, 1, 0]].copy()  # RGBA -> BGR
                encoding = "rgba8_to_bgr"
            elif raw.size == pixels:
                gray = raw.reshape((h, w))
                frame = np.repeat(gray[:, :, None], 3, axis=2)
                encoding = "gray8_to_bgr"
            else:
                return
        except Exception:
            return
        with self.lock:
            self.latest = FramePacket(frame, time.time(), w, h, encoding)
            self.count += 1


class DepthReceiver(BaseImageReceiver):
    """Gazebo depth image receiver.

    Returns depth as float32 metres if the Gazebo stream is float32, matching the sample
    RoboVerse depth_receiver.py. Some streams may use uint16 millimetres; this class
    attempts to convert those too.
    """
    def _callback(self, msg: Image) -> None:
        h, w = int(msg.height), int(msg.width)
        pixels = h * w
        data = msg.data
        encoding = "unknown"
        frame = None
        try:
            if len(data) == pixels * 4:
                frame = np.frombuffer(data, dtype=np.float32).reshape((h, w)).copy()
                encoding = "float32_m"
            elif len(data) == pixels * 2:
                frame_u16 = np.frombuffer(data, dtype=np.uint16).reshape((h, w)).astype(np.float32)
                frame = frame_u16 / 1000.0  # likely mm -> m
                encoding = "uint16_mm_to_m"
            elif len(data) == pixels:
                frame_u8 = np.frombuffer(data, dtype=np.uint8).reshape((h, w)).astype(np.float32)
                frame = frame_u8
                encoding = "uint8_raw"
            else:
                return
        except Exception:
            return
        with self.lock:
            self.latest = FramePacket(frame, time.time(), w, h, encoding)
            self.count += 1
