from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import config


@dataclass
class DetectionRecord:
    class_name: str
    confidence: float
    north_m: float
    east_m: float
    down_m: float
    time_s: float
    source: str
    image_path: str


class DetectionMemory:
    def __init__(self, log_dir: str = config.LOG_DIR) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.records: List[DetectionRecord] = []
        self.csv_path = os.path.join(log_dir, "detections.csv")
        self._csv = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._csv)
        self._writer.writerow(["time_s", "class_name", "confidence", "north_m", "east_m", "down_m", "source", "image_path"])
        self._csv.flush()

    def close(self) -> None:
        try:
            self._csv.close()
        except Exception:
            pass

    def is_duplicate(self, class_name: str, north_m: float, east_m: float) -> bool:
        for r in self.records:
            if r.class_name != class_name:
                continue
            if math.hypot(r.north_m - north_m, r.east_m - east_m) <= config.DETECTION_DEDUP_RADIUS_M:
                return True
        return False

    def add_if_new(
        self,
        class_name: str,
        confidence: float,
        north_m: float,
        east_m: float,
        down_m: float,
        source: str,
        image_path: Optional[str] = None,
    ) -> bool:
        if self.is_duplicate(class_name, north_m, east_m):
            return False
        rec = DetectionRecord(class_name, confidence, north_m, east_m, down_m, time.time(), source, image_path or "")
        self.records.append(rec)
        self._writer.writerow([rec.time_s, rec.class_name, rec.confidence, rec.north_m, rec.east_m, rec.down_m, rec.source, rec.image_path])
        self._csv.flush()
        print(f"🎯 NEW {class_name.upper()} detected | conf={confidence:.2f} | N={north_m:.1f} E={east_m:.1f} | total={len(self.records)}")
        return True

    def summary(self) -> str:
        yellow = sum(1 for r in self.records if "yellow" in r.class_name.lower())
        red = sum(1 for r in self.records if "red" in r.class_name.lower())
        return f"yellow={yellow}, red={red}, total={len(self.records)}"
