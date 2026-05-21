from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class DepthSummary:
    left_m: float
    center_m: float
    right_m: float
    min_m: float
    state: str
    recommended: str


class DepthAnalyzer:
    def __init__(self) -> None:
        self.safe_m = config.DEPTH_SAFE_M
        self.danger_m = config.DEPTH_DANGER_M
        self.percentile = config.DEPTH_PERCENTILE

    def _clean(self, depth: np.ndarray) -> np.ndarray:
        d = depth.astype(np.float32, copy=False)
        d = np.where(np.isfinite(d), d, np.nan)
        d = np.where((d >= config.DEPTH_MIN_VALID_M) & (d <= config.DEPTH_MAX_VALID_M), d, np.nan)
        return d

    def _region_percentile(self, region: np.ndarray) -> float:
        valid = region[np.isfinite(region)]
        if valid.size < 20:
            return float("inf")
        return float(np.nanpercentile(valid, self.percentile))

    def summarize(self, depth: np.ndarray) -> DepthSummary:
        d = self._clean(depth)
        h, w = d.shape[:2]

        # Use middle vertical band to reduce floor/ceiling noise.
        y1 = int(h * 0.25)
        y2 = int(h * 0.80)
        mid = d[y1:y2, :]

        left = self._region_percentile(mid[:, : w // 3])
        center = self._region_percentile(mid[:, w // 3 : 2 * w // 3])
        right = self._region_percentile(mid[:, 2 * w // 3 :])
        finite = mid[np.isfinite(mid)]
        min_m = float(np.nanmin(finite)) if finite.size else float("inf")

        if center < self.danger_m and left < self.safe_m and right < self.safe_m:
            state = "BLOCKED"
            recommended = "BACK_OR_YAW_SCAN"
        elif center < self.danger_m:
            state = "DANGER_FRONT"
            recommended = "SHIFT_LEFT" if left >= right else "SHIFT_RIGHT"
        elif center < self.safe_m:
            state = "CAUTION_FRONT"
            recommended = "SLOW_OR_SHIFT_LEFT" if left >= right else "SLOW_OR_SHIFT_RIGHT"
        else:
            state = "CLEAR"
            recommended = "FORWARD"

        return DepthSummary(left, center, right, min_m, state, recommended)


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    source: str
    center_px: Tuple[int, int]
    area_px: float


class ColorBarrelDetector:
    """Fast fallback detector for red/yellow barrels.

    This is not as semantically strong as YOLO, but it is fast and avoids losing the
    whole mission if model setup fails on competition day.
    """
    def __init__(self) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV/cv2 is required for ColorBarrelDetector")

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Red wraps around HSV hue boundary.
        red1 = cv2.inRange(hsv, np.array([0, 80, 70]), np.array([12, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([168, 80, 70]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)

        yellow_mask = cv2.inRange(hsv, np.array([18, 80, 80]), np.array([38, 255, 255]))

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

        detections: List[Detection] = []
        detections.extend(self._mask_to_detections(red_mask, "red_barrel"))
        detections.extend(self._mask_to_detections(yellow_mask, "yellow_barrel"))
        return detections

    def _mask_to_detections(self, mask: np.ndarray, class_name: str) -> List[Detection]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        h_img, w_img = mask.shape[:2]
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < config.MIN_BLOB_AREA_PX:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h < config.MIN_BLOB_HEIGHT_PX:
                continue
            # Barrels/canisters can be vertical-ish but perspective varies; keep filter loose.
            aspect = w / max(h, 1)
            if aspect > 3.2 or aspect < 0.15:
                continue
            # ignore very top/bottom UI/noise strips
            if y < 2 or y + h > h_img - 2:
                continue
            conf = min(0.95, 0.45 + area / max(w_img * h_img * 0.08, 1))
            out.append(
                Detection(
                    class_name=class_name,
                    confidence=float(conf),
                    bbox_xyxy=(int(x), int(y), int(x + w), int(y + h)),
                    source="color",
                    center_px=(int(x + w / 2), int(y + h / 2)),
                    area_px=area,
                )
            )
        out.sort(key=lambda d: d.area_px, reverse=True)
        return out


class OptionalYoloDetector:
    def __init__(self, model_path: str, confidence: float) -> None:
        self.available = False
        self.model = None
        self.names = {}
        self.confidence = confidence
        if not model_path:
            return
        if not os.path.exists(model_path):
            print(f"⚠️ YOLO model path not found: {model_path}. Colour fallback only.")
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.names = self.model.names
            self.available = True
            print(f"✅ YOLO model loaded: {model_path}")
        except Exception as exc:
            print(f"⚠️ Could not load YOLO model: {exc}. Colour fallback only.")

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if not self.available or self.model is None:
            return []
        detections: List[Detection] = []
        try:
            results = self.model(frame_bgr, verbose=False, conf=self.confidence)
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy().tolist()]
                    conf = float(box.conf[0].cpu().item())
                    cls_id = int(box.cls[0].cpu().item())
                    name = str(self.names.get(cls_id, cls_id))
                    # Normalize common class names to competition names when possible.
                    low = name.lower()
                    if "yellow" in low:
                        cname = "yellow_barrel"
                    elif "red" in low:
                        cname = "red_barrel"
                    else:
                        cname = name
                    detections.append(
                        Detection(
                            class_name=cname,
                            confidence=conf,
                            bbox_xyxy=(x1, y1, x2, y2),
                            source="yolo",
                            center_px=(int((x1+x2)/2), int((y1+y2)/2)),
                            area_px=float(max(0, x2-x1) * max(0, y2-y1)),
                        )
                    )
        except Exception as exc:
            print(f"⚠️ YOLO inference warning: {exc}")
        return detections


class CombinedDetector:
    def __init__(self) -> None:
        self.color = ColorBarrelDetector()
        self.yolo = OptionalYoloDetector(config.YOLO_MODEL_PATH, config.YOLO_CONFIDENCE)
        os.makedirs(config.DETECTION_DIR, exist_ok=True)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        detections = []
        detections.extend(self.yolo.detect(frame_bgr))
        detections.extend(self.color.detect(frame_bgr))
        return self._nms_simple(detections)

    def _nms_simple(self, detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
        # Simple class-wise non-max suppression so YOLO+colour duplicate less.
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for det in detections:
            same_overlap = False
            for k in kept:
                if k.class_name == det.class_name and self._iou(k.bbox_xyxy, det.bbox_xyxy) > iou_threshold:
                    same_overlap = True
                    break
            if not same_overlap:
                kept.append(det)
        return kept

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / max(area_a + area_b - inter, 1)

    def annotate_and_save(self, frame_bgr: np.ndarray, detections: List[Detection], prefix: str = "det") -> str | None:
        if cv2 is None or not config.SAVE_DETECTION_IMAGES or not detections:
            return None
        img = frame_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, f"{d.class_name} {d.confidence:.2f} {d.source}", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        path = os.path.join(config.DETECTION_DIR, f"{prefix}_{int(time.time()*1000)}.jpg")
        cv2.imwrite(path, img)
        return path
