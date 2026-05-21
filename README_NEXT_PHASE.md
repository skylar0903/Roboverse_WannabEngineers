# RoboVerse next phase test package

This package is the **compressed Phase 2 + Phase 3 test bridge** after your Phase 1.1 pass.

It is designed around the competition objective:

- search a virtual space port for yellow/red fuel barrels,
- use object detection and obstacle avoidance without GNSS,
- push for all detections within 5 minutes.

## What to run first

Start PX4/Gazebo first and choose the RoboVerse `x500_vision` drone. If PX4 needs it, enter:

```bash
commander set_ekf_origin 47.397742 8.545594 488.0
```

Then in a **new terminal**:

```bash
cd ~/Desktop/roboverse_phase2_next
python3 00_list_gz_topics.py
```

This lists camera/depth topics. If AUTO fails later, copy the exact topic names into `config.py`.

## Run order

### 1) Depth test only, no flying

```bash
python3 01_depth_test.py
```

Expected output:

```text
L=... m | C=... m | R=... m | CLEAR -> FORWARD
```

Move/face the drone toward walls/obstacles in simulator view and check whether `C` becomes smaller.

### 2) RGB/barrel detection only, no flying

```bash
python3 02_rgb_detection_test.py
```

Expected output when red/yellow objects are visible:

```text
red_barrel / yellow_barrel ... saved: detections/...
```

This uses a colour detector fallback immediately. If you have a YOLO model, put it in `models/best.pt` or `models/best.onnx`, then edit `YOLO_MODEL_PATH` in `config.py`.

### 3) Safe flying hover scan

Run only after step 2 receives camera frames:

```bash
python3 03_hover_scan_detect.py
```

It will:

1. connect,
2. take off,
3. enter Offboard,
4. hover and yaw scan,
5. log detections,
6. land.

### 4) Experimental 5-minute barrel hunter

Run only after steps 1–3 work:

```bash
python3 04_barrel_hunter_v0_experimental.py
```

It will:

- take off,
- fly a lawnmower pattern,
- detect red/yellow continuously,
- use depth left/centre/right as a simple obstacle guard,
- log detections in `logs/detections.csv`,
- save annotated images in `detections/`.

## Files you may need to edit

Open `config.py`.

Most likely edits:

```python
RGB_TOPIC = "AUTO"
DEPTH_TOPIC = "AUTO"
YOLO_MODEL_PATH = ""
SEARCH_N_MIN = -16.0
SEARCH_N_MAX = 16.0
SEARCH_E_MIN = -16.0
SEARCH_E_MAX = 16.0
LANE_SPACING_M = 4.0
```

If the drone starts near a corner instead of centre, the search bounds may need changing.

## What the logs mean

- `logs/detections.csv` = every new red/yellow detection with drone N/E/D position.
- `detections/*.jpg` = screenshots with detected object boxes.

## Important safety notes

- This is autonomous code. Start with `01` and `02`; do not jump straight to `04`.
- `04_barrel_hunter_v0_experimental.py` is intentionally simple and fast, not full SLAM.
- If the drone gets too close to obstacles, increase `DEPTH_DANGER_M` or reduce `LOOKAHEAD_M` in `config.py`.
- If the detector misses barrels, reduce `MIN_BLOB_AREA_PX` or add a YOLO model.
