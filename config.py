"""RoboVerse Phase 2 / 1.5 configuration.

Edit this file first, not the main scripts.
All coordinates are LOCAL NED metres relative to takeoff/start:
- North positive = +N
- East positive = +E
- Down negative = altitude above start
"""

# -----------------------------------------------------------------------------
# PX4 / MAVSDK
# -----------------------------------------------------------------------------
SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"
TAKEOFF_ALTITUDE_M = 2.0
CRUISE_DOWN_M = -2.0   # NED: -2 means 2 m above takeoff point
SETPOINT_RATE_HZ = 5.0
HEALTH_TIMEOUT_S = 45.0
TAKEOFF_TIMEOUT_S = 12.0
LAND_TIMEOUT_S = 60.0
LAND_ALTITUDE_ACCEPT_M = 0.18
WAYPOINT_ACCEPT_RADIUS_M = 0.8
ALTITUDE_ACCEPT_RADIUS_M = 0.9

# -----------------------------------------------------------------------------
# Gazebo topics
# Use "AUTO" first. If AUTO picks wrong, run 00_list_gz_topics.py and paste the
# exact topic name here.
# -----------------------------------------------------------------------------
RGB_TOPIC = "AUTO"
DEPTH_TOPIC = "AUTO"

# Known topic from the sample code. The auto-selector will prefer this if present.
PREFERRED_RGB_TOPICS = [
    "/world/roboverse/model/x500_mono_cam_0/link/camera_link/sensor/camera/image",
]

# Depth topic names vary by world/drone. These are guesses; auto-detection uses gz topic -l.
PREFERRED_DEPTH_TOPICS = [
    "/depth_camera",
    "/world/roboverse/model/x500_depth_0/link/camera_link/sensor/depth_camera/depth_image",
    "/world/roboverse/model/x500_vision_0/link/camera_link/sensor/depth_camera/depth_image",
]

# -----------------------------------------------------------------------------
# Perception
# -----------------------------------------------------------------------------
ENABLE_DISPLAY = True       # set False if OpenCV windows lag/crash in VM
SAVE_DETECTION_IMAGES = True
DETECTION_DIR = "detections"
LOG_DIR = "logs"

# If you have a trained model, put it in models/best.pt or models/best.onnx and set path.
# Leave as "" to use only colour detection fallback.
YOLO_MODEL_PATH = ""        # e.g. "models/best.pt" or "models/best.onnx"
YOLO_CONFIDENCE = 0.45

# Colour detector thresholds, tuned for obvious red/yellow barrels in RGB camera.
MIN_BLOB_AREA_PX = 350
MIN_BLOB_HEIGHT_PX = 18
DETECTION_COOLDOWN_S = 0.35
DETECTION_DEDUP_RADIUS_M = 1.8

# -----------------------------------------------------------------------------
# Depth / obstacle safety
# -----------------------------------------------------------------------------
DEPTH_SAFE_M = 3.0          # clear enough to fly forward
DEPTH_DANGER_M = 1.25       # too close: sidestep/slow immediately
DEPTH_PERCENTILE = 20       # robust: closer obstacles matter more than mean
DEPTH_MAX_VALID_M = 20.0
DEPTH_MIN_VALID_M = 0.05

# -----------------------------------------------------------------------------
# Fast barrel-hunter v0 search pattern
# These are conservative bounds assuming start near centre of a ~40 m x 40 m area.
# If the map start is not centre, tune these after seeing movement.
# -----------------------------------------------------------------------------
MISSION_TIME_LIMIT_S = 300.0      # 5-minute scoring push
ABSOLUTE_SAFETY_LIMIT_S = 570.0   # land before 10-minute hard limit
SEARCH_N_MIN = -16.0
SEARCH_N_MAX = 16.0
SEARCH_E_MIN = -16.0
SEARCH_E_MAX = 16.0
LANE_SPACING_M = 4.0
LOOKAHEAD_M = 1.4
CONTROL_LOOP_HZ = 5.0
MAX_STEP_TARGET_M = 1.8

# Yaw behaviour while searching.
# "PATH" = face direction of travel. Safer for obstacle avoidance.
# "SLOW_SWEEP" = slow side-to-side yaw modulation while moving; may improve detection but can blur.
YAW_MODE = "PATH"
YAW_SWEEP_DEG = 18.0
YAW_SWEEP_PERIOD_S = 8.0
