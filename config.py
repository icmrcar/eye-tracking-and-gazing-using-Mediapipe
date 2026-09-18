"""
config.py

Single source of truth for all tunable constants used across the system:
dwell times, EAR thresholds, deadzones, calibration defaults, serial settings,
and speed ramp parameters.
"""

# --- Camera / Frame settings ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# --- Eye feature smoothing ---
SMOOTHING_WINDOW_FRAMES = 4

# --- Gaze direction thresholds (calibrated per-user at runtime) ---
# NOTE: DOWN was removed as a calibrated/classified direction -- looking
# down partially occludes the iris with the eyelid, which fights with
# blink detection. STOP is triggered by long-blink / double-blink instead
# (see blink_detector.py). Only CENTER, LEFT, RIGHT, UP remain.
HORIZONTAL_RATIO_LEFT_THRESHOLD = None    # set during calibration.py
HORIZONTAL_RATIO_RIGHT_THRESHOLD = None   # set during calibration.py
VERTICAL_RATIO_UP_THRESHOLD = None        # set during calibration.py

# Sign convention flags -- since raw ratio direction depends on landmark
# indexing/camera orientation, calibration.py determines these empirically.
HORIZONTAL_LEFT_IS_LOWER = None    # set during calibration.py
VERTICAL_UP_IS_LOWER = None        # set during calibration.py

# --- Dwell times (ms) for gaze_classifier.py's short confirmation dwell ---
DWELL_TIME_UP_MS = 1000
DWELL_TIME_LEFT_MS = 500
DWELL_TIME_RIGHT_MS = 500

# --- Dwell flicker tolerance ---
DWELL_FLICKER_TOLERANCE_FRAMES = 2

# --- Blink / eye closure thresholds (ms) ---
# MIN: closures shorter than this are treated as landmark/camera noise, not
# a real blink -- ignored entirely (not counted, not eligible for
# double-blink pairing). Roughly 2-3 frames at 30fps.
MIN_BLINK_DURATION_MS = 80
NATURAL_BLINK_MAX_MS = 400
DELIBERATE_BLINK_MIN_MS = 800
EAR_CLOSED_THRESHOLD = 0.2  # EAR value below which eye is considered "closed"

# STOP is triggered by gaze via either a long blink (DELIBERATE_BLINK_MIN_MS
# above) or two natural blinks in quick succession.
DOUBLE_BLINK_WINDOW_MS = 600

# --- Joystick settings ---
JOYSTICK_DEADZONE = 0.15          # normalized axis value, 0.0-1.0
JOYSTICK_NEUTRAL_HOLD_MS = 1000
JOYSTICK_EMERGENCY_BUTTON_INDEX = 0

# --- Fail-safe thresholds ---
FACE_LOST_FRAME_THRESHOLD = 10     # consecutive frames with no face -> STOP
SERIAL_TIMEOUT_MS = 500

# --- Motor / Speed ramp settings ---
FORWARD_PWM = 150
TURN_FAST_PWM = 150
TURN_SLOW_PWM = 80
RAMP_DURATION_MS = 400
RAMP_STEP_MS = 50

# --- Serial communication ---
SERIAL_PORT = "/dev/ttyUSB0"   # update per platform, e.g. "COM3" on Windows
SERIAL_BAUD_RATE = 115200

# --- Data logging ---
CSV_LOG_PATH = "logs/session_log.csv"

# --- Debugging ---
DEBUG = True                  # console debug logging for face/eye/gaze events
FPS_WARNING_THRESHOLD = 15    # log a warning if FPS drops below this

# --- Head pose stability (calibration.py) ---
# Maximum allowed yaw/pitch deviation (degrees) from the CENTER-step head
# pose baseline during LEFT/RIGHT/UP calibration capture. Calibration
# capture actively PAUSES while the head is outside this tolerance.
HEAD_POSE_TOLERANCE_DEG = 12.0

# --- Hold-to-activate motion state machine (main_logic1.py / main_logic2.py) ---
# Fixed 3-second hold to activate any movement from a stop (or, in Logic 2,
# to turn directly from FORWARD). Same fixed time for UP and LEFT/RIGHT.
ACTIVATION_HOLD_MS = 3000
# Brief eye flickers under this duration are tolerated without resetting
# an activation hold in progress, or without ending an active turn.
MOTION_FLICKER_TOLERANCE_MS = 150
