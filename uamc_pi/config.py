"""Every number you should need to tune lives here."""

# --- Ports ---------------------------------------------------------------
# Use the stable paths printed by `python check.py ports`. /dev/ttyUSB0 and
# /dev/ttyUSB1 can swap between boots if the ESP32 and LiDAR adapters match.
ESP_PORT = "/dev/ttyUSB0"
LIDAR_PORT = "/dev/ttyUSB1"
ESP_BAUD = 115200
LIDAR_BAUD = 460800

# --- Camera --------------------------------------------------------------
CAMERA_INDEX = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 30
PROC_WIDTH = 320    # frames are shrunk to this before lane detection
PROC_HEIGHT = 240

# --- Lane detection ------------------------------------------------------
LINE_IS_BRIGHT = True      # True: light tape on a dark floor. False: dark tape.
LINE_CONTRAST = 28         # how far a line must differ from nearby floor (0-255)
ROW_NEAR = 0.92            # lowest scan row, as a fraction of image height
ROW_FAR = 0.55             # highest scan row. Keep it below the horizon.
NUM_ROWS = 12
MIN_LINE_PX = 2            # tape width limits in pixels, used to reject noise
MAX_LINE_PX = 45
LANE_WIDTH_NEAR = 1.40     # starting lane width at ROW_NEAR, as a fraction of image
LANE_WIDTH_FAR = 0.70      # width, and at ROW_FAR. Over 1.0 means wider than the view.
                           # Replaced by a measurement once both lines are seen.
MIN_WIDTH_ROWS = 3         # rows showing both lines needed to measure the width
CENTER_TRIM = 0.0          # use if the camera is mounted off-centre (-1 to 1)
MIN_CONFIDENCE = 0.35      # fraction of rows that must find the lane

# --- Steering and speed --------------------------------------------------
KP = 1.3
KD = 0.12
STEER_TRIM = 0.0           # adjust until the car rolls straight at steer=0
STEER_LIMIT = 0.9
BASE_THROTTLE = 0.15       # placeholder: find yours with `check.py throttle`
MIN_THROTTLE = 0.12        # lowest value that still keeps the car moving
TURN_SLOWDOWN = 0.5        # share of the throttle above MIN removed at full lock

# --- Obstacle stop -------------------------------------------------------
STOP_DISTANCE_M = 0.50     # measured forward from the LiDAR, not the bumper
CLEAR_MARGIN_M = 0.10      # path must be this much clearer before resuming
CLEAR_HOLD_S = 2.0         # path must stay clear this long before resuming
CORRIDOR_HALF_WIDTH_M = 0.15
LIDAR_FORWARD_DEG = 0.0    # LiDAR angle that points forward: see `check.py lidar`
LIDAR_MIN_RANGE_M = 0.12   # ignores the car's own body and wires
MIN_OBSTACLE_POINTS = 3    # points required, so one noisy return can't stop you
BRAKE_THROTTLE = 0.0       # leave at 0 until you know how your ESC treats reverse
BRAKE_TIME_S = 0.25

# --- Safety timeouts -----------------------------------------------------
LOOP_HZ = 25               # must stay well above the ESP32's 300 ms watchdog
LANE_LOST_S = 0.4          # no lane for this long: stop
CAMERA_STALE_S = 0.3
LIDAR_STALE_S = 0.5
