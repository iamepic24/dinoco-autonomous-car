"""Every number you should need to tune lives here."""

# --- Car measurements (metres) --------------------------------------------
CAR_LENGTH_M = 0.345
CAR_WIDTH_M = 0.205
CAR_HEIGHT_M = 0.160
WHEELBASE_M = 0.232
TRACK_WIDTH_M = 0.165
TIRE_DIAMETER_M = 0.090
TIRE_WIDTH_M = 0.038
CAMERA_POKE_M = 0.05       # camera tip past the front of the car
# Estimated from the side photo. Measure: LiDAR centre to the car's front.
LIDAR_TO_FRONT_M = 0.08
CRUISE_SPEED_MPS = 0.25    # placeholder: time the car over 2 m at BASE_THROTTLE
COAST_M = 0.15             # placeholder: distance rolled after throttle goes to 0

# --- Ports ---------------------------------------------------------------
# Use the stable paths printed by `python check.py ports`. /dev/ttyUSB0 and
# /dev/ttyUSB1 can swap between boots if the ESP32 and LiDAR adapters match.
ESP_PORT = "/dev/serial/by-path/platform-xhci-hcd.0-usb-0:2:1.0"
LIDAR_PORT = "/dev/serial/by-path/platform-xhci-hcd.1-usb-0:2:1.0-port0"
ESP_BAUD = 115200
LIDAR_BAUD = 460800

# --- Camera --------------------------------------------------------------
CAMERA_INDEX = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30
PROC_WIDTH = 426    # frames are shrunk to this before lane detection
PROC_HEIGHT = 240

# --- Lane detection ------------------------------------------------------
LINE_IS_BRIGHT = True      # True: light tape on a dark floor. False: dark tape.
LINE_COLOR = "red"         # "red" finds red tape. "" uses LINE_IS_BRIGHT instead.
RED_MARGIN = 50            # how much redder than green and blue tape must be
LINE_CONTRAST = 28         # how far a line must differ from nearby floor (0-255)
ROW_NEAR = 0.50            # lowest scan row, as a fraction of image height
ROW_FAR = 0.22             # highest scan row. Keep it below the horizon.
NUM_ROWS = 12
MIN_LINE_PX = 2            # tape width limits in pixels, used to reject noise
MAX_LINE_PX = 45
LANE_WIDTH_NEAR = 0.82     # starting lane width at ROW_NEAR, as a fraction of image
LANE_WIDTH_FAR = 0.39      # width, and at ROW_FAR. Over 1.0 means wider than the view.
                           # Replaced by a measurement once both lines are seen.
MIN_WIDTH_ROWS = 3         # rows showing both lines needed to measure the width
CENTER_TRIM = 0.0          # use if the camera is mounted off-centre (-1 to 1)
FAR_WEIGHT = 0.2           # steering weight of the farthest row, nearest row is 1
TRACK_MEMORY = 15          # frames a row remembers its lines after losing them
MIN_CONFIDENCE = 0.35      # fraction of rows that must find the lane

# --- Steering and speed --------------------------------------------------
KP = 1.3
KD = 0.12
STEER_TRIM = 0.0           # adjust until the car rolls straight at steer=0
STEER_LIMIT = 0.9
BASE_THROTTLE = 0.30       # placeholder: find yours with `check.py throttle`
MIN_THROTTLE = 0.20        # lowest value that still keeps the car moving
TURN_SLOWDOWN = 0.5        # share of the throttle above MIN removed at full lock

# --- Obstacle stop -------------------------------------------------------
OBSTACLE_GAP_M = 0.20      # wanted gap between the camera tip and the obstacle
# LiDAR reading that triggers the stop. Allows for rolling on after throttle cut.
STOP_DISTANCE_M = LIDAR_TO_FRONT_M + CAMERA_POKE_M + OBSTACLE_GAP_M + COAST_M
CLEAR_MARGIN_M = 0.10      # path must be this much clearer before resuming
CLEAR_HOLD_S = 2.0         # path must stay clear this long before resuming
CORRIDOR_HALF_WIDTH_M = 0.15
LIDAR_FORWARD_DEG = 0.0    # LiDAR angle that points forward: see `check.py lidar`
LIDAR_MIN_RANGE_M = 0.12   # ignores the car's own body and wires
MIN_OBSTACLE_POINTS = 3    # points required, so one noisy return can't stop you
BRAKE_THROTTLE = 0.0       # leave at 0 until you know how your ESC treats reverse
BRAKE_TIME_S = 0.25

# --- STOP sign ------------------------------------------------------------
SIGN_ENABLED = True
SIGN_GAP_M = 0.08          # wanted gap between the front of the car and the sign
SIGN_HEIGHT_M = 0.12       # printed red sign height. Measure yours.
CAMERA_F_PX = 640          # focal length at 1280x720: set with `check.py sign 0.40`
SIGN_RED_MARGIN = 45       # how much redder than green and blue the sign must be
SIGN_MIN_PX = 44           # smallest sign height to try reading, full-size pixels
SIGN_CONFIRM_READS = 2     # readings of the word needed to believe it
SIGN_CONFIRM_WINDOW_S = 1.5
SIGN_HOLD_S = 3.0          # full stop time
SIGN_COOLDOWN_S = 8.0      # ignore the same sign this long after moving off

# --- Safety timeouts -----------------------------------------------------
LOOP_HZ = 25               # must stay well above the ESP32's 300 ms watchdog
LANE_LOST_S = 0.4          # no lane for this long: stop
CAMERA_STALE_S = 0.3
LIDAR_STALE_S = 0.5
