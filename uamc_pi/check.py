"""Bring-up checks. Run these one at a time, before run.py.

    python check.py ports              list stable USB port paths
    python check.py steer              wheels off the ground
    python check.py throttle           wheels off the ground; finds the crawl value
    python check.py camera             saves check_camera.jpg and check_mask.jpg
    python check.py lidar              prints the nearest object and its angle
"""
import glob
import sys
import time

import cv2

import config as cfg


def hold(esp, steer, throttle, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        esp.send(steer, throttle)
        time.sleep(0.04)


def ports():
    paths = sorted(glob.glob("/dev/serial/by-path/*") + glob.glob("/dev/serial/by-id/*"))
    print("\n".join(paths) or "No USB serial devices found")


def steer():
    from esp_link import EspLink
    esp = EspLink()
    try:
        for value in (0, -0.5, 0.5, -1, 1, 0):
            print(f"steer {value:+}")
            hold(esp, value, 0, 1.5)
    finally:
        esp.close()


def throttle():
    from esp_link import EspLink
    limit = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    esp = EspLink()
    try:
        value = 0.04
        while value <= limit:
            print(f"throttle {value:.2f}")
            hold(esp, 0, value, 1.5)
            value += 0.02
    finally:
        esp.close()


def camera():
    from lane import LaneDetector, draw_overlay
    from run import Camera
    cam = Camera()
    time.sleep(2)     # gives auto-exposure time to settle
    frame, _ = cam.latest()
    cam.close()
    if frame is None:
        raise SystemExit("No frame from the camera")
    detector = LaneDetector()
    for _ in range(15):
        result, mask = detector.detect(frame)
    print(f"error={result.error:+.2f} confidence={result.confidence:.2f}")
    widths = (detector.lane_w / cfg.PROC_WIDTH).round(2)
    print(f"lane width measured: {detector.fitted}")
    print(f"LANE_WIDTH_NEAR = {widths[0]}  LANE_WIDTH_FAR = {widths[-1]}")
    if not detector.fitted:
        print("Centre the car in the lane and tilt the camera until both lines show.")
    cv2.imwrite("check_camera.jpg", draw_overlay(frame, result))
    cv2.imwrite("check_mask.jpg", mask)


def lidar():
    from lidar_c1 import Lidar
    sensor = Lidar()
    try:
        for _ in range(100):
            time.sleep(0.2)
            points = [p for p in sensor.last_scan if p[1] >= cfg.LIDAR_MIN_RANGE_M]
            if points:
                angle, dist = min(points, key=lambda p: p[1])
                print(f"{len(points)} points, nearest {dist:.2f} m at {angle:.0f} deg, "
                      f"in path: {sensor.obstacle_m}")
    finally:
        sensor.close()


if __name__ == "__main__":
    actions = dict(ports=ports, steer=steer, throttle=throttle, camera=camera, lidar=lidar)
    if len(sys.argv) < 2 or sys.argv[1] not in actions:
        raise SystemExit(__doc__)
    actions[sys.argv[1]]()
