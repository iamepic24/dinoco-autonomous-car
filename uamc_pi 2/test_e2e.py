"""Runs the real run.py with a fake camera, ESP32 and LiDAR on virtual serial ports."""
import os
import pty
import re
import sys
import threading
import time
from pathlib import Path

import cv2

import config as cfg
import run
from test_sim import RADIUS, node, render

T0 = time.monotonic()
FRAME = render(1.0, -RADIUS + 0.03, 0.0, 0.6)
OBSTACLE_AT, LIDAR_DIES_AT = 5.0, 12.0
commands = []          # (time, steer, throttle)


class FakeCapture:
    def __init__(self, *_):
        pass

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        time.sleep(1 / 30)
        return True, FRAME

    def release(self):
        pass


def fake_esp(fd):
    time.sleep(0.3)
    os.write(fd, b"READY\r\n")
    data = b""
    while True:
        try:
            data += os.read(fd, 1024)
        except OSError:
            return
        while b"\n" in data:
            line, data = data.split(b"\n", 1)
            m = re.match(rb"S,(-?[\d.]+),(-?[\d.]+)", line)
            if m:
                commands.append((time.monotonic() - T0, float(m[1]), float(m[2])))


def fake_lidar(fd):
    data = b""
    while b"\xa5\x20" not in data:
        data += os.read(fd, 16)
    os.write(fd, b"\xa5\x5a\x05\x00\x00\x40\x81")
    angle, first = 0.0, True
    while time.monotonic() - T0 < LIDAR_DIES_AT:
        burst = b""
        for _ in range(50):
            blocked = time.monotonic() - T0 > OBSTACLE_AT and (angle < 8 or angle > 352)
            burst += node(first, 40, angle, 0.40 if blocked else 3.0)
            first = False
            angle += 0.72
            if angle >= 360:
                angle, first = 0.0, True
        os.write(fd, burst)
        time.sleep(0.01)


def main():
    cv2.VideoCapture = FakeCapture
    esp_m, esp_s = pty.openpty()
    lid_m, lid_s = pty.openpty()
    cfg.ESP_PORT, cfg.LIDAR_PORT = os.ttyname(esp_s), os.ttyname(lid_s)
    threading.Thread(target=fake_esp, args=(esp_m,), daemon=True).start()
    threading.Thread(target=fake_lidar, args=(lid_m,), daemon=True).start()
    sys.argv = ["run.py", "--delay", "1", "--time", "13"]
    run.main()
    time.sleep(0.2)

    def phase(a, b):
        return [c for c in commands if a <= c[0] < b]

    moving, stopped, dead = phase(2.5, 4.5), phase(5.5, 11.5), phase(12.7, 14)
    gaps = [b[0] - a[0] for a, b in zip(commands, commands[1:])]
    print(f"{len(commands)} commands, {len(commands)/commands[-1][0]:.1f}/s, "
          f"largest gap {max(gaps)*1000:.0f} ms")
    assert moving and all(c[2] > 0 for c in moving), moving[:3]
    assert stopped and all(c[2] == 0 for c in stopped), stopped[:3]
    assert dead and all(c[2] == 0 for c in dead), dead[:3]
    assert max(gaps) < 0.2 and commands[-1][1:] == (0.0, 0.0)
    log = sorted(Path("runs").glob("*/log.csv"))[-1]
    states = {l.split(",")[1] for l in log.read_text().splitlines()[1:]}
    assert {"WAIT", "DRIVE", "OBSTACLE", "FAULT"} <= states, states
    print("PASS full loop: drives, stops for obstacle, stops on LiDAR failure, "
          "ends at neutral")


if __name__ == "__main__":
    main()
