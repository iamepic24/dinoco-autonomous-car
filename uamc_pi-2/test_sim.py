"""Hardware-free tests. Drives a simulated car on a rendered stadium track."""
import math
import os
import pty
import threading
import time

import cv2
import numpy as np

import config as cfg
from lane import LaneDetector, LaneResult
from lidar_c1 import nearest_in_corridor, parse_nodes
from run import Controller

cfg.LINE_COLOR = ""     # the simulated track is grey
W, H = cfg.PROC_WIDTH, cfg.PROC_HEIGHT
CAM_H, PITCH, FOCAL = 0.15, math.radians(28.8), 213.0   # measured from the car video
TAPE_W = 0.04
STRAIGHT, RADIUS = 2.0, 1.2
WHEELBASE, MAX_STEER = cfg.WHEELBASE_M, math.radians(25)   # steering angle assumed
DT = 1 / cfg.LOOP_HZ

# Ground position of every pixel, in car coordinates (x forward, y left).
_v, _u = np.mgrid[0:H, 0:W].astype(float)
_yr, _xr = (_v - H / 2) / FOCAL, (W / 2 - _u) / FOCAL
_den = math.sin(PITCH) + _yr * math.cos(PITCH)
with np.errstate(divide="ignore", invalid="ignore"):
    GROUND_F = CAM_H * (math.cos(PITCH) - _yr * math.sin(PITCH)) / _den
    GROUND_L = _xr * (GROUND_F * math.cos(PITCH) + CAM_H * math.sin(PITCH))
ON_FLOOR = (_den > 0.01) & (GROUND_F > 0) & (GROUND_F < 3.0)


def centre_offset(x, y):
    """Signed distance from the stadium centreline. Positive is outside."""
    straight = np.abs(y) - RADIUS
    right = np.hypot(x - STRAIGHT, y) - RADIUS
    left = np.hypot(x, y) - RADIUS
    return np.where(x > STRAIGHT, right, np.where(x < 0, left, straight))


def render(x, y, yaw, lane_w, floor=80, tape=205, shadow=False, rng=None, offset=None):
    wx = x + GROUND_F * math.cos(yaw) - GROUND_L * math.sin(yaw)
    wy = y + GROUND_F * math.sin(yaw) + GROUND_L * math.cos(yaw)
    d = np.abs(np.abs((offset or centre_offset)(wx, wy)) - lane_w / 2)
    img = np.full((H, W), float(floor))
    img[ON_FLOOR & (d < TAPE_W / 2)] = tape
    img[~ON_FLOOR] = 150                                 # walls above the horizon
    if shadow:                                           # soft shadow band
        img *= 1 - 0.45 * np.exp(-((_u - W * 0.6 + _v * 0.3) / 35) ** 2)
    if rng is not None:
        img += rng.normal(0, 3, img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def drive(seconds=45, speed=0.6, lane_w=0.6, shadow=False, obstacle_after=None,
          latency_steps=3):
    x, y, yaw = 0.2, -RADIUS + 0.06, 0.08    # starts off-centre and slightly crooked
    detector, controller = LaneDetector(), Controller()
    rng = np.random.default_rng(1)
    queue = [0.0] * latency_steps
    offsets, states, stop_gap, travelled = [], [], None, 0.0
    obstacle_at = None
    v = 0.0
    for i in range(int(seconds / DT)):
        now = i * DT
        frame = render(x, y, yaw, lane_w, shadow=shadow, rng=rng)
        lane, _ = detector.detect(frame)
        distance = None
        if obstacle_after is not None and now >= obstacle_after:
            if obstacle_at is None:
                obstacle_at = travelled + 1.0
            distance = obstacle_at - travelled
        steer, throttle, state = controller.step(lane, distance, True, now)
        queue.append(steer)
        applied = queue.pop(0)
        v = speed if throttle > 0 else max(0.0, v - 2.0 * DT)
        # Positive steer turns right, matching the firmware, so yaw decreases.
        yaw -= v / WHEELBASE * math.tan(applied * MAX_STEER) * DT
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        travelled += v * DT
        offsets.append(abs(float(centre_offset(np.array(x), np.array(y)))))
        states.append(state)
        if state == "OBSTACLE":
            stop_gap = distance
            if v == 0:
                break
    return np.array(offsets), states, stop_gap, travelled


def test_lane_sign():
    for y_off, sign in ((-0.1, -1), (0.1, 1)):
        # y is left. Car left of centre: lane is to the right, error positive.
        det = LaneDetector()
        for _ in range(5):
            result, _ = det.detect(render(1.0, -RADIUS + y_off, 0.0, 0.6))
        assert result.confidence > 0.7, result
        assert np.sign(result.error) == sign, (y_off, result.error)
    print("PASS lane error points back to the lane centre")


def test_laps():
    track_len = 2 * STRAIGHT + 2 * math.pi * RADIUS
    for lane_w, shadow in ((0.45, False), (0.6, False), (0.6, True)):
        offsets, states, _, travelled = drive(lane_w=lane_w, shadow=shadow)
        lost = sum(s != "DRIVE" for s in states)
        rms = float(np.sqrt(np.mean(offsets[75:] ** 2)))
        print(f"lane {lane_w:.2f} m shadow={shadow}: {travelled / track_len:.1f} laps, "
              f"max {offsets.max()*100:.1f} cm, rms {rms*100:.1f} cm, "
              f"non-DRIVE steps {lost}")
        assert offsets.max() < lane_w / 2 - 0.05 and lost == 0
        assert travelled > 1.5 * track_len
    offsets, _, _, _ = drive(lane_w=0.8)
    assert offsets.max() < 0.4 - 0.10
    print(f"lane 0.80 m: max {offsets.max()*100:.1f} cm, stayed inside the lane")
    print("PASS stays inside the lane through straights and curves")


def test_obstacle():
    _, states, gap, _ = drive(seconds=12, obstacle_after=2)
    assert states[-1] == "OBSTACLE" and gap > 0
    print(f"PASS stops for obstacle, gap left {gap:.2f} m")


def test_failsafes():
    c = Controller()
    good = LaneResult(0.1, 1.0)
    assert c.step(good, None, True, 0.0)[2] == "DRIVE"
    assert c.step(LaneResult(), None, True, 0.1)[1] > 0        # brief dropout: carry on
    out = c.step(LaneResult(), None, True, 1.0)
    assert out[1] == 0 and out[2] == "LOST"
    assert c.step(good, None, False, 1.1)[:2] == (0.0, 0.0)
    c = Controller()
    c.step(good, 0.4, True, 0.0)
    assert c.step(good, None, True, 1.0)[2] == "OBSTACLE"
    assert c.step(good, None, True, 2.0)[2] == "OBSTACLE"
    assert c.step(good, None, True, 3.1)[2] == "DRIVE"
    print("PASS lost lane, sensor fault and resume-after-clear")


def node(start, quality, angle, dist):
    b0 = (quality << 2) | ((0 if start else 1) << 1) | int(start)
    a = (int(angle * 64) << 1) | 1
    d = int(dist * 4000)
    return bytes([b0, a & 255, a >> 8, d & 255, d >> 8])


def test_lidar():
    data = b"\x00\x03" + node(True, 40, 10.5, 1.25) + node(False, 40, 200, 0.5)
    nodes, rest = parse_nodes(data + b"\x01")
    assert len(nodes) == 2 and rest == b"\x01"
    assert nodes[0][0] and abs(nodes[0][2] - 10.5) < 0.02 and abs(nodes[0][3] - 1.25) < 0.001
    wall = [(a, 0.4) for a in (-3, -1, 0, 1, 3)]
    assert abs(nearest_in_corridor(wall) - 0.4) < 0.01
    assert nearest_in_corridor([(90, 0.3)] * 5) is None       # beside the car
    assert nearest_in_corridor([(0, 0.3)]) is None            # single noisy point
    print("PASS LiDAR parsing and corridor filter")


def test_serial_protocol():
    import esp_link
    master, slave = pty.openpty()
    received = []

    def fake_esp():
        time.sleep(0.3)        # READY comes after the port is opened, as on the ESP32
        os.write(master, b"READY. Send: S,steer,throttle\r\n")
        data = b""
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            data += os.read(master, 256)
            if data.count(b"\n") >= 4:
                break
        received.extend(data.decode().splitlines())

    thread = threading.Thread(target=fake_esp, daemon=True)
    thread.start()
    link = esp_link.EspLink(port=os.ttyname(slave))
    link.send(0.25, 0.15)
    link.close()
    thread.join(1)
    assert received[0] == "S,0.250,0.150", received
    assert received[-1] == "S,0.000,0.000", received
    print("PASS serial line format and neutral on close")


if __name__ == "__main__":
    test_lane_sign()
    test_failsafes()
    test_lidar()
    test_serial_protocol()
    test_laps()
    test_obstacle()
