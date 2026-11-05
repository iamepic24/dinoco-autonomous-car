"""Minimal RPLIDAR C1 reader using the standard scan command. Needs only pyserial."""
import math
import threading
import time

import serial

import config as cfg

CMD_STOP = b"\xA5\x25"
CMD_SCAN = b"\xA5\x20"
DESCRIPTOR = b"\xA5\x5A\x05\x00\x00\x40\x81"
NODE_SIZE = 5


def parse_nodes(buf):
    """Return ([(new_scan, quality, angle_deg, dist_m), ...], unused_bytes)."""
    nodes, i = [], 0
    while i + NODE_SIZE <= len(buf):
        b0, b1, b2, b3, b4 = buf[i:i + NODE_SIZE]
        start, inverse = b0 & 1, (b0 >> 1) & 1
        if start == inverse or not (b1 & 1):
            i += 1                       # lost sync: slide forward one byte
            continue
        angle = (((b2 << 8) | b1) >> 1) / 64.0
        dist = ((b4 << 8) | b3) / 4000.0
        if angle > 360:                  # only possible on a false sync
            i += 1
            continue
        nodes.append((bool(start), b0 >> 2, angle, dist))
        i += NODE_SIZE
    return nodes, bytes(buf[i:])


def nearest_in_corridor(points):
    """Forward distance to the obstacle in the car's path, or None if clear."""
    ahead = []
    for angle, dist in points:
        if dist < cfg.LIDAR_MIN_RANGE_M:
            continue
        rel = math.radians(angle - cfg.LIDAR_FORWARD_DEG)
        forward, side = dist * math.cos(rel), dist * math.sin(rel)
        if forward > 0 and abs(side) <= cfg.CORRIDOR_HALF_WIDTH_M:
            ahead.append(forward)
    if len(ahead) < cfg.MIN_OBSTACLE_POINTS:
        return None
    return sorted(ahead)[cfg.MIN_OBSTACLE_POINTS - 1]


class Lidar:
    def __init__(self, port=None, baud=None):
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate = port or cfg.LIDAR_PORT, baud or cfg.LIDAR_BAUD
        self.ser.timeout = 0.2
        self.ser.dtr = False
        self.ser.open()
        self.obstacle_m = None
        self.last_scan = []
        self.last_scan_time = 0.0
        self._running = True
        self._start_scan()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _start_scan(self):
        self.ser.write(CMD_STOP)
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        self.ser.write(CMD_SCAN)
        data, end = b"", time.monotonic() + 3.0
        while DESCRIPTOR not in data:
            if time.monotonic() > end:
                raise RuntimeError(f"No scan response from LiDAR. Got {data[:16].hex()}")
            data += self.ser.read(32)
        self._leftover = data.split(DESCRIPTOR, 1)[1]

    def _reader(self):
        buf, points = self._leftover, []
        while self._running:
            buf += self.ser.read(self.ser.in_waiting or NODE_SIZE)
            nodes, buf = parse_nodes(buf)
            for new_scan, quality, angle, dist in nodes:
                if new_scan and len(points) > 50:
                    self.last_scan = points
                    self.obstacle_m = nearest_in_corridor(points)
                    self.last_scan_time = time.monotonic()
                    points = []
                if quality > 0 and dist > 0:
                    points.append((angle, dist))

    def is_fresh(self):
        return time.monotonic() - self.last_scan_time < cfg.LIDAR_STALE_S

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self.ser.write(CMD_STOP)
        self.ser.close()
