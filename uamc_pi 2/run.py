"""Autonomous run: camera + LiDAR -> Pi -> ESP32.

    python run.py --dry-run --no-lidar   # watch decisions, motors never move
    python run.py --no-lidar             # lane-keeping only
    python run.py --delay 10             # full run, starts after 10 seconds
"""
import argparse
import csv
import shutil
import signal
import threading
import time
from pathlib import Path

import cv2

import config as cfg
from lane import LaneDetector, draw_overlay


class Camera:
    """Reads in a thread so the control loop always sees the newest frame."""

    def __init__(self):
        self.cap = cv2.VideoCapture(cfg.CAMERA_INDEX, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open the webcam")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAPTURE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.CAPTURE_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.stamp = None, 0.0
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                self.frame, self.stamp = frame, time.monotonic()

    def latest(self):
        if self.frame is None:
            return None, 0.0
        return cv2.resize(self.frame, (cfg.PROC_WIDTH, cfg.PROC_HEIGHT)), self.stamp

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


def sign_trigger_m():
    """Camera-to-sign distance at which throttle must cut to stop on target."""
    return max(0.0, cfg.SIGN_GAP_M - cfg.CAMERA_POKE_M) + cfg.COAST_M


class Controller:
    """All driving decisions. Has no hardware in it, so it can be simulated."""

    def __init__(self):
        self.prev_error = None
        self.prev_time = None
        self.d_error = 0.0
        self.last_lane_time = None
        self.last_steer = 0.0
        self.blocked = False
        self.clear_since = None
        self.brake_until = 0.0
        self.sign_stop_at = None
        self.sign_hold_until = 0.0
        self.sign_ignore_until = 0.0

    def _steering(self, error, now):
        if self.prev_error is not None and now > self.prev_time:
            rate = (error - self.prev_error) / (now - self.prev_time)
            self.d_error = 0.7 * self.d_error + 0.3 * rate
        self.prev_error, self.prev_time = error, now
        steer = cfg.KP * error + cfg.KD * self.d_error
        return max(-cfg.STEER_LIMIT, min(cfg.STEER_LIMIT, steer))

    def _update_obstacle(self, distance, now):
        if not self.blocked:
            if distance is not None and distance <= cfg.STOP_DISTANCE_M:
                self.blocked = True
                self.clear_since = None
                self.brake_until = now + cfg.BRAKE_TIME_S
            return
        limit = cfg.STOP_DISTANCE_M + cfg.CLEAR_MARGIN_M
        if distance is not None and distance <= limit:
            self.clear_since = None
        elif self.clear_since is None:
            self.clear_since = now
        elif now - self.clear_since >= cfg.CLEAR_HOLD_S:
            self.blocked = False

    def _update_sign(self, sign, now):
        """Return True while the car must hold still for a STOP sign."""
        if now < self.sign_hold_until:
            return True
        if now < self.sign_ignore_until:
            return False
        if sign is not None and self.sign_stop_at is None:
            seen_at, sign_m = sign
            remaining = max(0.0, sign_m - sign_trigger_m())
            self.sign_stop_at = seen_at + remaining / cfg.CRUISE_SPEED_MPS
        if self.sign_stop_at is not None and now >= self.sign_stop_at:
            self.sign_stop_at = None
            self.sign_hold_until = now + cfg.SIGN_HOLD_S
            self.sign_ignore_until = self.sign_hold_until + cfg.SIGN_COOLDOWN_S
            return True
        return False

    def step(self, lane, distance, sensors_ok, now, sign=None):
        """Return (steer, throttle, state). LiDAR obstacles always come first."""
        if not sensors_ok:
            return 0.0, 0.0, "FAULT"

        self._update_obstacle(distance, now)
        if self.blocked:
            throttle = cfg.BRAKE_THROTTLE if now < self.brake_until else 0.0
            return self.last_steer, throttle, "OBSTACLE"
        if self._update_sign(sign, now):
            return self.last_steer, 0.0, "SIGN"

        if lane.confidence >= cfg.MIN_CONFIDENCE:
            self.last_lane_time = now
            self.last_steer = self._steering(lane.error, now)
        elif (self.last_lane_time is None
              or now - self.last_lane_time > cfg.LANE_LOST_S):
            self.prev_error = None
            return self.last_steer, 0.0, "LOST"

        span = max(0.0, cfg.BASE_THROTTLE - cfg.MIN_THROTTLE)
        throttle = cfg.BASE_THROTTLE - cfg.TURN_SLOWDOWN * span * abs(self.last_steer)
        return self.last_steer, throttle, "DRIVE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="never send to the ESP32")
    parser.add_argument("--no-lidar", action="store_true", help="skip the obstacle stop")
    parser.add_argument("--delay", type=float, default=5, help="seconds before moving")
    parser.add_argument("--time", type=float, default=0, help="stop after N seconds")
    parser.add_argument("--no-sign", action="store_true", help="skip the STOP sign")
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))

    camera, lidar, esp, video, log_file, signs = None, None, None, None, None, None
    try:
        camera = Camera()
        if not args.no_lidar:
            from lidar_c1 import Lidar
            lidar = Lidar()
        if cfg.SIGN_ENABLED and not args.no_sign:
            from sign import SignWatcher
            if not shutil.which("tesseract"):
                print("WARNING: tesseract is not installed, so STOP signs won't be read.")
            signs = SignWatcher(camera)
        if not args.dry_run:
            from esp_link import EspLink
            esp = EspLink()

        writer = None
        if not args.no_record:
            run_dir = Path("runs") / time.strftime("%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True)
            video = cv2.VideoWriter(str(run_dir / "camera.avi"),
                                    cv2.VideoWriter_fourcc(*"MJPG"), cfg.LOOP_HZ,
                                    (cfg.PROC_WIDTH, cfg.PROC_HEIGHT))
            log_file = open(run_dir / "log.csv", "w", newline="")
            writer = csv.writer(log_file)
            writer.writerow(["t", "state", "error", "confidence", "steer",
                             "throttle", "obstacle_m", "sign_m", "loop_ms"])

        detector = LaneDetector()
        controller = Controller()
        period = 1.0 / cfg.LOOP_HZ
        start = time.monotonic() + args.delay
        last_state, printed = None, 0.0
        print(f"Starting in {args.delay:.0f} s, once both lane lines are measured. "
              "Ctrl+C stops the car.")

        while not stop["now"]:
            tick = time.monotonic()
            if args.time and tick - start > args.time:
                break

            frame, stamp = camera.latest()
            camera_ok = frame is not None and tick - stamp < cfg.CAMERA_STALE_S
            lidar_ok = lidar is None or lidar.is_fresh()
            distance = lidar.obstacle_m if lidar else None
            sign = signs.confirmed() if signs else None

            if camera_ok:
                lane, _ = detector.detect(frame, signs.lane_ignore(tick) if signs else None)
                steer, throttle, state = controller.step(
                    lane, distance, camera_ok and lidar_ok, tick, sign)
            else:
                lane = None
                steer, throttle, state = controller.step(None, None, False, tick)

            if not detector.fitted:         # never drive on a guessed lane width
                start = max(start, tick + 1.0)
            if tick < start:
                throttle = 0.0
                state = "WAIT" if detector.fitted else "NO-WIDTH"
            if esp:
                esp.send(steer, throttle)

            loop_ms = (time.monotonic() - tick) * 1000
            if state != last_state or tick - printed > 1:
                err = f"{lane.error:+.2f}" if lane else "-"
                conf = f"{lane.confidence:.2f}" if lane else "-"
                print(f"{state:8} err={err} conf={conf} steer={steer:+.2f} "
                      f"thr={throttle:.2f} obstacle={distance} "
                      f"sign={sign[1] if sign else None} loop={loop_ms:.0f}ms")
                last_state, printed = state, tick
            if writer and lane:
                writer.writerow([f"{tick:.3f}", state, f"{lane.error:.3f}",
                                 f"{lane.confidence:.2f}", f"{steer:.3f}",
                                 f"{throttle:.3f}", distance,
                                 sign[1] if sign else "", f"{loop_ms:.1f}"])
                video.write(draw_overlay(frame, lane, f"{state} s={steer:+.2f}"))

            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    finally:
        for closer in (esp, lidar, signs, camera):
            if closer:
                closer.close()
        if video:
            video.release()
        if log_file:
            log_file.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
