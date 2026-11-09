"""Autonomous run: camera + LiDAR -> Pi -> ESP32.

    python run.py --dry-run --no-lidar   # watch decisions, motors never move
    python run.py --no-lidar             # lane-keeping only
    python run.py --delay 10             # full run, starts after 10 seconds
"""
import argparse
import collections
import csv
import shutil
import signal
import threading
import time
from pathlib import Path

import cv2

import config as cfg
from lane import LaneDetector, draw_overlay
from zebra import ZebraDetector


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
    return cfg.SIGN_CONTACT_M + cfg.SIGN_GAP_M + cfg.COAST_M


class SceneWatch:
    """Tells when the view ahead has changed and stayed changed while the car waits."""

    def __init__(self):
        self.reference = None
        self.start = None
        self.changed_since = None

    def moved(self, frame, waiting, now):
        if not waiting:
            self.reference = self.start = self.changed_since = None
            return False
        gray = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (106, 60)), cv2.COLOR_BGR2GRAY),
                                (5, 5), 0)
        if self.start is None:
            self.start = now
        if self.reference is None:
            if now - self.start >= 1.0:     # lets the car settle first
                self.reference = gray
            return False
        changed = (cv2.absdiff(gray, self.reference) > 30).mean() > cfg.SIGN_MOVED_AREA
        if not changed:
            self.changed_since = None
        elif self.changed_since is None:
            self.changed_since = now
        return self.changed_since is not None and now - self.changed_since >= cfg.SIGN_MOVED_S


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
        self.steer_history = collections.deque()
        self.zebra_until = 0.0
        self.zebra_ignore_until = 0.0
        self.sign_stop_at = None
        self.sign_holding = False
        self.sign_hold_until = 0.0
        self.sign_ignore_until = 0.0

    def _steering(self, error, now):
        if self.prev_error is not None and now > self.prev_time:
            rate = (error - self.prev_error) / (now - self.prev_time)
            self.d_error = 0.7 * self.d_error + 0.3 * rate
        self.prev_error, self.prev_time = error, now
        steer = cfg.KP * error + cfg.KD * self.d_error
        return max(-cfg.STEER_LIMIT, min(cfg.STEER_LIMIT, steer))

    def _delay_turn(self, steer, now):
        """Hold back turning in by TURN_DELAY_M. Straightening is never held back."""
        self.steer_history.append((now, steer))
        wait = cfg.TURN_DELAY_M / cfg.CRUISE_SPEED_MPS
        while len(self.steer_history) > 1 and self.steer_history[1][0] <= now - wait:
            self.steer_history.popleft()
        direction = 1.0 if steer >= 0 else -1.0
        earlier = max(0.0, self.steer_history[0][1] * direction)
        return direction * min(abs(steer), max(earlier, cfg.TURN_DELAY_BELOW))

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

    def _update_sign(self, sign, now, sign_moved):
        """Return True while the car must hold still for a STOP sign."""
        if self.sign_holding:
            if now < self.sign_hold_until or (cfg.SIGN_WAIT_FOR_MOVE and not sign_moved):
                return True
            self.sign_holding = False
            self.sign_ignore_until = now + cfg.SIGN_COOLDOWN_S
            return False
        if now < self.sign_ignore_until:
            return False
        if sign is not None and sign[0] > self.sign_ignore_until:
            seen_at, sign_m = sign
            remaining = max(0.0, sign_m - sign_trigger_m())
            stop_at = seen_at + remaining / cfg.CRUISE_SPEED_MPS
            if self.sign_stop_at is None or stop_at < self.sign_stop_at:
                self.sign_stop_at = stop_at     # a reading can only bring the stop forward
        if self.sign_stop_at is not None and now >= self.sign_stop_at:
            self.sign_stop_at = None
            self.sign_holding = True
            self.sign_hold_until = now + cfg.SIGN_HOLD_S
            return True
        return False

    def _update_zebra(self, zebra, now):
        """Return True while the car must hold still for a zebra crossing."""
        if now < self.zebra_until:
            return True
        if zebra and now >= self.zebra_ignore_until:
            self.zebra_until = now + cfg.ZEBRA_HOLD_S
            self.zebra_ignore_until = self.zebra_until + cfg.ZEBRA_COOLDOWN_S
            return True
        return False

    def step(self, lane, distance, sensors_ok, now, sign=None, sign_moved=False, zebra=False):
        """Return (steer, throttle, state). LiDAR obstacles always come first."""
        if not sensors_ok:
            return 0.0, 0.0, "FAULT"

        self._update_obstacle(distance, now)
        if self.blocked:
            throttle = cfg.BRAKE_THROTTLE if now < self.brake_until else 0.0
            return self.last_steer, throttle, "OBSTACLE"
        if self._update_sign(sign, now, sign_moved):
            return self.last_steer, 0.0, "SIGN"
        if self._update_zebra(zebra, now):
            return self.last_steer, 0.0, "ZEBRA"

        if lane.confidence >= cfg.MIN_CONFIDENCE:
            self.last_lane_time = now
            self.last_steer = self._delay_turn(self._steering(lane.error, now), now)
        elif (self.last_lane_time is None
              or now - self.last_lane_time > cfg.LANE_LOST_S):
            self.prev_error = None
            lost_for = None if self.last_lane_time is None else now - self.last_lane_time
            if lost_for is not None and lost_for < cfg.LANE_LOST_S + cfg.SEARCH_TURN_S:
                steer = cfg.SEARCH_DIRECTION * cfg.STEER_LIMIT
                return steer, cfg.BASE_THROTTLE + cfg.TURN_BOOST * cfg.STEER_LIMIT, "SEARCH"
            return self.last_steer, 0.0, "LOST"

        throttle = cfg.BASE_THROTTLE + cfg.TURN_BOOST * abs(self.last_steer)
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
        scene = SceneWatch()
        zebras = ZebraDetector() if cfg.ZEBRA_ENABLED else None
        period = 1.0 / cfg.LOOP_HZ
        launched = time.monotonic()
        start = launched + args.delay
        last_state, printed = None, 0.0
        print(f"Starting in {args.delay:.0f} s, once both lane lines are measured. "
              "Ctrl+C stops the car.")

        while not stop["now"]:
            tick = time.monotonic()
            if args.time and (tick - start > args.time
                              or tick - launched > args.delay + args.time + 30):
                break       # the second limit ends a run that never got to start

            frame, stamp = camera.latest()
            camera_ok = frame is not None and tick - stamp < cfg.CAMERA_STALE_S
            lidar_ok = lidar is None or lidar.is_fresh()
            distance = lidar.obstacle_m if lidar else None
            sign = signs.confirmed() if signs else None
            # Only readings taken while driving count, and old ones are dropped.
            if sign and (sign[0] < start or tick - sign[0] > 2.0):
                sign = None

            if camera_ok:
                lane, _ = detector.detect(frame, signs.lane_ignore(tick) if signs else None)
                moved = scene.moved(frame, controller.sign_holding, tick)
                zebra = zebras is not None and tick >= start and zebras.seen(frame)
                steer, throttle, state = controller.step(
                    lane, distance, camera_ok and lidar_ok, tick, sign, moved, zebra)
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
