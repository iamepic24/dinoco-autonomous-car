"""Zebra tests: a drawn crossing, stopping behaviour, and no false alarms on real runs."""
import sys

import cv2
import numpy as np

import config as cfg
from lane import LaneResult
from run import Controller
from zebra import ZebraDetector


def road(bars=0, floor=60):
    img = np.full((cfg.PROC_HEIGHT, cfg.PROC_WIDTH, 3), floor, np.uint8)
    cv2.line(img, (20, 120), (140, 20), (230, 230, 230), 7)
    cv2.line(img, (400, 120), (285, 20), (230, 230, 230), 7)
    for i in range(bars):
        x = 95 + i * 42
        cv2.fillConvexPoly(img, np.array([[x, 75], [x + 18, 75], [x + 22, 130], [x + 4, 130]],
                                         np.int32), (235, 235, 235))
    return img


def confirmed(frame):
    det = ZebraDetector()
    return any(det.seen(frame) for _ in range(6))


def test_detection():
    assert confirmed(road(6))
    assert confirmed(road(5))
    assert not confirmed(road(0))
    assert not confirmed(road(2))
    print("PASS finds a crossing, ignores plain lane lines and two bars")


def test_noise():
    rng = np.random.default_rng(1)
    noisy = np.clip(road(6).astype(np.int16) + rng.normal(0, 8, road(6).shape), 0, 255)
    assert confirmed(noisy.astype(np.uint8))
    print("PASS finds a crossing through camera noise")


def test_controller():
    good = LaneResult(0.0, 1.0)
    c = Controller()
    assert c.step(good, None, True, 0.0)[2] == "DRIVE"
    out = c.step(good, None, True, 1.0, zebra=True)
    assert out[2] == "ZEBRA" and out[1] == 0
    assert c.step(good, None, True, 3.9)[2] == "ZEBRA"
    assert c.step(good, None, True, 4.1)[2] == "DRIVE"
    assert c.step(good, None, True, 5.0, zebra=True)[2] == "DRIVE"      # still on it
    assert c.step(good, 0.2, True, 20.0, zebra=True)[2] == "OBSTACLE"
    print("PASS stops, holds, moves off, ignores the same crossing, obstacle comes first")


def test_real_videos():
    for path in sys.argv[1:]:
        cap = cv2.VideoCapture(path)
        det = ZebraDetector()
        frames = false = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            frame = cv2.resize(frame, (cfg.PROC_WIDTH, cfg.PROC_HEIGHT))
            false += det.seen(frame)
        print(f"real video: {frames} frames, {false} false zebra frames")
        assert false == 0


if __name__ == "__main__":
    test_detection()
    test_noise()
    test_controller()
    test_real_videos()
