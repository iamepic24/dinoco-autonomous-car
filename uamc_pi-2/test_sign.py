"""Sign tests: reading at different sizes, false alarms, and stopping behaviour."""
import random
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

import config as cfg
from lane import LaneResult
from run import Controller, sign_trigger_m
from sign import SignWatcher, distance_m, find_word


def scene(height, word="STOP", kind="red", x=640, y=200, blur=0, seed=0, light=1.0):
    rng = np.random.default_rng(seed)
    frame = np.full((720, 1280, 3), (58, 70, 75), np.uint8)
    scale = height / 22
    thick = max(2, int(height / 7))
    (w, h), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    pad = int(height * 0.7)
    paper = {"red": (35, 35, 200), "white": (240, 240, 240), "yellow": (40, 210, 235)}[kind]
    ink = (255, 255, 255) if kind == "red" else (15, 15, 15)
    cv2.rectangle(frame, (x - pad, y - h - pad), (x + w + pad, y + pad), paper, -1)
    cv2.putText(frame, word, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, ink, thick, cv2.LINE_AA)
    frame = np.clip(frame * light + rng.normal(0, 4, frame.shape), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(frame, (blur, blur), 0) if blur else frame


def test_reading():
    print("letters px : found out of 6 (red, white, yellow; plain and soft focus)")
    results = {}
    for height in (18, 24, 36, 56, 90):
        good, spent = 0, 0.0
        for i, (kind, blur) in enumerate(
                [("red", 0), ("white", 0), ("yellow", 0), ("red", 3), ("white", 3), ("yellow", 3)]):
            frame = scene(height, kind=kind, blur=blur, seed=i)
            start = time.time()
            box = find_word(frame)
            spent += time.time() - start
            good += box is not None
        results[height] = good
        print(f"{height:10} : {good}/6  ({spent / 6:.2f} s per scan)")
    assert min(v for k, v in results.items() if k >= 36) >= 5, results


def test_wrong_words():
    wrong = 0
    for word in ("GO", "SLOW", "PARK", "EXIT", "YIELD", "SHOP", "BUS", "SCHOOL"):
        if find_word(scene(56, word)) is not None:
            wrong += 1
            print("false read:", word)
    print(f"wrong words accepted: {wrong}/8 (SHOP is the closest case)")
    assert wrong <= 1


def test_real_video():
    if len(sys.argv) < 2:
        return
    cap = cv2.VideoCapture(sys.argv[1])
    frames = false = 0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index % 40:
            continue
        frames += 1
        full = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
        false += find_word(full) is not None
    print(f"real run video: {frames} frames sampled, {false} false STOP")
    assert false == 0


def test_controller():
    cfg.SIGN_WAIT_FOR_MOVE = True
    good = LaneResult(0.0, 1.0)
    c = Controller()
    assert c.step(good, None, True, 0.0)[2] == "DRIVE"
    seen = (1.0, sign_trigger_m() + cfg.CRUISE_SPEED_MPS * 2)      # two seconds away
    assert c.step(good, None, True, 2.9, seen)[2] == "DRIVE"
    assert c.step(good, None, True, 3.1, seen)[2] == "SIGN"
    late = 3.1 + cfg.SIGN_HOLD_S + 60
    assert c.step(good, None, True, late, seen)[2] == "SIGN"             # sign still there
    assert c.step(good, None, True, late + 0.1, seen, True)[2] == "DRIVE"     # sign moved
    assert c.step(good, None, True, late + 1, seen)[2] == "DRIVE"        # stays going
    cfg.SIGN_WAIT_FOR_MOVE = False
    c = Controller()
    c.step(good, None, True, 3.1, (3.0, 0.0))
    assert c.step(good, None, True, 3.1 + cfg.SIGN_HOLD_S + 0.1, (3.0, 0.0))[2] == "DRIVE"
    cfg.SIGN_WAIT_FOR_MOVE = True
    c = Controller()
    assert c.step(good, 0.2, True, 0.0, (0.0, 0.1))[2] == "OBSTACLE"
    print("PASS waits for the sign to move, timed mode still works, obstacle comes first")


def test_scene_watch():
    from run import SceneWatch
    floor = np.full((240, 426, 3), 70, np.uint8)
    sign = floor.copy()
    sign[40:180, 120:300] = 235
    w = SceneWatch()
    assert not w.moved(sign, True, 0.0)
    assert not w.moved(sign, True, 1.1)                 # remembers the view
    assert not w.moved(sign, True, 2.0)                 # same view
    hand = sign.copy()
    hand[100:230, 180:260] = 150
    assert not w.moved(hand, True, 2.5)
    assert not w.moved(hand, True, 3.0)                 # brief hand, not long enough
    assert not w.moved(sign, True, 3.2)                 # hand gone, sign still there
    assert not w.moved(floor, True, 4.0)
    assert not w.moved(floor, True, 5.0)
    assert w.moved(floor, True, 5.6)                    # stayed changed for 1.5 s
    print("PASS a passing hand is ignored, a moved sign releases the car")


def test_watcher_needs_two_reads():
    cam = SimpleNamespace(frame=None, stamp=0.0)
    w = SignWatcher(cam)
    w._record(1.0, 0.5)
    assert w.confirmed() is None
    w._record(1.2, 0.45)
    assert w.confirmed() == (1.2, 0.45)
    w.close()
    w = SignWatcher(cam)
    w._record(1.0, 0.5)
    w._record(9.0, 0.5)
    assert w.confirmed() is None
    w.close()
    print("PASS one lone read is never enough")


def test_distance():
    assert abs(distance_m((0, 0, 100, cfg.CAMERA_F_PX * cfg.WORD_HEIGHT_M / 0.4)) - 0.4) < 1e-6
    print(f"trigger distance: sign {sign_trigger_m():.2f} m from camera, "
          f"LiDAR stop at {cfg.STOP_DISTANCE_M:.2f} m")


if __name__ == "__main__":
    test_controller()
    test_scene_watch()
    test_watcher_needs_two_reads()
    test_distance()
    test_wrong_words()
    test_reading()
    test_real_video()
