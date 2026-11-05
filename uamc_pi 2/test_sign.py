"""Sign tests: reading at different sizes, false alarms, and stopping behaviour."""
import random
import sys
from types import SimpleNamespace

import cv2
import numpy as np

import config as cfg
from lane import LaneResult
from run import Controller, sign_trigger_m
from sign import SignWatcher, distance_m, find_signs, is_stop, read_word


def sign_image(size, word="STOP"):
    img = np.full((size, size, 3), 255, np.uint8)
    c = size / 2
    pts = np.array([[c + 0.49 * size * np.cos(np.radians(22.5 + 45 * i)),
                     c + 0.49 * size * np.sin(np.radians(22.5 + 45 * i))]
                    for i in range(8)], np.int32)
    cv2.fillConvexPoly(img, pts, (35, 35, 200))
    scale = size / 88
    thick = max(2, int(size / 26))
    (w, h), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    cv2.putText(img, word, (int(c - w / 2), int(c + h / 2)), cv2.FONT_HERSHEY_DUPLEX,
                scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def scene(size, word="STOP", x=760, y=170, angle=0, brightness=1.0, blur=0, seed=0):
    rng = np.random.default_rng(seed)
    frame = np.full((720, 1280, 3), (58, 70, 75), np.uint8)
    tile = sign_image(size, word)
    if angle:
        src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
        shrink = size * np.sin(np.radians(angle)) * 0.25
        dst = np.float32([[0, 0], [size, shrink], [size, size - shrink], [0, size]])
        tile = cv2.warpPerspective(tile, cv2.getPerspectiveTransform(src, dst), (size, size),
                                   borderValue=(58, 70, 75))
    frame[y:y + size, x:x + size] = tile
    frame = np.clip(frame.astype(np.float32) * brightness + rng.normal(0, 4, frame.shape),
                    0, 255).astype(np.uint8)
    if blur:
        frame = cv2.GaussianBlur(frame, (blur, blur), 0)
    return frame


def test_reading():
    print("size px : read correctly out of 8 varied views")
    hits_by_size = {}
    for size in (44, 60, 80, 120, 200, 320):
        good = 0
        for i, (angle, light, blur) in enumerate(
                [(0, 1, 0), (15, 1, 0), (30, 1, 0), (0, 0.6, 0), (0, 0.4, 0), (0, 1, 3), (0, 1, 5),
                 (20, 0.6, 3)]):
            frame = scene(size, angle=angle, brightness=light, blur=blur, seed=i)
            boxes = find_signs(frame)
            good += bool(boxes) and is_stop(read_word(frame, boxes[0]))
        hits_by_size[size] = good
        print(f"{size:7} : {good}/8")
    assert min(v for k, v in hits_by_size.items() if k >= 80) >= 6, hits_by_size


def test_wrong_words():
    wrong = 0
    for word in ("GO", "SLOW", "PARK", "EXIT", "YIELD", "60", "BUS", "SCHOOL"):
        frame = scene(180, word)
        boxes = find_signs(frame)
        if boxes and is_stop(read_word(frame, boxes[0])):
            wrong += 1
            print("false read:", word)
    print(f"wrong words accepted: {wrong}/8")
    assert wrong == 0


def test_real_video():
    if len(sys.argv) < 2:
        return
    cap = cv2.VideoCapture(sys.argv[1])
    frames = shapes = false = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        full = cv2.resize(frame, (1280, 960))[120:840]
        boxes = find_signs(full)
        shapes += bool(boxes)
        if boxes and is_stop(read_word(full, boxes[0])):
            false += 1
    print(f"real run video: {frames} frames, {shapes} with a sign shape, {false} read as STOP")
    assert false == 0


def test_controller():
    good = LaneResult(0.0, 1.0)
    c = Controller()
    assert c.step(good, None, True, 0.0)[2] == "DRIVE"
    seen = (1.0, sign_trigger_m() + cfg.CRUISE_SPEED_MPS * 2)      # two seconds away
    assert c.step(good, None, True, 1.1, seen)[2] == "DRIVE"
    assert c.step(good, None, True, 2.9, seen)[2] == "DRIVE"
    out = c.step(good, None, True, 3.1, seen)
    assert out[2] == "SIGN" and out[1] == 0
    assert c.step(good, None, True, 3.1 + cfg.SIGN_HOLD_S - 0.1, seen)[2] == "SIGN"
    assert c.step(good, None, True, 3.1 + cfg.SIGN_HOLD_S + 0.1, seen)[2] == "DRIVE"
    assert c.step(good, None, True, 3.1 + cfg.SIGN_HOLD_S + 5, seen)[2] == "DRIVE"
    # An obstacle beats the sign.
    c = Controller()
    assert c.step(good, 0.2, True, 0.0, (0.0, 0.1))[2] == "OBSTACLE"
    print("PASS sign stop, 3 second hold, moves off, ignores same sign, obstacle comes first")


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
    w._record(5.0, 0.5)
    assert w.confirmed() is None
    w.close()
    print("PASS one lone read is never enough")


def test_sign_not_taken_for_tape():
    from lane import LaneDetector
    frame = cv2.resize(scene(150, x=560, y=200), (cfg.PROC_WIDTH, cfg.PROC_HEIGHT))
    for x in (40, cfg.PROC_WIDTH - 40):
        cv2.line(frame, (x, 40), (x, 130), (35, 35, 200), 8)
    boxes = find_signs(cv2.resize(frame, (1280, 720)))
    assert boxes
    scale = cfg.PROC_WIDTH / 1280
    x, y, w, h = [int(v * scale) for v in boxes[0]]
    plain, covered = LaneDetector(), LaneDetector()
    for _ in range(5):
        a, _m = plain.detect(frame)
        b, _m = covered.detect(frame, (x - 4, y - 4, w + 8, h + 8))
    print(f"sign in lane: lane error {a.error:+.2f} without masking, {b.error:+.2f} with")
    assert abs(b.error) < 0.1


def test_distance():
    cfg_f, cfg_h = cfg.CAMERA_F_PX, cfg.SIGN_HEIGHT_M
    assert abs(distance_m((0, 0, 100, cfg_f * cfg_h / 0.4)) - 0.4) < 1e-6
    print(f"trigger distance: sign {sign_trigger_m():.2f} m from camera, "
          f"LiDAR stop at {cfg.STOP_DISTANCE_M:.2f} m")


if __name__ == "__main__":
    test_controller()
    test_watcher_needs_two_reads()
    test_distance()
    test_sign_not_taken_for_tape()
    test_wrong_words()
    test_reading()
    test_real_video()
