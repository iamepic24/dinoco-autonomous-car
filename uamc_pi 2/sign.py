"""Finds a red STOP sign and reads the word on it.

Word reading is slow, so it runs in its own thread. The control loop only ever
reads the latest answer and is never held up by it.
"""
import difflib
import re
import threading
import time

import cv2
import numpy as np

import config as cfg

OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SCAN_WIDTH = 640


def find_signs(bgr):
    """Return sign-shaped red boxes (x, y, w, h) in full-size pixels, largest first."""
    scale = SCAN_WIDTH / bgr.shape[1]
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(small)
    red = cv2.threshold(cv2.subtract(r, cv2.max(g, b)), cfg.SIGN_RED_MARGIN, 255,
                        cv2.THRESH_BINARY)[1]
    # Closing fills the white letters, so a sign becomes one solid blob.
    size = max(5, int(SCAN_WIDTH * 0.02) | 1)
    solid = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    contours = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        touches_edge = (x == 0 or y == 0 or x + w >= small.shape[1]
                        or y + h >= small.shape[0])
        if (touches_edge or h / scale < cfg.SIGN_MIN_PX or not 0.7 <= w / h <= 1.45
                or cv2.contourArea(contour) / (w * h) < 0.65):
            continue
        boxes.append(tuple(int(v / scale) for v in (x, y, w, h)))
    return sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)


def read_word(bgr, box):
    """Return the text on a sign, in capitals with anything else removed."""
    import pytesseract
    x, y, w, h = box
    inset_x, inset_y = int(w * 0.03), int(h * 0.18)
    crop = bgr[y + inset_y:y + h - inset_y, x + inset_x:x + w - inset_x]
    if crop.size == 0:
        return ""
    # White letters are bright in green. The red sign is dark in green.
    green = cv2.resize(crop[:, :, 1], None, fx=160 / crop.shape[0], fy=160 / crop.shape[0],
                       interpolation=cv2.INTER_CUBIC)
    letters = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Keep letter-sized shapes. This drops sign corners and thin border strips.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(letters)
    high, wide = letters.shape
    keep = [i for i in range(1, count)
            if 0.25 * high < stats[i][3] < 0.97 * high
            and 0.05 * wide < stats[i][2] < 0.45 * wide]
    binary = np.where(np.isin(labels, keep), 0, 255).astype(np.uint8)
    binary = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    text = pytesseract.image_to_string(binary, config=OCR_CONFIG, timeout=2)
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_stop(word):
    """Accept the word with at most one misread letter, such as ST0P."""
    return 3 <= len(word) <= 5 and difflib.SequenceMatcher(None, "STOP", word).ratio() >= 0.75


def distance_m(box):
    """Distance from the camera, from how tall the sign looks."""
    return cfg.CAMERA_F_PX * cfg.SIGN_HEIGHT_M / box[3]


class SignWatcher:
    def __init__(self, camera):
        self.camera = camera
        self.reads = []         # (frame time, distance) for each time the word was read
        self.box = None         # latest sign shape and its frame time
        self.result = None
        self.ok = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        last_stamp = 0.0
        while self._running:
            frame, stamp = self.camera.frame, self.camera.stamp
            if frame is None or stamp == last_stamp:
                time.sleep(0.01)
                continue
            last_stamp = stamp
            boxes = find_signs(frame)
            if not boxes:
                time.sleep(0.03)
                continue
            self.box = (boxes[0], stamp, frame.shape[1])
            try:
                word = read_word(frame, boxes[0])
            except Exception as error:      # missing OCR, or one failed read
                print(f"WARNING: sign reading failed: {error}")
                self.ok = False
                return
            if is_stop(word):
                self._record(stamp, distance_m(boxes[0]))

    def _record(self, stamp, distance):
        self.reads = [r for r in self.reads
                      if stamp - r[0] <= cfg.SIGN_CONFIRM_WINDOW_S] + [(stamp, distance)]
        if len(self.reads) >= cfg.SIGN_CONFIRM_READS:
            self.result = (stamp, distance)

    def lane_ignore(self, now):
        """Return the sign's box at lane-detection size, so its red isn't taken for tape."""
        if self.box is None or now - self.box[1] > 0.3:
            return None
        (x, y, w, h), _, width = self.box
        scale = cfg.PROC_WIDTH / width
        margin = 0.1 * w
        return (int((x - margin) * scale), int((y - margin) * scale),
                int((w + 2 * margin) * scale), int((h + 2 * margin) * scale))

    def confirmed(self):
        """Return (time seen, distance) for the latest confirmed sign, or None."""
        return self.result

    def close(self):
        self._running = False
        self._thread.join(timeout=3)
