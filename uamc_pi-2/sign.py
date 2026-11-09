"""Reads the word STOP anywhere in the camera view, in any colour.

Word reading is slow, so it runs in its own thread. The control loop only ever
reads the latest answer and is never held up by it.
"""
import difflib
import os
import re
import threading
import time

import cv2
import numpy as np

import config as cfg

# Tesseract would otherwise use every CPU core and slow the control loop.
os.environ["OMP_THREAD_LIMIT"] = "1"
OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def is_stop(word):
    """Accept the word with at most one misread letter, such as ST0P."""
    return 3 <= len(word) <= 5 and difflib.SequenceMatcher(None, "STOP", word).ratio() >= 0.75


def letter_shapes(source):
    """Return letter-sized dark shapes in source as (x, y, w, h, label), and the label image."""
    binary = cv2.adaptiveThreshold(source, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 101, cfg.WORD_CONTRAST)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    shapes = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if (cfg.WORD_MIN_PX <= h <= 0.3 * source.shape[0] and 0.2 * h <= w <= 1.5 * h
                and area >= 0.12 * w * h):
            shapes.append((int(x), int(y), int(w), int(h), i))
    return shapes, labels


def letter_groups(shapes):
    """Chain neighbouring shapes of similar size into words of 3 to 5 letters."""
    shapes = sorted(shapes)
    used, groups = set(), []
    for start, first in enumerate(shapes):
        if start in used:
            continue
        chain, last = [start], first
        for i in range(start + 1, len(shapes)):
            s = shapes[i]
            gap = s[0] - (last[0] + last[2])
            drift = abs((s[1] + s[3] / 2) - (last[1] + last[3] / 2))
            if (-0.2 * last[3] <= gap <= 0.9 * last[3] and drift <= 0.7 * last[3]
                    and 0.6 <= s[3] / last[3] <= 1.6):
                chain.append(i)
                last = s
        if 3 <= len(chain) <= 5:
            used.update(chain)
            groups.append([shapes[i] for i in chain])
    return groups


def read_group(group, labels):
    """Redraw a word as filled black letters, level it, and read it."""
    import pytesseract
    x0 = min(s[0] for s in group)
    y0 = min(s[1] for s in group)
    x1 = max(s[0] + s[2] for s in group)
    y1 = max(s[1] + s[3] for s in group)
    pad = (y1 - y0) // 2
    mask = np.isin(labels[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad],
                   [s[4] for s in group]).astype(np.uint8) * 255
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    filled = np.full(mask.shape, 255, np.uint8)
    cv2.drawContours(filled, contours, -1, 0, -1)
    middles = np.array([(s[0] + s[2] / 2, s[1] + s[3] / 2) for s in group])
    slope = np.polyfit(middles[:, 0], middles[:, 1], 1)[0]
    high, wide = filled.shape
    turn = cv2.getRotationMatrix2D((wide / 2, high / 2), np.degrees(np.arctan(slope)), 1)
    level = cv2.warpAffine(filled, turn, (wide, high), borderValue=255)
    scale = 64 / max(1, y1 - y0)
    level = cv2.resize(level, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    level = cv2.GaussianBlur(level, (3, 3), 0)
    data = pytesseract.image_to_data(level, config=OCR_CONFIG, timeout=2,
                                     output_type=pytesseract.Output.DICT)
    words = [(re.sub(r"[^A-Z0-9]", "", t.upper()), float(c))
             for t, c in zip(data["text"], data["conf"]) if t.strip()]
    box = (x0, y0, x1 - x0, int(np.median([s[3] for s in group])))    # h is letter height
    return (words[0][0], words[0][1], box) if len(words) == 1 else ("", 0.0, box)


def on_red(bgr, box):
    """True when the area around a word is red, as on a real stop sign."""
    x, y, w, h = box
    area = bgr[max(0, y - h):y + 2 * h, max(0, x - h // 2):x + w + h // 2].astype(np.int16)
    if area.size == 0:
        return False
    redness = area[:, :, 2] - np.maximum(area[:, :, 0], area[:, :, 1])
    return (redness > 50).mean() > 0.3


def find_word(bgr):
    """Return (x, y, w, h) of the word STOP, or None. Handles dark or light letters."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for source in (gray, cv2.bitwise_not(gray)):
        shapes, labels = letter_shapes(source)
        groups = sorted(letter_groups(shapes),
                        key=lambda g: sum(s[2] * s[3] for s in g), reverse=True)
        for group in groups[:cfg.WORD_MAX_BLOCKS]:
            word, confidence, box = read_group(group, labels)
            if not is_stop(word):
                continue
            if confidence >= cfg.WORD_MIN_CONFIDENCE or (
                    confidence >= cfg.WORD_MIN_CONFIDENCE_ON_RED and on_red(bgr, box)):
                return box
    return None


def distance_m(box):
    """Distance from the camera, from how tall the letters look."""
    return cfg.CAMERA_F_PX * cfg.WORD_HEIGHT_M / box[3]


class SignWatcher:
    def __init__(self, camera):
        self.camera = camera
        self.reads = []         # (frame time, distance) for each time the word was read
        self.box = None         # latest word position and its frame time
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
            try:
                box = find_word(frame)
            except Exception as error:      # OCR missing or failed
                print(f"WARNING: sign reading failed: {error}")
                self.ok = False
                return
            if box is not None:
                self.box = (box, stamp, frame.shape[1])
                self._record(stamp, distance_m(box))

    def _record(self, stamp, distance):
        self.reads = [r for r in self.reads
                      if stamp - r[0] <= cfg.SIGN_CONFIRM_WINDOW_S] + [(stamp, distance)]
        if len(self.reads) >= cfg.SIGN_CONFIRM_READS:
            self.result = (stamp, distance)

    def lane_ignore(self, now):
        """Return the word's area at lane-detection size, so a red sign isn't taken for tape."""
        if self.box is None or now - self.box[1] > 2.0:
            return None
        (x, y, w, h), _, width = self.box
        scale = cfg.PROC_WIDTH / width
        margin = 2 * h
        return (int((x - margin) * scale), int((y - margin) * scale),
                int((w + 2 * margin) * scale), int((h + 2 * margin) * scale))

    def confirmed(self):
        """Return (time seen, distance) for the latest confirmed sign, or None."""
        return self.result

    def close(self):
        self._running = False
        self._thread.join(timeout=5)
