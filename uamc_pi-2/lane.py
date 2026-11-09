"""Finds the lane centre by scanning rows of the image from near to far.

No bird's-eye transform is needed. Expected lane width is fitted against image
row from rows that show both lines, then used on rows that show only one.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

import config as cfg


@dataclass
class LaneResult:
    error: float = 0.0          # -1 lane is far left, +1 lane is far right
    confidence: float = 0.0     # fraction of scan rows that found the lane
    centers: list = field(default_factory=list)   # (x, y) points, for the overlay


def line_mask(bgr):
    """White wherever a pixel looks like tape."""
    if cfg.LINE_COLOR == "red":
        b, g, r = cv2.split(cv2.GaussianBlur(bgr, (3, 3), 0))
        score = cv2.subtract(r, cv2.max(g, b))
        _, mask = cv2.threshold(score, cfg.RED_MARGIN, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    local = cv2.blur(gray, (61, 61))
    diff = cv2.subtract(gray, local) if cfg.LINE_IS_BRIGHT else cv2.subtract(local, gray)
    _, mask = cv2.threshold(diff, cfg.LINE_CONTRAST, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def line_centers(row):
    """X position of each run of white pixels of a plausible tape width."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], row > 0, [0])).astype(np.int8)))
    return [(a + b - 1) / 2 for a, b in zip(edges[::2], edges[1::2])
            if cfg.MIN_LINE_PX <= b - a <= cfg.MAX_LINE_PX]


def nearest(lines, target, gate):
    """The line closest to target, if one lies within gate pixels."""
    best = min(lines, key=lambda v: abs(v - target), default=None)
    return best if best is not None and abs(best - target) <= gate else None


class LaneDetector:
    def __init__(self, width=cfg.PROC_WIDTH, height=cfg.PROC_HEIGHT):
        self.w = width
        self.rows = (np.linspace(cfg.ROW_NEAR, cfg.ROW_FAR, cfg.NUM_ROWS)
                     * height).astype(int)
        # On a flat floor, lane width in pixels is a straight line against row:
        # width = slope * y + offset. Near rows often show only one line, so
        # their width has to come from this fit, not from a direct measurement.
        near, far = float(self.rows[0]), float(self.rows[-1])
        self.slope = (cfg.LANE_WIDTH_NEAR - cfg.LANE_WIDTH_FAR) * width / (near - far)
        self.offset = cfg.LANE_WIDTH_NEAR * width - self.slope * near
        self.fitted = False     # True once the width has been measured from the image
        self.start_x = width / 2
        # Near rows steer more, so a bend far ahead doesn't start the turn early.
        self.weights = np.linspace(1.0, cfg.FAR_WEIGHT, cfg.NUM_ROWS)
        # Lines are followed from frame to frame, row by row, so one visible line
        # keeps its identity instead of being guessed from which side it is on.
        self.left = np.full(cfg.NUM_ROWS, np.nan)
        self.right = np.full(cfg.NUM_ROWS, np.nan)
        self.missed = np.full(cfg.NUM_ROWS, cfg.TRACK_MEMORY)

    @property
    def lane_w(self):
        return np.maximum(10.0, self.slope * self.rows + self.offset)

    def _update_width(self, samples):
        if len(samples) < cfg.MIN_WIDTH_ROWS:
            return
        ys, widths = np.array(samples, dtype=float).T
        if ys.max() - ys.min() < 15:
            return
        slope, offset = np.polyfit(ys, widths, 1)
        if slope <= 0:
            return
        blend = 0.1 if self.fitted else 1.0
        self.slope += blend * (slope - self.slope)
        self.offset += blend * (offset - self.offset)
        self.fitted = True

    def detect(self, bgr, ignore=None):
        """Find the lane. ignore is an (x, y, w, h) box to leave out, such as a sign."""
        mask = line_mask(bgr)
        if ignore is not None:
            x0, y0, w0, h0 = ignore
            mask[max(0, y0):y0 + h0, max(0, x0):x0 + w0] = 0
        x = self.start_x
        shift = 0.0             # how far the lane has moved this frame, from nearer rows
        centers, weights, samples = [], [], []
        low, high = (0.5, 1.6) if self.fitted else (0.3, 3.0)
        for i, (y, expected) in enumerate(zip(self.rows, self.lane_w)):
            lines = line_centers(mask[y])
            tracked = self.missed[i] < cfg.TRACK_MEMORY
            if tracked:
                want_l, want_r = self.left[i] + shift, self.right[i] + shift
                gate = max(25.0, 0.2 * expected)
            else:
                want_l, want_r = x - expected / 2, x + expected / 2
                gate = 0.45 * expected
            left = nearest(lines, want_l, gate)
            right = nearest(lines, want_r, gate)
            if left is not None and left == right:
                if abs(left - want_l) <= abs(right - want_r):
                    right = None
                else:
                    left = None
            if left is not None and right is not None:
                measured = right - left
                if low * expected < measured < high * expected:
                    samples.append((y, measured))
            elif left is not None:
                right = left + expected
            elif right is not None:
                left = right - expected
            else:
                self.missed[i] += 1
                continue
            if tracked:
                shift = ((left - self.left[i]) + (right - self.right[i])) / 2
            self.left[i], self.right[i], self.missed[i] = left, right, 0
            x = (left + right) / 2
            centers.append((x, int(y)))
            weights.append(self.weights[i])

        self._update_width(samples)
        if not centers:
            if self.missed.min() >= cfg.TRACK_MEMORY:
                self.start_x = self.w / 2
            return LaneResult(), mask

        self.start_x = centers[0][0]
        mean_x = np.average([c[0] for c in centers], weights=weights)
        error = (mean_x - self.w / 2) / (self.w / 2) - cfg.CENTER_TRIM
        return LaneResult(float(error), len(centers) / cfg.NUM_ROWS, centers), mask


def draw_overlay(bgr, result, text=""):
    out = bgr.copy()
    h, w = out.shape[:2]
    cv2.line(out, (w // 2, h - 1), (w // 2, int(h * 0.5)), (255, 255, 255), 1)
    for x, y in result.centers:
        cv2.circle(out, (int(x), y), 3, (0, 255, 0), -1)
    if text:
        cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return out
