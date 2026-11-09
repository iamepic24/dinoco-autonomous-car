"""Spots a zebra crossing: several evenly sized white bars side by side across the road."""
import cv2
import numpy as np

import config as cfg


def bright_runs(row):
    """Return (start, width) of each bright run in a row of pixels."""
    bright = row > row.mean() + cfg.ZEBRA_CONTRAST
    edges = np.flatnonzero(np.diff(np.concatenate(([0], bright, [0])).astype(np.int8)))
    return [(a, b - a) for a, b in zip(edges[::2], edges[1::2])]


def row_has_stripes(row):
    """True when a row crosses ZEBRA_MIN_BARS neighbouring bars of matching size and spacing."""
    runs = [r for r in bright_runs(row) if cfg.ZEBRA_MIN_BAR_PX <= r[1]]
    for i in range(len(runs) - cfg.ZEBRA_MIN_BARS + 1):
        bars = runs[i:i + cfg.ZEBRA_MIN_BARS]
        widths = [r[1] for r in bars]
        gaps = [b[0] - (a[0] + a[1]) for a, b in zip(bars, bars[1:])]
        if max(widths) <= 2.0 * min(widths) and max(gaps) <= 2.0 * max(1, min(gaps)):
            return True
    return False


class ZebraDetector:
    def __init__(self, height=cfg.PROC_HEIGHT):
        self.rows = (np.linspace(cfg.ZEBRA_ROW_NEAR, cfg.ZEBRA_ROW_FAR, 6) * height).astype(int)
        self.streak = 0

    def seen(self, bgr):
        """True once stripes have filled the band for ZEBRA_CONFIRM_FRAMES frames in a row."""
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        hits = sum(row_has_stripes(gray[y]) for y in self.rows)
        self.streak = self.streak + 1 if hits >= cfg.ZEBRA_MIN_ROWS else 0
        return self.streak >= cfg.ZEBRA_CONFIRM_FRAMES
