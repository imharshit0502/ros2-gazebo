"""
Classical (no training) shelf-void detector, tuned for the Gazebo retail store.

IDEA IN ONE SENTENCE
  Find each shelf level, then look along that level for a stretch that is
  "back-panel grey" from the shelf board all the way up to the next board AND
  is wider than a normal gap between two products.

HOW, STEP BY STEP (each step is a few lines of OpenCV below)
  1. Shelf rails. Every shelf level has a yellow price strip along its front
     edge. In the image these are long, THIN, horizontal yellow lines. We keep
     yellow pixels that are thin vertically and long horizontally, then look
     for image rows that contain a lot of them -> one row band per rail.
     This is the same "shelf-row band" idea as the DBSCAN step in the
     ShelfVoid notebook (cell 5), only taken from the rails instead of from
     product boxes.
  2. Row bands. The space between two neighbouring rails is one shelf level.
  3. "Empty" pixels. Inside a band, a pixel is empty if it is grey with almost no colour
     (low HSV saturation) and not too dark. The back panel and the shelf board
     look like that. Products have saturated packaging colours, and even the
     black/navy brands keep a slight colour tint, so they are excluded.
  4. Empty columns. A column of the band is empty if >= 85 % of its pixels
     are empty, so it is grey from the board below up to the board above.
     A short product with headroom above it is NOT an empty column.
  5. Width test. Neighbouring products always leave small gaps. A void is a
     run of empty columns wider than `min_width_ratio` x the band height.
     Band height ~ 0.40 m in the world, so 1.0 means "wider than ~40 cm"
     (gaps between products in this store are < 30 cm, voids are > 55 cm).
  6. Only columns that have a rail underneath count, so the floor, walls and
     the next aisle seen past the shelf end are never reported.

Parameters are scale free (ratios of the band height), so the detector keeps
working when the robot is a bit closer to or farther from the shelf.
"""

import cv2
import numpy as np

from .base import VoidBox, VoidDetector


def _runs(flags: np.ndarray, max_gap: int = 0):
    """[[start, end], ...] of consecutive True entries (gaps <= max_gap are bridged)."""
    out = []
    for i in np.flatnonzero(flags):
        if out and i - out[-1][1] <= max_gap + 1:
            out[-1][1] = int(i)
        else:
            out.append([int(i), int(i)])
    return out


def _close_1d(flags: np.ndarray, width: int) -> np.ndarray:
    """Morphological closing of a 1-D boolean array (fills holes narrower than `width`)."""
    k = np.ones((1, width), np.uint8)
    return cv2.morphologyEx(flags.astype(np.uint8)[None], cv2.MORPH_CLOSE, k)[0] > 0


class ShelfRailVoidDetector(VoidDetector):
    name = "classical"

    def __init__(self,
                 rail_hsv_low=(20, 80, 60), rail_hsv_high=(34, 255, 255),
                 rail_max_thickness_px=30, rail_min_row_fraction=0.30,
                 empty_max_saturation=20, empty_min_value=72,
                 empty_column_fraction=0.85, min_width_ratio=1.0,
                 min_band_height_frac=0.10, top_band_scale=1.15, expected_rails=4):
        self.expected_rails = int(expected_rails)   # shelf levels in the store (0..3)
        self.img_h = 0
        self.rail_lo = tuple(int(v) for v in rail_hsv_low)
        self.rail_hi = tuple(int(v) for v in rail_hsv_high)
        self.rail_max_thick = int(rail_max_thickness_px)
        self.rail_min_frac = float(rail_min_row_fraction)
        self.empty_max_s = int(empty_max_saturation)
        self.empty_min_v = int(empty_min_value)
        self.col_frac = float(empty_column_fraction)
        self.min_width_ratio = float(min_width_ratio)
        # A shelf level right in front of the robot is ~15-20 % of the image height.
        # Much thinner "levels" come from a FAR shelf (seen past a shelf end) whose rails
        # mix with ours. They are not on the shelf we are inspecting, so skip them.
        self.min_band_frac = float(min_band_height_frac)
        self.min_band_h = 30
        self.top_band_scale = float(top_band_scale)
        self._debug = None

    # ------------------------------------------------------------------ steps
    def _rail_mask(self, hsv):
        yellow = cv2.inRange(hsv, self.rail_lo, self.rail_hi)
        # remove yellow things TALLER than a rail (e.g. yellow product packs)
        tall = cv2.morphologyEx(yellow, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.rail_max_thick - 4)))
        thin = cv2.subtract(yellow, tall)
        # keep only LONG horizontal pieces
        return cv2.morphologyEx(thin, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (81, 1)))

    def _find_rails(self, rail_mask):
        row_frac = (rail_mask > 0).mean(axis=1)
        rails = _runs(row_frac > self.rail_min_frac, max_gap=20)
        return [r for r in rails if r[1] - r[0] <= self.rail_max_thick]

    def _bands(self, rails):
        """(top_row, bottom_row, rail_index) of the shelf level ABOVE each rail."""
        bands = []
        for k, (a, _b) in enumerate(rails):
            if k > 0:
                top = rails[k - 1][1] + 2
            else:
                # Top shelf level: nothing yellow above it. Use the spacing of the
                # rails below (perspective makes upper levels slightly taller).
                spacing = (rails[1][0] - rails[0][1]) if len(rails) > 1 else 120
                top = max(0, a - int(self.top_band_scale * spacing))
            bot = a - 2
            if bot - top >= self.min_band_h:
                bands.append((top, bot, k))
        # Bottom shelf level with an invisible rail. The deck's price strip sits exactly on
        # the deck's front face, so the renderer often hides it ("z-fighting"), and from
        # a moving robot we then see only the 3 upper rails. In that case add the level
        # BELOW the lowest rail, as tall as the level above it. Its columns are checked
        # against the rail above (has_rail uses rail index k).
        if 2 <= len(rails) < self.expected_rails and self.img_h:
            spacing = rails[-1][0] - rails[-2][1]
            top = rails[-1][1] + 2
            bot = min(self.img_h - 1, top + int(0.95 * spacing))
            if bot - top >= max(self.min_band_h, int(0.8 * spacing)):
                bands.append((top, bot, len(rails) - 1))
        return bands

    # ------------------------------------------------------------------ main
    def detect(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        self.img_h = bgr.shape[0]
        self.min_band_h = int(self.min_band_frac * self.img_h)
        rail_mask = self._rail_mask(hsv)
        rails = self._find_rails(rail_mask)
        bands = self._bands(rails)
        empty = ((hsv[..., 1] < self.empty_max_s) & (hsv[..., 2] > self.empty_min_v)).astype(np.uint8)

        boxes = []
        for top, bot, k in bands:
            band_h = bot - top
            a, b = rails[k]
            # columns that actually have shelf (a rail) under them
            has_rail = _close_1d((rail_mask[a:b + 1] > 0).any(axis=0), 15)
            frac = empty[top:bot].mean(axis=0)
            col_empty = _close_1d((frac > self.col_frac) & has_rail, 9)  # bridge thin uprights
            for x1, x2 in _runs(col_empty):
                width_ratio = (x2 - x1 + 1) / band_h
                if width_ratio < self.min_width_ratio:
                    continue
                fill = float(frac[x1:x2 + 1].mean())
                score = round(min(1.0, fill * min(1.0, width_ratio / 1.5)), 2)
                boxes.append(VoidBox(x1, top, x2, bot, score))

        self._debug = (bgr, empty, rail_mask, bands)
        return boxes

    def debug_image(self):
        """Grey = 'empty' pixels, yellow = detected rails, blue ticks = shelf levels."""
        if self._debug is None:
            return None
        bgr, empty, rail_mask, bands = self._debug
        vis = (bgr * 0.35).astype(np.uint8)
        vis[empty > 0] = (200, 200, 200)
        vis[rail_mask > 0] = (0, 220, 255)
        for top, bot, _ in bands:
            cv2.line(vis, (0, top), (40, top), (255, 80, 0), 3)
            cv2.line(vis, (0, bot), (40, bot), (255, 80, 0), 3)
        return vis
