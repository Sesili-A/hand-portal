"""Hand helpers: finger states, fist, pinch, and on-screen 'hover to click' buttons."""
import math

import cv2
import numpy as np

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
TIPS, PIPS = (8, 12, 16, 20), (6, 10, 14, 18)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Hand:
    """One detected hand in pixel coordinates."""

    PINCH_ON, PINCH_OFF = 0.33, 0.50  # thumb-index distance / palm size (with hysteresis)

    def __init__(self, pts, label):
        self.pts = pts
        self.label = label
        self.size = max(dist(pts[WRIST], pts[MIDDLE_MCP]), 1.0)
        # index, middle, ring, pinky: is the fingertip further from the wrist than its middle joint?
        self.up = [dist(pts[t], pts[WRIST]) > dist(pts[p], pts[WRIST]) * 1.1 for t, p in zip(TIPS, PIPS)]
        self.pinch_ratio = dist(pts[THUMB_TIP], pts[INDEX_TIP]) / self.size

    @property
    def index_tip(self):
        return self.pts[INDEX_TIP]

    @property
    def pinch_point(self):
        (x1, y1), (x2, y2) = self.pts[THUMB_TIP], self.pts[INDEX_TIP]
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def is_fist(self):
        return not any(self.up)

    @property
    def is_pointing(self):
        """Only the index finger is up."""
        return self.up[0] and not self.up[1] and not self.up[2]


def build_hands(result, w, h):
    hands, seen = [], set()
    for i, lms in enumerate(result.hand_landmarks):
        label = result.handedness[i][0].category_name if result.handedness else str(i)
        if label in seen:  # the model occasionally labels both hands the same
            label = f"{label}{i}"
        seen.add(label)
        hands.append(Hand([(int(p.x * w), int(p.y * h)) for p in lms], label))
    return hands


def draw_skeleton(img, hand):
    for i, j in HAND_CONNECTIONS:
        cv2.line(img, hand.pts[i], hand.pts[j], (230, 230, 230), 1, cv2.LINE_AA)
    for k in (THUMB_TIP, INDEX_TIP):
        cv2.circle(img, hand.pts[k], 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, hand.pts[k], 9, (255, 0, 255), 2, cv2.LINE_AA)


class HoldCounter:
    """Fires once after a condition has been true for `frames` frames in a row."""

    def __init__(self, frames):
        self.frames, self.count = frames, 0

    def __call__(self, condition):
        self.count = self.count + 1 if condition else 0
        return self.count == self.frames


class Smoother:
    """Exponential smoothing so points don't jitter."""

    def __init__(self, alpha=0.5):
        self.alpha, self.value = alpha, None

    def __call__(self, pts):
        pts = np.asarray(pts, np.float32)
        if self.value is None or self.value.shape != pts.shape:
            self.value = pts
        else:
            self.value = self.alpha * pts + (1 - self.alpha) * self.value
        return self.value.astype(np.int32)

    def reset(self):
        self.value = None


class DwellButton:
    """A button you 'click' by holding your index fingertip over it."""

    def __init__(self, rect, label, color=(70, 70, 70), dwell=0.8):
        self.rect, self.label, self.color, self.dwell = rect, label, color, dwell
        self.since, self.fired = None, False

    def contains(self, p):
        x, y, w, h = self.rect
        return x <= p[0] <= x + w and y <= p[1] <= y + h

    def update(self, pointers, now):
        if any(self.contains(p) for p in pointers):
            if self.since is None:
                self.since = now
            elif not self.fired and now - self.since >= self.dwell:
                self.fired = True
                return True
        else:
            self.since, self.fired = None, False
        return False

    def draw(self, img, now, active=False):
        x, y, w, h = self.rect
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.color, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        border = (255, 255, 255) if active else (150, 150, 150)
        cv2.rectangle(img, (x, y), (x + w, y + h), border, 3 if active else 1, cv2.LINE_AA)
        txt_color = (0, 0, 0) if sum(self.color) > 450 else (255, 255, 255)
        (tw, th), _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(img, self.label, (x + (w - tw) // 2, y + (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    txt_color, 1, cv2.LINE_AA)
        if self.since is not None and not self.fired:
            frac = min((now - self.since) / self.dwell, 1.0)
            cv2.rectangle(img, (x, y + h - 6), (x + int(w * frac), y + h), (0, 255, 120), -1)
