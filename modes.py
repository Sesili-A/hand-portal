"""The three app modes: Portal, Draw and Drag."""
import glob
import math
import os

import cv2
import numpy as np

from filters import FILTERS
from gestures import DwellButton, HoldCounter, Smoother, THUMB_TIP, INDEX_TIP

BAR_H = 70  # height of the top button bar; drawing/grabbing is ignored up here
FONT = cv2.FONT_HERSHEY_SIMPLEX


# =================================================================== PORTAL
class PortalMode:
    name = "Portal"
    hint = "Frame with thumbs + index fingers  |  hold a PINCH or make a FIST to change filter"
    PINCH_HOLD = 0.35  # seconds a pinch must be held to switch filter

    def __init__(self, app):
        self.app = app
        self.idx = 0
        self.smoother = Smoother(0.55)
        self.fist = HoldCounter(6)
        self.cooldown = 0.0
        self.pinch_since = {}  # hand label -> time the current pinch started (None once it has fired)
        self.pinch_was = {}  # hand label -> pinched last frame (for hysteresis)
        self.buttons = []

    def _pinch_switch(self, hands, now):
        """Hold thumb + index together (not a fist) -> next filter. Fires once per pinch."""
        fired = False
        labels = {hd.label for hd in hands}
        for label in list(self.pinch_was):
            if label not in labels:
                self.pinch_was.pop(label)
                self.pinch_since.pop(label, None)
        for hd in hands:
            was = self.pinch_was.get(hd.label, False)
            pinched = not hd.is_fist and hd.pinch_ratio < (hd.PINCH_OFF if was else hd.PINCH_ON)
            self.pinch_was[hd.label] = pinched
            if not pinched:
                self.pinch_since.pop(hd.label, None)
            elif not was:
                self.pinch_since[hd.label] = now
            else:
                since = self.pinch_since.get(hd.label)
                if since is not None and now - since >= self.PINCH_HOLD:
                    self.pinch_since[hd.label] = None  # wait for release before firing again
                    fired = True
        return fired

    def set_filter(self, idx):
        self.idx = idx % len(FILTERS)
        self.app.toast(f"Filter: {FILTERS[self.idx][0]}")

    def handle_key(self, key):
        if ord("1") <= key <= ord("9"):
            self.set_filter(key - ord("1"))
        elif key == ord("0"):
            self.set_filter(9)
        elif key == ord("n"):
            self.set_filter(self.idx + 1)
        elif key == ord("b"):
            self.set_filter(self.idx - 1)

    @staticmethod
    def polygon(hands):
        if len(hands) >= 2:
            a, b = sorted(hands[:2], key=lambda hd: hd.pts[0][0])  # left -> right on screen
            # crossing your hands naturally gives the bow-tie shape from cv2
            return [a.pts[INDEX_TIP], b.pts[INDEX_TIP], b.pts[THUMB_TIP], a.pts[THUMB_TIP]]
        if len(hands) == 1:
            (x1, y1), (x2, y2) = hands[0].pts[THUMB_TIP], hands[0].pts[INDEX_TIP]
            if abs(x1 - x2) > 25 and abs(y1 - y2) > 25:
                return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return None

    def process(self, frame, hands, now):
        # gestures: hold a fist or a pinch briefly -> next filter
        fist = self.fist(any(hd.is_fist for hd in hands))
        pinch = self._pinch_switch(hands, now)
        if (fist or pinch) and now > self.cooldown:
            self.set_filter(self.idx + 1)
            self.cooldown = now + 0.8

        out = frame.copy()
        poly = self.polygon([hd for hd in hands if not hd.is_fist])
        if poly is None:
            self.smoother.reset()
            return out

        name, kind, fn, edge_color = FILTERS[self.idx]
        poly = self.smoother(poly)
        h, w = frame.shape[:2]
        x, y, bw, bh = cv2.boundingRect(poly)
        x, y = max(x, 0), max(y, 0)
        bw, bh = min(bw, w - x), min(bh, h - y)
        if bw < 3 or bh < 3:
            return out

        roi = frame[y:y + bh, x:x + bw]
        if kind == "face":
            # only process the portal's box: shift face points into its coordinates
            faces = [[(px - x, py - y) for px, py in pts] for pts in self.app.faces(frame)]
            filtered = fn(roi, now, faces)
        else:
            filtered = fn(roi, now)

        mask = np.zeros((bh, bw), np.uint8)
        cv2.fillPoly(mask, [poly - (x, y)], 255, cv2.LINE_AA)
        m3 = cv2.merge([mask, mask, mask])
        out[y:y + bh, x:x + bw] = cv2.add(cv2.multiply(filtered, m3, scale=1 / 255),
                                          cv2.multiply(roi, 255 - m3, scale=1 / 255))
        # glowing border, blurred only around the portal (a full-frame blur is slow)
        pad = 24
        gx0, gy0 = max(x - pad, 0), max(y - pad, 0)
        gx1, gy1 = min(x + bw + pad, w), min(y + bh + pad, h)
        glow = np.zeros((gy1 - gy0, gx1 - gx0, 3), np.uint8)
        cv2.polylines(glow, [poly - (gx0, gy0)], True, edge_color, 6, cv2.LINE_AA)
        out[gy0:gy1, gx0:gx1] = cv2.add(out[gy0:gy1, gx0:gx1], cv2.GaussianBlur(glow, (0, 0), 6))
        cv2.polylines(out, [poly], True, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def draw_ui(self, img, hands, now):
        cv2.putText(img, f"Filter {self.idx + 1}/{len(FILTERS)}: {FILTERS[self.idx][0]}", (15, 45), FONT, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        # progress ring while a pinch is being held
        for hd in hands:
            since = self.pinch_since.get(hd.label)
            if since is not None:
                frac = min((now - since) / self.PINCH_HOLD, 1.0)
                cv2.ellipse(img, hd.pinch_point, (18, 18), -90, 0, int(360 * frac), (0, 255, 120), 3, cv2.LINE_AA)


# =================================================================== DRAW
COLORS = [("Pink", (255, 0, 255)), ("Cyan", (255, 255, 0)), ("Lime", (0, 255, 120)),
          ("Gold", (0, 215, 255)), ("Red", (60, 60, 255)), ("White", (255, 255, 255))]


class DrawMode:
    name = "Draw"
    hint = "Point INDEX finger to draw  |  index+middle up = move without drawing  |  hold FIST = clear"

    def __init__(self, app):
        self.app = app
        self.canvas = None
        self.color = COLORS[0][1]
        self.eraser = False
        self.thickness = 8
        self.prev = {}
        self.smooth = {}
        self.fist = HoldCounter(18)
        self.buttons = []
        x = 10
        for name, col in COLORS:
            self.buttons.append(DwellButton((x, 10, 62, 50), name, col, dwell=0.5))
            x += 68
        self.btn_eraser = DwellButton((x, 10, 72, 50), "Eraser", (40, 40, 40), dwell=0.5)
        self.btn_clear = DwellButton((x + 78, 10, 64, 50), "Clear", (30, 30, 120), dwell=0.8)
        self.buttons += [self.btn_eraser, self.btn_clear]

    def clear(self):
        if self.canvas is not None:
            self.canvas[:] = 0
        self.app.toast("Canvas cleared")

    def handle_key(self, key):
        if key == ord("c"):
            self.clear()
        elif key == ord("e"):
            self.eraser = not self.eraser
        elif key in (ord("+"), ord("=")):
            self.thickness = min(self.thickness + 2, 40)
        elif key == ord("-"):
            self.thickness = max(self.thickness - 2, 2)

    def on_button(self, btn):
        if btn is self.btn_clear:
            self.clear()
        elif btn is self.btn_eraser:
            self.eraser = True
            self.app.toast("Eraser")
        else:
            self.color, self.eraser = btn.color, False
            self.app.toast(f"Colour: {btn.label}")

    def process(self, frame, hands, now):
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)
        if self.fist(any(hd.is_fist for hd in hands)):
            self.clear()

        active = set()
        for hd in hands:
            tip = tuple(int(v) for v in self.smooth.setdefault(hd.label, Smoother(0.5))(hd.index_tip))
            if hd.is_pointing and tip[1] > BAR_H:
                active.add(hd.label)
                prev = self.prev.get(hd.label)
                if prev is not None:
                    if self.eraser:
                        cv2.line(self.canvas, prev, tip, (0, 0, 0), self.thickness * 5, cv2.LINE_AA)
                    else:
                        cv2.line(self.canvas, prev, tip, self.color, self.thickness, cv2.LINE_AA)
                self.prev[hd.label] = tip
        for label in list(self.prev):
            if label not in active:
                del self.prev[label]
        for label in list(self.smooth):
            if label not in {hd.label for hd in hands}:
                del self.smooth[label]

        # glowing ink over the camera image
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        out = frame.copy()
        ink = gray > 0
        out[ink] = self.canvas[ink]
        # glow computed at half resolution (4x cheaper, looks the same)
        h, w = frame.shape[:2]
        small = cv2.resize(self.canvas, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3.5), (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.add(out, glow)

    def draw_ui(self, img, hands, now):
        for hd in hands:
            tip = hd.index_tip
            if self.eraser:
                cv2.circle(img, tip, self.thickness * 5 // 2, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(img, tip, self.thickness, self.color, -1 if hd.is_pointing else 2, cv2.LINE_AA)


# =================================================================== DRAG
class Item:
    def __init__(self, img, cx, cy, scale=1.0, angle=0.0):
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        self.img, self.cx, self.cy, self.scale, self.angle = img, float(cx), float(cy), scale, angle
        self.grab_sig = None  # which hands hold it, so we can re-anchor when that changes
        self.ref = None

    def matrix(self):
        h, w = self.img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), self.angle, self.scale)
        m[0, 2] += self.cx - w / 2
        m[1, 2] += self.cy - h / 2
        return m

    def corners(self):
        h, w = self.img.shape[:2]
        pts = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float32)
        return pts @ self.matrix().T

    def contains(self, p):
        h, w = self.img.shape[:2]
        u, v = cv2.invertAffineTransform(self.matrix()) @ np.array([p[0], p[1], 1.0])
        return 0 <= u < w and 0 <= v < h

    def render(self, dst):
        c = self.corners()
        H, W = dst.shape[:2]
        x0, y0 = max(int(c[:, 0].min()), 0), max(int(c[:, 1].min()), 0)
        x1, y1 = min(int(math.ceil(c[:, 0].max())), W), min(int(math.ceil(c[:, 1].max())), H)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        m = self.matrix()
        m[0, 2] -= x0
        m[1, 2] -= y0
        patch = cv2.warpAffine(self.img, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = patch[..., 3]
        region = dst[y0:y1, x0:x1]
        a3 = cv2.merge([a, a, a])
        dst[y0:y1, x0:x1] = cv2.add(cv2.multiply(patch[..., :3], a3, scale=1 / 255),
                                    cv2.multiply(region, 255 - a3, scale=1 / 255))


def _sticker(draw_fn, size=200):
    """Draw a glowing neon sticker on a transparent canvas."""
    color = np.zeros((size, size, 3), np.uint8)
    draw_fn(color)
    glow = cv2.GaussianBlur(color, (0, 0), 8)
    rgb = cv2.add(color, glow)
    alpha = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    alpha = np.clip(alpha.astype(np.int32) * 3, 0, 255).astype(np.uint8)
    return cv2.merge([*cv2.split(rgb), alpha])


def default_stickers():
    def star(img):
        c, pts = 100, []
        for i in range(10):
            r = 80 if i % 2 == 0 else 34
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((int(c + r * math.cos(a)), int(c + r * math.sin(a))))
        cv2.fillPoly(img, [np.array(pts)], (0, 215, 255), cv2.LINE_AA)

    def heart(img):
        cv2.circle(img, (70, 80), 42, (180, 40, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (130, 80), 42, (180, 40, 255), -1, cv2.LINE_AA)
        cv2.fillPoly(img, [np.array([(30, 98), (170, 98), (100, 175)])], (180, 40, 255), cv2.LINE_AA)

    def ring(img):
        cv2.circle(img, (100, 100), 70, (255, 255, 0), 14, cv2.LINE_AA)
        cv2.circle(img, (100, 100), 30, (255, 0, 255), -1, cv2.LINE_AA)

    def text(img):
        cv2.rectangle(img, (15, 60), (185, 140), (0, 255, 120), 4, cv2.LINE_AA)
        cv2.putText(img, "HELLO!", (28, 115), FONT, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    return [_sticker(f) for f in (star, heart, ring, text)]


def load_stickers(folder):
    """Any PNG/JPG you drop in the stickers/ folder becomes a draggable item."""
    imgs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jp*g"))):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        s = 240 / max(img.shape[:2])
        imgs.append(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    return imgs


class DragMode:
    name = "Drag"
    hint = "PINCH an item to drag it  |  pinch it with BOTH hands to resize + rotate  |  hover 'Snap' for a photo"

    def __init__(self, app):
        self.app = app
        self.items = None
        self.pinched = {}  # hand label -> bool
        self.grabs = {}  # hand label -> Item
        self.last_frame = None
        self.btn_snap = DwellButton((10, 10, 110, 50), "Snap photo", (90, 50, 20), dwell=0.8)
        self.btn_reset = DwellButton((128, 10, 80, 50), "Reset", (30, 30, 120), dwell=0.8)
        self.buttons = [self.btn_snap, self.btn_reset]

    def reset(self, w, h):
        imgs = load_stickers(os.path.join(self.app.here, "stickers")) + default_stickers()
        self.items = []
        n = len(imgs)
        for i, img in enumerate(imgs):
            cx = w * (i + 1) / (n + 1)
            cy = h * (0.45 if i % 2 == 0 else 0.7)
            self.items.append(Item(img, cx, cy, scale=0.8))
        self.grabs.clear()

    def snap(self):
        if self.last_frame is None:
            return
        h, w = self.last_frame.shape[:2]
        photo = cv2.resize(self.last_frame, (320, int(320 * h / w)))
        photo = cv2.copyMakeBorder(photo, 12, 50, 12, 12, cv2.BORDER_CONSTANT, value=(250, 250, 250))
        cv2.putText(photo, "hand-portal", (16, photo.shape[0] - 16), FONT, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
        self.items.append(Item(photo, w / 2, h / 2, scale=0.9, angle=-6))
        self.app.toast("Photo snapped - pinch it to move")

    def handle_key(self, key):
        if key == ord("p"):
            self.snap()
        elif key == ord("c") and self.last_frame is not None:
            self.reset(self.last_frame.shape[1], self.last_frame.shape[0])

    def on_button(self, btn):
        if btn is self.btn_snap:
            self.snap()
        elif btn is self.btn_reset and self.last_frame is not None:
            self.reset(self.last_frame.shape[1], self.last_frame.shape[0])
            self.app.toast("Items reset")

    def process(self, frame, hands, now):
        h, w = frame.shape[:2]
        self.last_frame = frame
        if self.items is None:
            self.reset(w, h)

        labels = {hd.label for hd in hands}
        for label in list(self.pinched):
            if label not in labels:
                del self.pinched[label]
                self.grabs.pop(label, None)

        points = {}
        for hd in hands:
            was = self.pinched.get(hd.label, False)
            now_pinched = hd.pinch_ratio < (hd.PINCH_OFF if was else hd.PINCH_ON)
            self.pinched[hd.label] = now_pinched
            if now_pinched:
                points[hd.label] = hd.pinch_point
                if not was:  # pinch just started: grab the top-most item under the fingers
                    for item in reversed(self.items):
                        if item.contains(hd.pinch_point):
                            self.items.remove(item)
                            self.items.append(item)
                            self.grabs[hd.label] = item
                            break
            else:
                self.grabs.pop(hd.label, None)

        held = {}
        for label, item in self.grabs.items():
            held.setdefault(id(item), (item, []))[1].append(label)
        for item in self.items:
            if id(item) not in held:
                item.grab_sig = None

        for item, hand_labels in held.values():
            sig = tuple(sorted(hand_labels))
            if len(sig) >= 2:
                p1, p2 = np.array(points[sig[0]], float), np.array(points[sig[1]], float)
                mid, d = (p1 + p2) / 2, max(np.linalg.norm(p2 - p1), 1.0)
                ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
                if item.grab_sig != sig:
                    item.grab_sig = sig
                    item.ref = (d, item.scale, ang, item.angle, np.array([item.cx, item.cy]) - mid)
                d0, s0, a0, item_a0, offset = item.ref
                item.scale = float(np.clip(s0 * d / d0, 0.15, 5.0))
                item.angle = item_a0 - (ang - a0)
                r = math.radians(ang - a0)
                rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
                item.cx, item.cy = mid + rot @ offset * (item.scale / s0)
            else:
                p = np.array(points[sig[0]], float)
                if item.grab_sig != sig:
                    item.grab_sig = sig
                    item.ref = np.array([item.cx, item.cy]) - p
                item.cx, item.cy = p + item.ref

        out = frame.copy()
        for item in self.items:
            item.render(out)
        return out

    def draw_ui(self, img, hands, now):
        for item in self.items or []:
            if item.grab_sig is not None:
                cv2.polylines(img, [item.corners().astype(np.int32)], True, (0, 255, 120), 2, cv2.LINE_AA)
        for hd in hands:
            p = hd.pinch_point
            if self.pinched.get(hd.label):
                cv2.circle(img, p, 12, (0, 255, 120), -1, cv2.LINE_AA)
            else:
                cv2.circle(img, p, 12, (255, 255, 255), 2, cv2.LINE_AA)
