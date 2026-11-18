"""Gesture logic tests for the Python app: simulated hands, no camera needed."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the hand-portal folder
OUT = os.path.join(HERE, "tests", "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from gestures import Hand
from modes import PortalMode, DrawMode, DragMode
from filters import FILTERS


def hand(label, thumb, index, up=(True, False, False, False), wrist=None):
    """Fake 21-point hand: wrist far below, chosen finger states."""
    cx, cy = index
    wrist = wrist or (cx, cy + 200)
    pts = [wrist] * 21
    pts[9] = (wrist[0], wrist[1] - 100)  # middle mcp -> size 100
    for t, p, u in zip((8, 12, 16, 20), (6, 10, 14, 18), up):
        pts[p] = (wrist[0], wrist[1] - 120)
        pts[t] = (wrist[0], wrist[1] - (180 if u else 60))
    pts[8] = index if up[0] else pts[8]
    pts[4] = thumb
    return Hand(pts, label)

class FakeApp:
    here = HERE
    def __init__(self):
        self.msgs = []
        self.face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(HERE, "face_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE, num_faces=2))
        self.frame = None
    def toast(self, m, secs=1.3): self.msgs.append(m)
    def faces(self, frame):
        h, w = frame.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self.face_lm.detect(img)
        return [[(int(p.x * w), int(p.y * h)) for p in f] for f in res.face_landmarks]

app = FakeApp()
frame = np.full((720, 1280, 3), 90, np.uint8)
ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name); ok &= bool(cond)

# 1) pinch switches filter in portal
pm = PortalMode(app); t = 100.0
pinch = hand("Right", (600, 400), (610, 400))  # thumb next to index -> ratio 0.1
for i in range(5): pm.process(frame, [pinch], t + i * 0.05)          # 0.2 s: not yet
check("pinch not fired before hold", pm.idx == 0)
for i in range(5, 12): pm.process(frame, [pinch], t + i * 0.05)      # 0.55 s held
check("pinch hold -> next filter", pm.idx == 1)
for i in range(12, 40): pm.process(frame, [pinch], t + i * 0.05)     # keep holding
check("held pinch fires only once", pm.idx == 1)
openh = hand("Right", (500, 300), (700, 500))
pm.process(frame, [openh], t + 3);
for i in range(12): pm.process(frame, [pinch], t + 3.1 + i * 0.05)
check("release + pinch again -> next", pm.idx == 2)
# fist still works
fist = hand("Right", (600, 400), (600, 400), up=(False, False, False, False))
for i in range(8): pm.process(frame, [fist], t + 10 + i * 0.05)
check("fist -> next filter", pm.idx == 3)
check("fist does not count as pinch", not any(pm.pinch_was.values()))

# 2) face filter inside portal, on a real video frame
cap = cv2.VideoCapture(os.path.join(HERE, "..", "cv2.mp4"))
found = None
for k in range(0, 200, 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, k); r, f = cap.read()
    if not r: break
    f = cv2.resize(f, (1280, int(1280 * f.shape[0] / f.shape[1])))
    if app.faces(f): found = f; break
if found is None:
    cap = cv2.VideoCapture(os.path.join(HERE, "..", "cv1.mp4"))
    for k in range(0, 200, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); r, f = cap.read()
        if not r: break
        f = cv2.resize(f, (1280, int(1280 * f.shape[0] / f.shape[1])))
        if app.faces(f): found = f; break
if found is None:
    print("SKIP face test: no face in sample videos")
else:
    H, W = found.shape[:2]
    face = app.faces(found)[0]
    xs, ys = [p[0] for p in face], [p[1] for p in face]
    pm.idx = [n for n, *_ in FILTERS].index("Shades")
    a = hand("Left", (min(xs) - 40, max(ys) + 40), (min(xs) - 40, min(ys) - 40))
    b = hand("Right", (max(xs) + 40, max(ys) + 40), (max(xs) + 40, min(ys) - 40))
    out = pm.process(found, [a, b], 200.0)
    inside = cv2.absdiff(out, found)[min(ys):max(ys), min(xs):max(xs)].mean()
    check(f"face filter draws on face inside portal (diff {inside:.1f})", inside > 3)
    cv2.imwrite(os.path.join(OUT, "portal_face.png"), out)

# 3) draw in the air
dm = DrawMode(app)
for i in range(10):
    dm.process(frame, [hand("Right", (300, 600), (300 + i * 30, 400))], 300 + i * 0.05)
check("pointing draws ink", dm.canvas.any())

# 4) pinch drag + two-hand resize
gm = DragMode(app); gm.process(frame, [], 400)
it = gm.items[0]; x0, s0 = it.cx, it.scale
cx, cy = int(it.cx), int(it.cy)
p = lambda lbl, x, y: hand(lbl, (x - 3, y), (x + 3, y))
gm.process(frame, [hand("Right", (cx - 100, cy), (cx + 100, cy))], 400.1)
for i in range(1, 6): gm.process(frame, [p("Right", cx + i * 20, cy)], 400.1 + i * 0.05)
check("pinch drags item", abs(it.cx - (x0 + 80)) < 2)
cx = int(it.cx)
gm.process(frame, [p("Right", cx + 10, cy), p("Left", cx - 10, cy)], 401)
gm.process(frame, [p("Right", cx + 40, cy), p("Left", cx - 40, cy)], 401.05)
check(f"two-hand pinch resizes ({s0:.2f} -> {it.scale:.2f})", it.scale > s0 * 2)

print("ALL PASS" if ok else "SOME FAILED")
