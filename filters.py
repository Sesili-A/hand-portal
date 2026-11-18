"""Image filters used inside the portal.

Two kinds:
  * "roi"  filters get just the cropped image region:  fn(img, t) -> img
  * "face" filters also get face points:              fn(img, t, faces) -> img
           (faces = list of 478 (x, y) points per face, in img's coordinates;
            points may lie outside img when the face is only partly inside)
"""
import math

import cv2
import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarksConnections as FLC

TESSELATION = [(c.start, c.end) for c in FLC.FACE_LANDMARKS_TESSELATION]
FACE_OVAL = [c.start for c in FLC.FACE_LANDMARKS_FACE_OVAL]


def _blend_mask(fg, bg, mask):
    """Mix fg over bg using a 0-255 single-channel mask (fast, uint8 only)."""
    m3 = cv2.merge([mask, mask, mask])
    return cv2.add(cv2.multiply(fg, m3, scale=1 / 255), cv2.multiply(bg, 255 - m3, scale=1 / 255))


# ----------------------------------------------------------------- ROI filters
def neon_edges(img, t):
    """Rainbow neon edges (cv1, first effect)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 110)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    h, w = gray.shape
    hue_row = ((np.arange(w) // 3 + int(t * 120)) % 180).astype(np.int16)
    hue_col = (np.arange(h) // 3).astype(np.int16)
    hue = ((hue_col[:, None] + hue_row[None, :]) % 180).astype(np.uint8)
    full = np.full_like(hue, 255)
    rainbow = cv2.cvtColor(cv2.merge([hue, full, full]), cv2.COLOR_HSV2BGR)
    mask = cv2.max(edges, cv2.GaussianBlur(edges, (0, 0), 3))
    return _blend_mask(rainbow, cv2.convertScaleAbs(img, alpha=0.25), mask)


def line_art(img, t):
    """Purple line drawing on dark background (cv1, triangle effect)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 90)
    mask = cv2.max(edges, cv2.GaussianBlur(edges, (0, 0), 2.5))
    return _blend_mask(np.full_like(img, (255, 60, 200)), np.full_like(img, (40, 10, 20)), mask)


def thermal(img, t):
    """Pink/purple heat-map look (cv1, last effect)."""
    return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_PLASMA)


def purple_glass(img, t):
    """Solid tinted glass (cv2 purple portal)."""
    return cv2.addWeighted(img, 0.35, np.full_like(img, (160, 20, 130)), 0.65, 0)


def pixelate(img, t, block=18):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def invert(img, t):
    return 255 - img


def glitch(img, t):
    """RGB split + scanlines."""
    shift = 6 + int(4 * math.sin(t * 8))
    b, g, r = cv2.split(img)
    out = cv2.merge([np.roll(b, -shift, axis=1), g, np.roll(r, shift, axis=1)])
    out[::3] = (out[::3] * 0.55).astype(np.uint8)
    return out


def sketch(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(255 - gray, (21, 21), 0)
    return cv2.cvtColor(cv2.divide(gray, 255 - blur, scale=256), cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------- face filters
def cyber_mesh(frame, t, faces):
    """Dark frame with a glowing wireframe over every face."""
    out = cv2.convertScaleAbs(frame, alpha=0.35)
    lines = np.zeros_like(frame)
    pulse = 0.5 + 0.5 * math.sin(t * 4)
    color = (255, int(80 + 120 * pulse), 255)
    for pts in faces:
        for a, b in TESSELATION:
            cv2.line(lines, pts[a], pts[b], color, 1, cv2.LINE_AA)
    return cv2.add(cv2.add(out, lines), cv2.GaussianBlur(lines, (0, 0), 4))


def shades(frame, t, faces):
    """Cartoon sunglasses + moustache that follow your face."""
    out = frame.copy()
    for p in faces:
        p = np.asarray(p, np.float32)
        right_eye = (p[33] + p[133]) / 2
        left_eye = (p[263] + p[362]) / 2
        eye_dist = float(np.linalg.norm(left_eye - right_eye))
        ang = math.degrees(math.atan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
        axes = (int(eye_dist * 0.42), int(eye_dist * 0.30))
        for c in (right_eye, left_eye):
            c = tuple(int(v) for v in c)
            cv2.ellipse(out, c, axes, ang, 0, 360, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.ellipse(out, c, axes, ang, 0, 360, (0, 215, 255), 3, cv2.LINE_AA)
            hl = (int(c[0] - axes[0] * 0.35), int(c[1] - axes[1] * 0.35))
            cv2.ellipse(out, hl, (axes[0] // 4, axes[1] // 6), ang - 20, 0, 360, (230, 230, 230), -1, cv2.LINE_AA)
        # bridge + arms
        ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        r_in = (int(right_eye[0] + ux * axes[0]), int(right_eye[1] + uy * axes[0]))
        l_in = (int(left_eye[0] - ux * axes[0]), int(left_eye[1] - uy * axes[0]))
        cv2.line(out, r_in, l_in, (0, 215, 255), 4, cv2.LINE_AA)
        r_out = (int(right_eye[0] - ux * axes[0]), int(right_eye[1] - uy * axes[0]))
        l_out = (int(left_eye[0] + ux * axes[0]), int(left_eye[1] + uy * axes[0]))
        cv2.line(out, r_out, (int(r_out[0] - ux * eye_dist * 0.35), int(r_out[1] - uy * eye_dist * 0.35)),
                 (0, 215, 255), 4, cv2.LINE_AA)
        cv2.line(out, l_out, (int(l_out[0] + ux * eye_dist * 0.35), int(l_out[1] + uy * eye_dist * 0.35)),
                 (0, 215, 255), 4, cv2.LINE_AA)
        # moustache: two curls between nose and upper lip
        m = (p[164] * 0.6 + p[0] * 0.4)
        mw = eye_dist * 0.38
        for side in (-1, 1):
            c = (int(m[0] + side * ux * mw * 0.55), int(m[1] + side * uy * mw * 0.55))
            cv2.ellipse(out, c, (int(mw * 0.6), int(mw * 0.22)), ang + side * 12, 0, 360, (25, 25, 40), -1,
                        cv2.LINE_AA)
            tip = (int(m[0] + side * ux * mw * 1.15), int(m[1] + side * uy * mw * 1.15 - mw * 0.25))
            cv2.circle(out, tip, max(2, int(mw * 0.13)), (25, 25, 40), -1, cv2.LINE_AA)
    return out


def face_pixel(frame, t, faces):
    """Censor only the face (cv2 pixel effect)."""
    out = frame.copy()
    for pts in faces:
        hull = np.array([pts[i] for i in FACE_OVAL], np.int32)
        x, y, w, h = cv2.boundingRect(hull)
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        if w < 4 or h < 4:
            continue
        roi = out[y:y + h, x:x + w]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [hull - (x, y)], 255)
        out[y:y + h, x:x + w] = _blend_mask(pixelate(roi, t, block=max(6, w // 10)), roi, mask)
    return out


# name, kind, function, border colour (BGR)
FILTERS = [
    ("Neon Edges", "roi", neon_edges, (255, 100, 255)),
    ("Line Art", "roi", line_art, (255, 60, 200)),
    ("Thermal", "roi", thermal, (180, 80, 255)),
    ("Cyber Mesh", "face", cyber_mesh, (255, 200, 255)),
    ("Shades", "face", shades, (0, 215, 255)),
    ("Purple Glass", "roi", purple_glass, (255, 255, 255)),
    ("Pixelate", "roi", pixelate, (255, 255, 255)),
    ("Face Pixel", "face", face_pixel, (255, 255, 255)),
    ("Glitch", "roi", glitch, (255, 255, 0)),
    ("Invert", "roi", invert, (0, 255, 255)),
    ("Sketch", "roi", sketch, (255, 255, 255)),
]
