"""
Hand Tracking Playground
------------------------
Three modes, all controlled with your hands. Switch modes by holding your
index fingertip over the buttons in the top-right corner (or press M).

  PORTAL  Frame a window with both hands; a filter is applied inside it.
          Hold a pinch or make a fist to change filter. Face filters (Cyber Mesh, Shades,
          Face Pixel) draw on your face, but only inside the portal.
  DRAW    Point your index finger to draw in the air. Raise index + middle
          to move without drawing. Hover the colour buttons to change colour.
          Hold a fist to clear.
  DRAG    Pinch a sticker to drag it. Pinch it with both hands to resize and
          rotate. Hover "Snap photo" to turn the camera image into a polaroid.
          Put your own PNGs in the stickers/ folder.

Keys:  M mode | S screenshot | R record mp4 | H hand skeleton | Q/Esc quit
       Portal: 1-9, 0, N/B filters   Draw: C clear, E eraser, +/- size
       Drag: P snap photo, C reset
"""
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from gestures import DwellButton, build_hands, draw_skeleton
from modes import DragMode, DrawMode, PortalMode

HERE = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(HERE, "hand_landmarker.task")
FACE_MODEL = os.path.join(HERE, "face_landmarker.task")
OUT_DIR = os.path.join(HERE, "captures")
FONT = cv2.FONT_HERSHEY_SIMPLEX


class App:
    def __init__(self):
        self.here = HERE
        self.hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))
        self.face_lm = None  # created the first time a face filter is used
        self.modes = [PortalMode(self), DrawMode(self), DragMode(self)]
        self.mode_idx = 0
        self.mode_buttons = []
        self.toast_msg, self.toast_until = "", 0.0
        self.show_hands = True
        self.ts = -1
        self._mp_image = None
        self._faces_cache = None

    # ------------------------------------------------------------ services for modes
    @property
    def mode(self):
        return self.modes[self.mode_idx]

    def toast(self, msg, secs=1.3):
        self.toast_msg, self.toast_until = msg, time.time() + secs

    def faces(self, frame):
        """Face landmarks for the current frame (computed at most once per frame)."""
        if self._faces_cache is None:
            if self.face_lm is None:
                self.face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=FACE_MODEL),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=2,
                ))
            if self.ts <= getattr(self, "_face_ts", -1):  # MediaPipe needs strictly increasing timestamps
                return self._faces_last
            self._face_ts = self.ts
            h, w = frame.shape[:2]
            res = self.face_lm.detect_for_video(self._mp_image, self.ts)
            self._faces_cache = [[(int(p.x * w), int(p.y * h)) for p in face] for face in res.face_landmarks]
            self._faces_last = self._faces_cache
        return self._faces_cache

    def set_mode(self, idx):
        self.mode_idx = idx % len(self.modes)
        self.toast(f"{self.mode.name} mode")

    # ------------------------------------------------------------ drawing
    def draw_overlay(self, img, hands, now, fps, recording):
        h, w = img.shape[:2]
        for i, btn in enumerate(self.mode_buttons):
            btn.draw(img, now, active=(i == self.mode_idx))
        for btn in self.mode.buttons:
            active = getattr(self.mode, "color", None) == btn.color and not getattr(self.mode, "eraser", True)
            btn.draw(img, now, active=active)
        self.mode.draw_ui(img, hands, now)
        if self.show_hands:
            for hd in hands:
                draw_skeleton(img, hd)

        # bottom hint bar
        cv2.rectangle(img, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(img, self.mode.hint, (10, h - 10), FONT, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, f"{fps:3.0f} fps", (w - 75, h - 10), FONT, 0.5, (120, 255, 120), 1, cv2.LINE_AA)
        if recording and int(now * 2) % 2 == 0:
            cv2.circle(img, (w - 95, h - 15), 7, (0, 0, 255), -1)
            cv2.putText(img, "REC", (w - 140, h - 10), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        if now < self.toast_until:
            (tw, th), _ = cv2.getTextSize(self.toast_msg, FONT, 1.0, 2)
            x, y = (w - tw) // 2, h // 2 + 200
            cv2.rectangle(img, (x - 16, y - th - 14), (x + tw + 16, y + 14), (0, 0, 0), -1)
            cv2.putText(img, self.toast_msg, (x, y), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------ main loop
    def run(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise SystemExit("Could not open the webcam.")

        win = "Hand Tracking Playground"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        start, prev, fps = time.time(), time.time(), 0.0
        writer = None

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera returned no image.")
                break
            frame = cv2.flip(frame, 1)  # mirror like a selfie camera
            h, w = frame.shape[:2]
            now = time.time()

            if not self.mode_buttons:
                # mode switcher: a column on the right edge, below the top bar
                self.mode_buttons = [DwellButton((w - 110, 80 + i * 60, 100, 50), m.name, (90, 30, 90))
                                     for i, m in enumerate(self.modes)]

            self.ts = max(int((now - start) * 1000), self.ts + 1)
            self._mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._faces_cache = None
            hands = build_hands(self.hand_lm.detect_for_video(self._mp_image, self.ts), w, h)

            # hover-to-click buttons use the tip of any raised index finger
            pointers = [hd.index_tip for hd in hands if hd.up[0]]
            for i, btn in enumerate(self.mode_buttons):
                if btn.update(pointers, now) and i != self.mode_idx:
                    self.set_mode(i)
            for btn in self.mode.buttons:
                if btn.update(pointers, now):
                    self.mode.on_button(btn)

            out = self.mode.process(frame, hands, now)
            if writer is not None:
                writer.write(out)  # recording has no buttons/hints

            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            display = out.copy()
            self.draw_overlay(display, hands, now, fps, writer is not None)
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                self.set_mode(self.mode_idx + 1)
            elif key == ord("h"):
                self.show_hands = not self.show_hands
            elif key == ord("s"):
                path = os.path.join(OUT_DIR, f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(path, out)
                self.toast("Screenshot saved")
                print("Saved", path)
            elif key == ord("r"):
                if writer is None:
                    path = os.path.join(OUT_DIR, f"clip_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                    self.toast("Recording...")
                    print("Recording ->", path)
                else:
                    writer.release()
                    writer = None
                    self.toast("Recording saved")
            elif key != 255:
                self.mode.handle_key(key)

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

        if writer is not None:
            writer.release()
        cap.release()
        self.hand_lm.close()
        if self.face_lm is not None:
            self.face_lm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    App().run()
