"""Turn ../cv2.mp4 into tests/out/cam.y4m, a fake webcam feed for the browser test."""
import os
import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "tests", "out")
os.makedirs(OUT, exist_ok=True)
W, H = 640, 352
cap = cv2.VideoCapture(os.path.join(HERE, "..", "cv2.mp4"))
with open(os.path.join(OUT, "cam.y4m"), "wb") as f:
    f.write(f"YUV4MPEG2 W{W} H{H} F30:1 Ip A1:1 C420jpeg\n".encode())
    for _ in range(240):
        ok, fr = cap.read()
        if not ok:
            break
        h = fr.shape[0]
        crop = fr[int(h * 0.345):int(h * 0.67)]  # just the camera window of the screen recording
        f.write(b"FRAME\n" + cv2.cvtColor(cv2.resize(crop, (W, H)), cv2.COLOR_BGR2YUV_I420).tobytes())
print("wrote", os.path.join(OUT, "cam.y4m"))
