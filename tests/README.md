# Tests

Run from the `hand-portal` folder. Output (screenshots, fake camera feed) goes to `tests/out/`.

1. Gesture logic for the Python app (simulated hands, no camera):
   `.venv\Scripts\python.exe tests\test_gestures.py`
2. Website in headless Chrome, using `../cv2.mp4` as a fake webcam:
   ```
   .venv\Scripts\python.exe tests\make_fake_cam.py
   start .venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1
   node tests\browser_test.mjs
   ```
   Then look at the PNG screenshots in `tests/out/`.
