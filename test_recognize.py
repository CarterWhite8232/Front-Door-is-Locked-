"""Live face recognition test (no deadbolt calibration needed):  python test_recognize.py

Draws a box on each face with the matched name and score. Press q to quit.
Green = recognized (score >= threshold), red = unknown.
"""
import time

import cv2

from common import detect, embed, load_models, open_cam, read
from door_monitor import MATCH_THRESH, load_known, identify

names, mat = load_known()
if len(names) == 0:
    raise SystemExit("Nobody enrolled yet - run: python enroll.py <name>")
print("Enrolled:", sorted(set(names.tolist())), f"| threshold {MATCH_THRESH}")

det, rec = load_models()
cam = open_cam()
t0, frames = time.monotonic(), 0
try:
    while True:
        frame = read(cam)
        for f in detect(det, frame):
            name, score = identify(embed(rec, frame, f), names, mat)
            x, y, w, h = f[:4].astype(int)
            color = (0, 200, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} {score:.2f}", (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        frames += 1
        fps = frames / max(time.monotonic() - t0, 1e-6)
        cv2.putText(frame, f"{fps:.1f} fps  (q to quit)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("recognition test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cam.release()
    cv2.destroyAllWindows()