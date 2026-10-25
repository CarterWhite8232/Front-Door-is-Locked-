"""Enroll a person:  python enroll.py carter

Look at the camera, move your head a little (turn, tilt, glasses on/off).
Samples are taken automatically, only when exactly one face is visible.
Only the 128-d embeddings are stored (known/<name>.npy), not photos.
Re-running for the same name replaces that person's samples.
"""
import re
import sys
import time

import cv2
import numpy as np

from common import KNOWN, detect, embed, load_models, open_cam, read

N = 20
if len(sys.argv) != 2 or not re.fullmatch(r"[A-Za-z0-9_-]+", sys.argv[1]):
    raise SystemExit("usage: python enroll.py <name>   (letters, digits, _ or - only)")
name = sys.argv[1].lower()

det, rec = load_models()
cam = open_cam()
embs, last = [], 0.0

try:
    while len(embs) < N:
        frame = read(cam)
        faces = detect(det, frame)
        view = frame.copy()
        for f in faces:
            x, y, w, h = f[:4].astype(int)
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if len(faces) == 1 and faces[0][14] > 0.85 and time.monotonic() - last > 0.6:
            embs.append(embed(rec, frame, faces[0]))
            last = time.monotonic()
        cv2.putText(view, f"{name}: {len(embs)}/{N} % (q to abort)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("enroll", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit("aborted, nothing saved")
finally:
    cam.release()
    cv2.destroyAllWindows()

np.save(KNOWN / f"{name}.npy", np.array(embs, dtype=np.float32))
print(f"Saved {len(embs)} embeddings for {name}")
