"""Deadbolt calibration.

  python calibrate.py         pick a NEW deadbolt box (clears old reference samples)
  python calibrate.py --add   keep the box, add more samples (e.g. at night with lamps on)

1. Drag a box tightly around the deadbolt thumb-turn, press ENTER.
2. Set the bolt to LOCKED, tap 'l' ~10-15 times (wiggle nothing, just let lighting vary).
3. Set the bolt to UNLOCKED, tap 'u' ~10-15 times.
4. Press 't' to test live (shows the reading), 'q' to finish.
"""
import argparse
import json
import time

import cv2

from common import CFG, REFS, lock_patch, open_cam, read

ap = argparse.ArgumentParser()
ap.add_argument("--add", action="store_true", help="reuse the existing box and add samples")
args = ap.parse_args()

cam = open_cam()
for _ in range(10):          # let auto-exposure settle
    read(cam)

if args.add:
    if not CFG.exists():
        raise SystemExit("No existing calibration - run without --add first")
    roi = tuple(json.loads(CFG.read_text())["roi"])
else:
    frame = read(cam)
    roi = cv2.selectROI("Select deadbolt, then ENTER", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        raise SystemExit("No region selected")
    roi = tuple(int(v) for v in roi)
    for state in ("locked", "unlocked"):           # old samples belong to the old box
        for p in (REFS / state).glob("*.png"):
            p.unlink()
    CFG.write_text(json.dumps({"roi": list(roi)}))

counts = {s: len(list((REFS / s).glob("*.png"))) for s in ("locked", "unlocked")}
tester = None

while True:
    frame = read(cam)
    x, y, w, h = roi
    view = frame.copy()
    cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(view, f"l=locked({counts['locked']}) u=unlocked({counts['unlocked']}) t=test q=done",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    if tester:
        state, margin = tester.read(frame)
        cv2.putText(view, f"reading: {state} ({margin:.2f})", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("calibrate", view)
    k = cv2.waitKey(30) & 0xFF
    if k == ord("q"):
        break
    if k == ord("t"):
        from lockstate import LockReader
        tester = LockReader()          # reloads samples, so press t again after adding more
    for key, name in ((ord("l"), "locked"), (ord("u"), "unlocked")):
        if k == key:
            cv2.imwrite(str(REFS / name / f"{int(time.time() * 1000)}.png"), lock_patch(frame, roi))
            counts[name] += 1

cv2.destroyAllWindows()
cam.release()
print(f"Saved. locked={counts['locked']} unlocked={counts['unlocked']}")
if min(counts.values()) < 5:
    print("Warning: capture at least ~10 of each state for reliable results.")
