"""Deadbolt state from the camera: nearest reference sample on a small normalized patch.

Run calibrate.py once to pick the deadbolt region and capture reference frames in each
position. Nearest-sample (not an average) keeps daylight and lamp-light samples separate.
Patches are blurred and mean/variance normalized, so overall brightness changes don't matter
and camera noise isn't amplified.
"""
import json

import cv2
import numpy as np

from common import CFG, REFS, lock_patch

MIN_MARGIN = 0.08   # locked/unlocked too close to call -> "uncertain"
REJECT = 1.0        # nearest sample farther than REJECT x (typical locked-vs-unlocked gap)
                    # -> "uncertain" (door swung open, arm in the way, ...)
MIN_STD = 4.0       # patch nearly flat (lights off, lens covered) -> "uncertain"


def load_cfg():
    if not CFG.exists():
        raise SystemExit("No lock_cfg.json - run: python calibrate.py")
    return json.loads(CFG.read_text())


def _vec(patch):
    """Return a unit-scale feature vector, or None if the patch has no usable contrast."""
    p = cv2.GaussianBlur(patch, (5, 5), 0)
    p = cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).flatten()
    sd = float(p.std())
    if sd < MIN_STD:
        return None
    return (p - p.mean()) / (sd * np.sqrt(p.size))     # distances fall in [0, 2]


def _samples(folder):
    vs = []
    for p in sorted(folder.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        v = _vec(img) if img is not None else None
        if v is not None:
            vs.append(v)
    if len(vs) < 3:
        raise SystemExit(f"Need at least 3 usable reference images in {folder} - run: python calibrate.py")
    return np.stack(vs)


class LockReader:
    def __init__(self):
        self.roi = tuple(load_cfg()["roi"])
        self.locked = _samples(REFS / "locked")
        self.unlocked = _samples(REFS / "unlocked")
        # typical gap between the two states: each locked sample's distance to its nearest unlocked one
        cross = np.linalg.norm(self.locked[:, None] - self.unlocked[None], axis=2).min(1)
        self.gap = float(np.median(cross))
        if self.gap < 0.05:
            raise SystemExit("Locked and unlocked references look identical - recalibrate with a tighter box")

    def classify(self, patch):
        v = _vec(patch)
        if v is None:
            return "uncertain", 0.0
        dl = float(np.linalg.norm(self.locked - v, axis=1).min())
        du = float(np.linalg.norm(self.unlocked - v, axis=1).min())
        margin = abs(dl - du) / max(dl + du, 1e-6)
        if min(dl, du) > REJECT * self.gap or margin < MIN_MARGIN:
            return "uncertain", margin
        return ("locked" if dl < du else "unlocked"), margin

    def read(self, frame):
        """Return ('locked'|'unlocked'|'uncertain', margin)."""
        return self.classify(lock_patch(frame, self.roi))
