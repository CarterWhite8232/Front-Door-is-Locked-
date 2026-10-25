import os
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent
MODELS = Path(os.environ.get("MODELS_DIR", BASE / "models"))
KNOWN = BASE / "known"
REFS = BASE / "refs"
EVENTS = BASE / "events"
CFG = BASE / "lock_cfg.json"

W, H = 640, 480          # capture size
DET = (320, 240)         # YuNet input size (faster than full res, same aspect ratio)
SCALE = W / DET[0]

for d in (KNOWN, REFS / "locked", REFS / "unlocked", EVENTS):
    d.mkdir(parents=True, exist_ok=True)


try:   # hide OpenCV 5's harmless "Targets are not supported by the new graph engine" warnings
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def open_cam():
    # Linux (the Pi): V4L2. Windows: DirectShow, because OpenCV's default there (Media
    # Foundation) can hang for a long time when setting the resolution on USB webcams.
    if sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2
    elif sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_ANY
    cam = cv2.VideoCapture(int(os.environ.get("CAM", 0)), backend)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # always process the newest frame, not a stale one
    if not cam.isOpened():
        raise SystemExit("Camera not found. Try another index: CAM=1 (Windows PowerShell: $env:CAM=1). "
                         "On the Pi check: v4l2-ctl --list-devices. "
                         "On Windows also check Settings > Privacy > Camera and close apps using it.")
    return cam


def read(cam):
    ok, frame = cam.read()
    if not ok:
        raise SystemExit("Camera read failed")
    if frame.shape[1] != W or frame.shape[0] != H:
        frame = cv2.resize(frame, (W, H))
    return frame


def load_models():
    yunet = MODELS / "face_detection_yunet_2023mar.onnx"
    sface = MODELS / "face_recognition_sface_2021dec.onnx"
    for m in (yunet, sface):
        if not m.exists():
            raise SystemExit(f"Missing {m} - run: bash download_models.sh")
    det = cv2.FaceDetectorYN.create(str(yunet), "", DET, 0.7, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(str(sface), "")
    return det, rec


def detect(det, frame):
    """Faces in full-res coords. Each row: x, y, w, h, 5 landmark (x, y) pairs, score."""
    det.setInputSize(DET)
    faces = det.detect(cv2.resize(frame, DET))[1]
    if faces is None:
        return []
    faces[:, :14] *= SCALE
    return list(faces)


def embed(rec, frame, face):
    """128-d SFace embedding, L2-normalized (so cosine similarity = dot product)."""
    v = rec.feature(rec.alignCrop(frame, face.reshape(1, -1))).flatten()
    return v / (np.linalg.norm(v) + 1e-9)


def lock_patch(frame, roi):
    """Raw grayscale 64x64 crop of the deadbolt (normalization happens in lockstate)."""
    x, y, w, h = roi
    g = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
