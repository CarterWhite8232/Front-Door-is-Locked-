"""Door monitor: who was near the door, and was the deadbolt left unlocked?

  python door_monitor.py            run the monitor (headless, Ctrl+C to stop)
  python door_monitor.py --show     run with a preview window (debugging, q to quit)
  python door_monitor.py --last 5   show the last 5 recorded visits

Logic
  * A "visit" starts when a face appears and ends after VISIT_END s with no face. Identity
    is a vote over the visit (SFace cosine match vs enrolled people, else "unknown").
    Finished visits are logged to events/log.jsonl with the best snapshot.
  * The deadbolt is read every LOCK_EVERY s, skipped while a face is in view (people block
    the bolt). A state needs STABLE_N identical readings in a row. Door open / blocked
    reads as "uncertain" and never changes the state.
  * If the bolt stays UNLOCKED for UNLOCK_GRACE s, one alert is sent naming the people seen
    from ATTRIB_BEFORE s before the unlock up to now (most recent first). It re-arms once
    the bolt reads locked again.
  * Optional phone push via ntfy: export NTFY_TOPIC=some-long-random-name
    (ntfy.sh is public: anyone who guesses the topic can read it, so make it unguessable)
"""
import argparse
import json
import os
import threading
import time
import urllib.request
from collections import Counter, deque

import cv2
import numpy as np

from common import EVENTS, KNOWN, detect, embed, load_models, open_cam, read

MATCH_THRESH = 0.363     # SFace cosine threshold from the OpenCV docs
MIN_FACE_SCORE = 0.8     # only embed confident detections
VISIT_END = 6            # s without a face that closes a visit
LOCK_EVERY = 2.0         # s between deadbolt readings
STABLE_N = 3             # identical readings required to accept a state
UNLOCK_GRACE = int(os.environ.get("UNLOCK_GRACE", 120))
ATTRIB_BEFORE = 120      # s before the unlock that a visit still counts toward the alert
DETECT_EVERY = 3         # run YuNet on every Nth frame
LOG = EVENTS / "log.jsonl"


def stamp(wall=None):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall))


def load_known():
    names, mats = [], []
    for p in sorted(KNOWN.glob("*.npy")):
        a = np.load(p).astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
        names += [p.stem] * len(a)
        mats.append(a)
    return np.array(names), (np.vstack(mats) if mats else np.zeros((0, 128), np.float32))


def identify(emb, names, mat):
    """Best cosine match over all enrolled samples. emb must be L2-normalized."""
    if len(names) == 0:
        return "unknown", -1.0
    sims = mat @ emb
    i = int(sims.argmax())
    best = float(sims[i])
    return (str(names[i]) if best >= MATCH_THRESH else "unknown"), best


def decide_name(votes):
    """Known name with >=2 votes and >=30% of the visit's votes, else unknown."""
    total = sum(votes.values())
    for name, n in votes.most_common():
        if name != "unknown" and n >= 2 and n >= 0.3 * total:
            return name
    return "unknown"


def log(event):
    with LOG.open("a") as f:
        f.write(json.dumps(event) + "\n")
    print(event, flush=True)


def _send_ntfy(topic, title, body, image_path):
    try:
        headers = {"Title": title, "Priority": "high", "Tags": "unlock"}
        if image_path and os.path.exists(image_path):
            headers["Filename"] = "snapshot.jpg"
            headers["Message"] = body
            data = open(image_path, "rb").read()
        else:
            data = body.encode()
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=data,
                                     headers=headers, method="PUT")
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:                      # never crash the monitor over a notification
        print("notify failed:", e, flush=True)


def notify(title, body, image_path=None):
    topic = os.environ.get("NTFY_TOPIC")
    if topic:     # in a thread so a slow network never stalls the camera loop
        threading.Thread(target=_send_ntfy, args=(topic, title, body, image_path),
                         daemon=True).start()


def fmt_secs(s):
    return f"{s // 60} min" if s >= 60 and s % 60 == 0 else f"{s} s"


class Monitor:
    """All timing uses time.monotonic(): the Pi has no RTC, so wall-clock time can jump
    by hours when NTP syncs after boot. Wall time is only used for labels."""

    def __init__(self, det, rec, lock, known, notifier=notify):
        self.det, self.rec, self.lock = det, rec, lock
        self.names, self.mat = known
        self.notifier = notifier
        self.visit = None                    # in-progress visit
        self.visits = deque(maxlen=50)       # finished visits
        self.last_face_t = -1e9
        self.lock_hist, self.lock_state = [], "uncertain"
        self.unlocked_since, self.alerted = None, False
        self.next_lock_t = 0.0
        self.frame_i = 0
        self.faces = []

    # ---- faces / visits -------------------------------------------------------------
    def _faces(self, frame, now):
        self.faces = detect(self.det, frame)
        if self.faces:
            f = max(self.faces, key=lambda r: r[2] * r[3])        # biggest = closest face
            if self.visit is None:
                self.visit = {"start_t": now, "start_wall": time.time(), "votes": Counter(),
                              "scores": {}, "best_q": 0.0, "snap": None}
            self.last_face_t = now
            if f[14] >= MIN_FACE_SCORE:
                name, score = identify(embed(self.rec, frame, f), self.names, self.mat)
                v = self.visit
                v["votes"][name] += 1
                v["scores"][name] = max(v["scores"].get(name, -1.0), score)
                q = float(f[14]) * float(f[2] * f[3])             # confidence x face area
                if q > v["best_q"]:
                    v["best_q"], v["snap"] = q, frame.copy()
        elif self.visit and now - self.last_face_t > VISIT_END:
            self._close_visit()

    def _close_visit(self):
        v, self.visit = self.visit, None
        if not v["votes"]:                    # only low-confidence detections: drop it
            return
        name = decide_name(v["votes"])
        tag = time.strftime("%Y%m%d-%H%M%S", time.localtime(v["start_wall"]))
        snap = EVENTS / f"{tag}-{name}.jpg"
        cv2.imwrite(str(snap), v["snap"])
        rec = {"type": "visit", "name": name, "time": stamp(v["start_wall"]),
               "snapshot": str(snap), "score": round(float(v["scores"].get(name, 0.0)), 3),
               "start_t": v["start_t"], "end_t": self.last_face_t}
        self.visits.append(rec)
        log({k: val for k, val in rec.items() if not k.endswith("_t")})

    # ---- deadbolt -------------------------------------------------------------------
    def _lock(self, frame, now):
        if now < self.next_lock_t or now - self.last_face_t < 2:   # skip while someone's there
            return
        self.next_lock_t = now + LOCK_EVERY
        state, _ = self.lock.read(frame)
        self.lock_hist = (self.lock_hist + [state])[-STABLE_N:]
        if len(self.lock_hist) < STABLE_N or len(set(self.lock_hist)) != 1 or state == "uncertain":
            return
        if state != self.lock_state:
            self.lock_state = state
            log({"type": "lock", "state": state, "time": stamp()})
            if state == "unlocked":
                self.unlocked_since = now
            else:
                self.unlocked_since, self.alerted = None, False

    def _maybe_alert(self, now):
        if self.lock_state != "unlocked" or self.alerted or self.unlocked_since is None \
                or now - self.unlocked_since < UNLOCK_GRACE:
            return
        self.alerted = True
        u = self.unlocked_since
        seen = [v for v in self.visits if v["end_t"] >= u - ATTRIB_BEFORE]
        head = f"Door unlocked for {fmt_secs(UNLOCK_GRACE)}+."
        if seen:
            # the person in view closest to the moment the bolt went unlocked is the likely one;
            # (bolt readings pause while someone is in view, so the unlock is usually detected
            # just before they appear or just after they leave)
            def gap(v):
                return 0.0 if v["start_t"] <= u <= v["end_t"] else min(abs(v["start_t"] - u), abs(v["end_t"] - u))
            suspect = min(seen, key=gap)
            msg = f'{head} Most likely left unlocked by: {suspect["name"]} at {suspect["time"]}.'
            others = [f'{v["name"]} ({v["time"][11:16]})' for v in reversed(seen) if v is not suspect]
            if others:
                msg += " Also seen: " + ", ".join(others)
            snap = suspect["snapshot"]
        elif self.visits:
            last = self.visits[-1]
            msg = f'{head} Nobody seen around the unlock; last person before that: ' \
                  f'{last["name"]} at {last["time"]}.'
            snap = last["snapshot"]
        else:
            msg, snap = f"{head} No visitor recorded.", None
        log({"type": "alert", "message": msg, "time": stamp()})
        self.notifier("Door is unlocked", msg, snap)

    # ---- main step ------------------------------------------------------------------
    def step(self, frame, now=None):
        now = time.monotonic() if now is None else now
        self.frame_i += 1
        if self.frame_i % DETECT_EVERY == 0:
            self._faces(frame, now)
        self._lock(frame, now)
        self._maybe_alert(now)


def show_last(n):
    if not LOG.exists():
        print("No events yet.")
        return
    visits = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    visits = [v for v in visits if v.get("type") == "visit"][-n:]
    for v in reversed(visits):
        print(f'{v["time"]}  {v["name"]:<10} score={v["score"]:.2f}  {v["snapshot"]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, metavar="N")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.last is not None:
        return show_last(args.last)

    from lockstate import LockReader
    known = load_known()
    if len(known[0]) == 0:
        print("Warning: nobody enrolled, everyone will be 'unknown'. Run: python enroll.py <name>")
    det, rec = load_models()
    mon = Monitor(det, rec, LockReader(), known)
    cam = open_cam()

    print("Monitoring. Ctrl+C to stop.", flush=True)
    try:
        while True:
            frame = read(cam)
            mon.step(frame)
            if args.show:
                view = frame.copy()
                for f in mon.faces:
                    x, y, w, h = f[:4].astype(int)
                    cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
                x, y, w, h = mon.lock.roi
                cv2.rectangle(view, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(view, f"lock: {mon.lock_state}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("door", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if mon.visit:
            mon._close_visit()       # don't lose an in-progress visit on shutdown
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
