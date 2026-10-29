# Front Door Is Locked?

A Raspberry Pi watches the inside of the front door. It recognizes who comes and goes, checks
whether the deadbolt is thrown, and if the door is left unlocked for 2 minutes it sends a phone
notification naming the person most likely to have left it that way, with a snapshot.

Everything runs locally on a Raspberry Pi 4 with a USB webcam. No cloud vision service, no GPU.

```
camera frame ──┬─ every 3rd frame ─► YuNet: find faces ─► SFace: 128-d vector ─► match enrolled people ─► visit log
               │
               └─ every 2 s, no one in view ─► crop deadbolt ─► compare to reference photos ─► locked / unlocked
                                                                                                     │
                       unlocked for 2 min ─► alert: person seen closest to the moment it was unlocked ◄┘
```

## How it works

| Part | Method |
| --- | --- |
| Face detection | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (OpenCV Zoo, ~230 KB ONNX), run on a 320×240 copy of each frame |
| Face recognition | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (OpenCV Zoo, ~37 MB ONNX): each face is aligned and turned into a 128-number vector, then matched to enrolled people by cosine similarity (threshold 0.363) |
| Identity per visit | Every analyzed frame votes; a known name needs ≥ 2 votes and ≥ 30% of the visit's votes, otherwise "unknown" |
| Deadbolt state | No sensor: a nearest-neighbor comparison of the deadbolt region against your own reference photos, normalized for brightness. Door open, blocked view or darkness reads as "uncertain" and never changes the state |
| Debouncing | A lock state is accepted only after 3 identical readings in a row; readings pause while someone is in view |
| Alerts | Sent once per unlock via [ntfy](https://ntfy.sh) (free push notifications), in a background thread |
| Timing | `time.monotonic()` throughout, because the Pi has no hardware clock and wall time can jump on boot |

## Hardware

- Raspberry Pi 4 Model B running Raspberry Pi OS (64-bit)
- USB webcam, mounted inside facing the door so it sees both people entering and the deadbolt
- Official 5 V / 3 A USB-C power supply
- Planned: HC-SR501 PIR motion sensor and an I2C display (see [Roadmap](#roadmap))

## Files

| File | Purpose | Run on |
| --- | --- | --- |
| `common.py` | Shared code: camera, model loading, face detection and embedding, deadbolt crop | — |
| `enroll.py` | Enroll a person from the live camera (20 samples) | Laptop or Pi |
| `enroll_photos.py` | Enroll a person from a folder of photos | Laptop or Pi |
| `test_recognize.py` | Live recognition demo: names, scores, FPS | Laptop or Pi |
| `calibrate.py` | Draw the deadbolt box and capture locked/unlocked reference photos | Pi, camera in final position |
| `lockstate.py` | Deadbolt classifier | — |
| `door_monitor.py` | The monitor itself | Pi |
| `download_models.sh` | Downloads the two ONNX models into `models/` | Once per machine |
| `requirements.txt` | `opencv-python>=4.8`, `numpy` | — |

Private data is never committed (see `.gitignore`): `known/` (face vectors), `refs/` (deadbolt
photos), `events/` (snapshots and log), `lock_cfg.json` and `models/`.

## Setup on the Pi

```bash
# system packages
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git gh v4l-utils python3-venv python3-pip

# project
gh auth login
gh repo clone Carterwhite8232/Front-Door-is-Locked- ~/Front-Door-is-Locked-
cd ~/Front-Door-is-Locked-
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import cv2; print(cv2.__version__)"   # must be 4.8 or newer
bash download_models.sh
```

`--system-site-packages` lets the venv use the OS's GPIO libraries for the planned motion sensor.
Every new terminal needs `source venv/bin/activate` before running the scripts.

## Usage

```bash
# 1. find the webcam (prefix commands with CAM=N if it isn't /dev/video0)
v4l2-ctl --list-devices

# 2. enroll each person
python enroll.py carter

# 3. check recognition: green box + name = recognized
python test_recognize.py

# 4. calibrate the deadbolt (camera mounted in its final spot)
python calibrate.py            # box the bolt, then l = locked sample, u = unlocked, t = test, q = done
python calibrate.py --add      # later: add night-time samples under the same box

# 5. try the monitor with a 30-second alert delay and a preview window
UNLOCK_GRACE=30 python door_monitor.py --show

# recent visits
python door_monitor.py --last 10
```

### Settings

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `CAM` | `0` | Camera index |
| `UNLOCK_GRACE` | `120` | Seconds unlocked before alerting |
| `NTFY_TOPIC` | unset | ntfy topic for phone alerts; no alerts are sent if unset |
| `MODELS_DIR` | `./models` | Where the ONNX models live |

Other tuning values (match threshold, visit timeout, readings needed) are constants at the top of
`door_monitor.py` and `lockstate.py`.

## Phone notifications

1. Install the **ntfy** app and allow notifications.
2. Make an unguessable topic, since ntfy.sh topics are public:
   `python3 -c "import secrets; print('door-' + secrets.token_hex(8))"`
3. Subscribe to it in the app (server `ntfy.sh`).
4. Test: `curl -H "Title: Test" -d "Hello from the door Pi" ntfy.sh/<your-topic>`
5. Set `NTFY_TOPIC=<your-topic>` for manual runs (e.g. in `~/.bashrc`) and in the service file below.

## Run on boot

```bash
sudo tee /etc/systemd/system/door-monitor.service > /dev/null <<EOF
[Unit]
Description=Door monitor
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$HOME/Front-Door-is-Locked-
Environment=NTFY_TOPIC=<your-topic>
Environment=PYTHONUNBUFFERED=1
ExecStart=$HOME/Front-Door-is-Locked-/venv/bin/python door_monitor.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now door-monitor
systemctl status door-monitor          # should be "active (running)"
journalctl -u door-monitor -f          # live output
```

Replace `<your-topic>` before pasting. Only one program can use the camera at a time, so stop the
service before running `enroll.py`, `calibrate.py` or `test_recognize.py`:

```bash
sudo systemctl stop door-monitor
# ...
sudo systemctl start door-monitor
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named cv2` | Activate the venv: `source venv/bin/activate` |
| `Camera not found` | Try another index (`CAM=1 python ...`); check `v4l2-ctl --list-devices`; stop the service if it's running |
| `Missing models/...onnx` | `bash download_models.sh` |
| `No lock_cfg.json` | `python calibrate.py` |
| Warning `skipping <name>.npy` | That face file is damaged (often a power cut while saving); re-run `python enroll.py <name>` |
| Often "unknown" | Re-enroll on the Pi's own camera at the door, moving your head during capture |
| Lock reads wrong at night | `python calibrate.py --add` with the lights on, a round of `l` and `u` each |
| No phone alert | Check the `curl` test above, the topic name in the app, and phone notification permissions |

Always shut down with `sudo shutdown now` before unplugging the Pi; pulling power can corrupt files
on the SD card.

## Privacy

Only 128-number face vectors are stored, never enrollment photos. Snapshots of visits stay on the
Pi in `events/`. Recognition runs fully offline; the only data that leaves the Pi is the alert text
and one snapshot sent to your ntfy topic. Let housemates and regular guests know about the camera.

## Roadmap

- [ ] PIR motion sensor (HC-SR501 on GPIO17) to skip processing when no one is near the door
- [ ] I2C display in the housing for messages like "Welcome, Carter" and "Door unlocked, please lock it"
- [ ] Enclosure and permanent wiring
