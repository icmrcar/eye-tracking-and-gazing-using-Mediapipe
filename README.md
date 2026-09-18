# eye-tracking-and-gazing-using-Mediapipe
eye tracking and gazing using Mediapipe 



# Eye-Gaze Controlled Smart Wheelchair

A low-cost, camera-based assistive mobility system that lets a person drive a wheelchair using only their eyes and blinks — with a manual joystick always available as an override. Developed at CEERI, Pilani.

---

## Introduction

Standard wheelchairs rely on hand and arm control, which many people with severe motor impairments (spinal cord injury, ALS, muscular dystrophy) don't have. Existing gaze-controlled wheelchairs solve this, but typically rely on costly, specialized eye-tracking hardware.

This project uses an ordinary webcam or tablet camera combined with real-time computer vision to detect gaze direction and blinks, converting them into wheelchair movement commands — without any dedicated eye-tracking hardware.

---

## Features

- **Eye-gaze control** — look Up to move forward, Left/Right to turn, from a full stop.
- **3-second hold-to-activate** — filters out accidental glances before any command is sent.
- **Blink-to-stop** — a long blink (>0.8s) or a quick double-blink stops the chair instantly.
- **Manual joystick override** — always takes priority the instant it's moved, with a dedicated emergency-stop button.
- **Dual-layer fail-safe** — stops automatically if the face is lost or the serial link to the ESP32 drops, both on the PC side and independently on the ESP32 itself.
- **Automatic USB reconnect** — recovers on its own if the serial cable is unplugged and replugged.
- **Per-user calibration** — a one-time setup (look Center, Up, Left, Right) calibrates gaze thresholds to the individual user.

---

## How It Works

1. A camera continuously watches the user's eyes.
2. MediaPipe Face Mesh detects face and eye landmarks each frame.
3. Eye landmark positions are converted into a Horizontal Ratio and Vertical Ratio (0–1 scale) representing where the iris sits within its usable range of motion.
4. EAR (Eye Aspect Ratio) is computed each frame to detect blinks:

   ```
   EAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 × ‖p1 − p4‖)
   ```

   where p1–p6 are the six eye landmark points (corners + top/bottom pairs). EAR below **0.2** is classified as a closed eye.
5. The gaze direction (Up / Left / Right / Center) is classified from the ratios against calibrated thresholds.
6. Holding a direction for **3 seconds** confirms it into a command (Forward / Left / Right).
7. A long blink or double-blink triggers **Stop**, independent of gaze direction.
8. The confirmed command is sent over serial to an **ESP32**, which drives the motors through an **MDDS30** motor driver with a relay-based mechanical brake.
9. The joystick (wired directly to the ESP32) always takes priority over gaze input if moved.
10. This entire loop repeats continuously, about 20–30 times per second, acting as a safety heartbeat — if commands stop arriving, the ESP32's own watchdog forces a stop.

---

## Hardware

| Component | Role |
|---|---|
| Webcam / tablet camera | Captures the live eye/face feed |
| ESP32 | Reads commands over serial, drives the motors |
| MDDS30 motor driver | Dual-channel motor driver with PWM + direction control |
| Relay module | Engages/releases the mechanical brake |
| Analog joystick | Wired directly to the ESP32 for manual override |

See `esp32_firmware/esp32_firmware.ino` for pin assignments and wiring notes.

---

## Software Requirements

- **Python 3.11** (mediapipe's legacy `solutions` API — which this project depends on — does not have official wheels for Python 3.13, and was removed entirely from mediapipe's newest releases)
- Dependencies pinned in `requirements.txt`:

```
mediapipe==0.10.21
numpy==1.26.4
opencv-python==4.10.0.84
pyserial==3.5
pygame==2.6.1
```

---

## Installation

```bash
# Clone the repository
git clone <https://github.com/icmrcar/eye-tracking-and-gazing-using-Mediapipe.git>
cd <eye-tracking-and-gazing-using-Mediapipe.git>

# Create a Python 3.11 virtual environment
py -3.11 -m venv venv        # Windows
python3.11 -m venv venv      # macOS/Linux

# Activate it
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

Flash `esp32_firmware/esp32_firmware.ino` to the ESP32 using the Arduino IDE before running the Python side.

---

## Usage

```bash
python main_logic1.py
```

On first run, the system will walk through calibration (look Center, Up, Left, Right as prompted). Calibration is saved to `calibration_data.json` and reused on future runs.

Update `SERIAL_PORT` in `config.py` to match your ESP32's port (e.g. `COM3` on Windows, `/dev/ttyUSB0` on Linux) before running.

Press `q` at any time to quit.

---

## Project Structure

```
main_logic1.py       # Main control loop (strict: turning only from a full stop) -- recommended starting point
main_logic2.py        # Alternative control loop (smoother: turning allowed while moving)
main.py                # Legacy instant-command version (no hold-to-activate) -- not recommended for real use
config.py              # All tunable constants: thresholds, timings, pins, serial settings
camera.py              # Webcam initialization and frame capture
face_tracking.py       # MediaPipe Face Mesh wrapper, landmark + head pose extraction
eye_features.py         # Computes horizontal/vertical gaze ratios and EAR from landmarks
gaze_classifier.py     # Classifies raw gaze direction and applies dwell/hold confirmation
blink_detector.py      # Detects long blinks and double blinks for the Stop command
calibration.py          # Per-user calibration routine
input_arbiter.py       # Decides gaze vs. joystick control priority each frame
joystick_input.py      # Reads the PC gamepad (pygame) for manual override
serial_comm.py         # Serial communication with the ESP32, with automatic reconnect
speed_ramp.py           # Gradual PWM ramping instead of instant full-speed jumps
fail_safe.py            # Forces STOP on face loss or serial loss
data_logger.py          # CSV logging of session data
display.py              # Live debugging dashboard (camera feed, eye zooms, gaze plot, status)
fsm.py                  # Legacy state machine, not currently used
gaze_test.py            # Standalone vision-pipeline test (no ESP32/joystick required)
test_phase1.py           # Standalone test: camera + face/iris landmark detection only
test_phase2.py           # Standalone test: camera + eye feature ratios + calibration
esp32_firmware/
  esp32_firmware.ino    # ESP32 firmware: joystick + gaze/serial control, motor driver, fail-safe watchdog
requirements.txt        # Pinned Python dependencies
```

---

## Safety Notes

- Always test with the wheelchair's wheels off the ground first after any code or wiring change.
- Verify the brake relay's engaged/released wiring matches your hardware before trusting the chair on the floor.
- Confirm forward/reverse motor direction matches expectations before real-world use.

---

## Applications

- Assistive mobility for individuals with spinal cord injury, ALS, or muscular dystrophy
- Elderly-care and rehabilitation centres
- Educational institutions, supporting independent campus mobility
- Extendable beyond wheelchairs — the same gaze interface can control robotic arms or home-automation systems

---

## Future Plans

- Gaze/gesture-based reverse (backward) command
- Improved tracking robustness under varying lighting conditions
- Obstacle-detection sensors for collision avoidance
- Onboard, laptop-free embedded processing
- Voice command integration
- Mobile app for caregiver monitoring and control

---

## Acknowledgements

Developed at CEERI (Central Electronics Engineering Research Institute), Pilani.

- Website: [www.ceeri.res.in](https://www.ceeri.res.in)
- E-mail: nishantraj7479@gmail.com
- Phone: 7479747634
