# ESP32 Firmware Setup

## Pins
- LEFT MOTOR (MDDS30): AN1 = 25 (PWM), IN1 = 26 (direction)
- RIGHT MOTOR (MDDS30): AN2 = 33 (PWM), IN2 = 32 (direction)
- JOYSTICK: VRX = 34, VRY = 35
- BRAKE RELAY: 18

## Before flashing
1. Install the ESP32 board package in Arduino IDE (Boards Manager -> "esp32" by Espressif Systems).
2. Confirm your sketch folder contains **only one `.ino` file** (`esp32_firmware.ino`) -- a duplicate file in the same folder causes "redefinition" compile errors, since Arduino merges every `.ino` file in the folder into one program.
3. `SERIAL_BAUD_RATE` (115200) must match `config.SERIAL_BAUD_RATE` on the Python side.

## Verify on a stand (wheels off the ground) before real use
- Which relay state actually engages the physical brake (`BRAKE_ENGAGED`/`BRAKE_RELEASED`).
- Which motor direction pin level means "forward" on your wiring.
- Joystick axis polarity: pull back and confirm it reverses, not turns. If it turns instead, check for X-axis crosstalk -- `STRAIGHT_LOCK_THRESHOLD` (150) should suppress this when pushing/pulling hard. If pulling back drives forward instead of reverse, flip `INVERT_Y`.

## Flashing
1. Open `esp32_firmware.ino` in Arduino IDE.
2. Select your ESP32 board and port under Tools.
3. Upload.
4. Open Serial Monitor at 115200 baud -- you should see joystick center calibration output, then `READY`.
