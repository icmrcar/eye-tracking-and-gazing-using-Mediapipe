"""
main.py

Full production control loop: camera -> face/eye tracking -> instant gaze
classification -> blink-based STOP latch -> joystick override -> fail-safe
-> speed ramp -> ESP32 serial -> logging -> dashboard display.

Priority order each frame, highest to lowest:
    1. fail_safe_triggered (face lost / serial lost) -> STOP, hard latch
    2. EMERGENCY_STOP (joystick button) -> STOP, hard latch
    3. Joystick outside deadzone -> manual override, CAN break a
       blink-triggered STOP latch, but not a fail-safe/emergency latch
    4. Blink-triggered STOP (long blink or double blink) -> STOP, latches
       until a confirmed UP gaze (eyes stably OPEN) or a joystick push
       clears it
    5. Mid-blink (eyes not fully OPEN) -> hold the last known command
    6. Otherwise: instant gaze direction -> F/L/R/-, no dwell delay

NOTE: this is the simple "instant command" version. See main_logic1.py and
main_logic2.py for the alternative hold-to-activate state machine designs
under evaluation.
"""

import time
import traceback

import config
from camera import Camera
from face_tracking import FaceTracker
from eye_features import EyeFeatureExtractor
from calibration import Calibration
from gaze_classifier import GazeClassifier, GazeDirection
from blink_detector import BlinkDetector, BlinkState
from joystick_input import JoystickInput
from input_arbiter import InputArbiter
from fail_safe import FailSafeChecker
from speed_ramp import SpeedRamper
from serial_comm import SerialComm
from data_logger import DataLogger
from display import Display

CALIBRATION_FILE = "calibration_data.json"

DIRECTION_TO_COMMAND = {
    GazeDirection.UP: "F",
    GazeDirection.LEFT: "L",
    GazeDirection.RIGHT: "R",
    GazeDirection.CENTER: "-",
}

COMMAND_TO_STATE_NAME = {
    "F": "FORWARD",
    "L": "TURN_LEFT",
    "R": "TURN_RIGHT",
    "S": "STOP",
    "-": "IDLE",
}


def log(message: str) -> None:
    if config.DEBUG:
        print(f"[main.py] {message}")


def joystick_to_command(joystick_command) -> str:
    if joystick_command is None or joystick_command.in_deadzone:
        return "-"
    if joystick_command.y < -0.5:
        return "F"
    if joystick_command.x < -0.5:
        return "L"
    if joystick_command.x > 0.5:
        return "R"
    return "-"


def run_calibration(camera, face_tracker, eye_extractor, calibration) -> bool:
    log("Starting calibration...")
    success = calibration.run_initial_calibration(camera, face_tracker, eye_extractor)
    if success:
        calibration.save_calibration(CALIBRATION_FILE)
        log("Calibration successful and saved.")
    else:
        log("ERROR: Calibration failed or was cancelled.")
    return success


def main():
    camera = Camera()
    face_tracker = FaceTracker()
    eye_extractor = EyeFeatureExtractor()
    calibration = Calibration()
    classifier = GazeClassifier()
    blink_detector = BlinkDetector()
    joystick = JoystickInput()
    arbiter = InputArbiter()
    fail_safe = FailSafeChecker()
    speed_ramper = SpeedRamper()
    serial_comm = SerialComm()
    logger = DataLogger()
    display = Display()

    log("Opening camera...")
    if not camera.open():
        raise RuntimeError("Camera failed to open. Aborting startup.")

    log("Initializing MediaPipe Face Mesh...")
    face_tracker.initialize()

    joystick.initialize()

    if not serial_comm.open():
        raise RuntimeError("Serial connection to ESP32 failed. Aborting startup.")

    logger.open()

    if calibration.load_calibration(CALIBRATION_FILE):
        log(f"Loaded existing calibration from {CALIBRATION_FILE}.")
    else:
        log("No existing calibration found.")
        if not run_calibration(camera, face_tracker, eye_extractor, calibration):
            camera.release()
            face_tracker.close()
            serial_comm.close()
            logger.close()
            return

    log("Entering main control loop.")

    prev_time = time.time()
    last_command = "-"
    no_face_warned = False
    stopped = False
    blink_counts = {"total": 0, "long": 0, "double": 0}

    running = True
    while running:
        loop_start = time.time()

        try:
            success, frame = camera.read_frame()
            if not success:
                continue

            tracking_result = face_tracker.process_frame(frame)

            if not tracking_result.face_detected:
                if not no_face_warned:
                    log("WARNING: Face lost.")
                    no_face_warned = True
            else:
                if no_face_warned:
                    log("Face reacquired.")
                    no_face_warned = False

            eye_features = eye_extractor.get_smoothed_features(tracking_result)
            raw_direction = classifier.classify_raw_direction(eye_features)

            blink_state = blink_detector.update(eye_features.raw_ear)
            raw_stop_event = blink_detector.get_stop_event()
            holding = blink_detector.is_holding_stop()
            blink_stop_event = raw_stop_event or holding

            if blink_state in (BlinkState.NATURAL_BLINK, BlinkState.DELIBERATE_BLINK):
                blink_counts["total"] += 1
            if raw_stop_event:
                if blink_state == BlinkState.HOLDING_CLOSED:
                    blink_counts["long"] += 1
                    log(f"Long blink counted (total long blinks: {blink_counts['long']}).")
                elif blink_state == BlinkState.NATURAL_BLINK:
                    blink_counts["double"] += 1
                    log(f"Double blink counted (total double blinks: {blink_counts['double']}).")

            joystick_command = joystick.read_state()
            arbiter_decision = arbiter.resolve(
                raw_direction, joystick_command, tracking_result.face_detected
            )

            fail_safe_result = fail_safe.check(
                face_detected=tracking_result.face_detected,
                serial_connected=serial_comm.is_connected(),
                last_serial_ack_time=serial_comm.last_ack_time,
            )

            is_emergency = (arbiter_decision.source == "EMERGENCY_STOP")
            hard_stop = fail_safe_result.triggered or is_emergency

            newly_stopped = False
            forced_stop = hard_stop or blink_stop_event
            if forced_stop and not stopped:
                stopped = True
                newly_stopped = True
                if fail_safe_result.triggered:
                    log(f"STOP triggered by fail-safe ({fail_safe_result.reason}).")
                elif is_emergency:
                    log("STOP triggered by joystick emergency button.")
                else:
                    log(f"STOP triggered by blink (state={blink_state}). Look UP to resume.")

            if stopped:
                command = "S"

                if hard_stop:
                    pass
                elif arbiter_decision.source == "JOYSTICK":
                    candidate = joystick_to_command(arbiter_decision.direction_or_command)
                    if candidate != "-":
                        stopped = False
                        command = candidate
                        log("Resumed: joystick override cleared STOP latch.")
                elif (not newly_stopped and blink_state == BlinkState.OPEN
                      and raw_direction == GazeDirection.UP):
                    stopped = False
                    command = "F"
                    log("Resumed: UP detected with eyes stably open, STOP latch cleared.")

            elif arbiter_decision.source == "JOYSTICK":
                command = joystick_to_command(arbiter_decision.direction_or_command)

            elif blink_state != BlinkState.OPEN:
                command = last_command

            elif arbiter_decision.source == "GAZE":
                command = DIRECTION_TO_COMMAND.get(raw_direction, "-")

            else:
                command = last_command

            if command != last_command:
                log(f"Command changed: {last_command} -> {command}  "
                    f"(direction={raw_direction}, source={arbiter_decision.source}, "
                    f"blink={blink_state}, H={eye_features.horizontal_ratio:.3f}, "
                    f"V={eye_features.vertical_ratio:.3f})")
                last_command = command

            state_name = COMMAND_TO_STATE_NAME.get(command, "IDLE")
            left_pwm, right_pwm = speed_ramper.step(state_name, is_emergency=is_emergency)

            serial_comm.send_command(command, left_pwm, right_pwm)

            now = time.time()
            frame_dt = now - prev_time
            fps = 1.0 / frame_dt if frame_dt > 0 else 0.0
            prev_time = now
            if fps < config.FPS_WARNING_THRESHOLD:
                log(f"WARNING: Low FPS ({fps:.1f}).")

            logger.log_frame({
                "Timestamp": time.time(),
                "Trial_ID": None,
                "Horizontal_Ratio": eye_features.horizontal_ratio,
                "Vertical_Ratio": eye_features.vertical_ratio,
                "EAR": eye_features.ear,
                "Eye_Direction": raw_direction,
                "Dwell_Confirmed": True,
                "Blink_Duration_ms": None,
                "Input_Source": arbiter_decision.source,
                "FSM_State": state_name,
                "Wheelchair_Command": command,
                "Left_PWM": left_pwm,
                "Right_PWM": right_pwm,
                "Fail_Safe_Triggered": fail_safe_result.reason,
                "FPS": fps,
            })

            display.render(frame, state_name, raw_direction,
                            arbiter_decision.source, command, fps,
                            tracking_result=tracking_result)

            if display.check_exit_key():
                running = False

        except Exception:
            log("ERROR: Exception during frame processing:")
            traceback.print_exc()
            continue

    camera.release()
    face_tracker.close()
    serial_comm.close()
    logger.close()
    display.close()
    log("Shutdown complete.")


if __name__ == "__main__":
    main()
