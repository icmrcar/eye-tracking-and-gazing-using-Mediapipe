"""
main_logic2.py

LOGIC 2 -- SMOOTH hold-to-activate motion state machine.

States: STOPPED, FORWARD, TURN_LEFT, TURN_RIGHT

Rules (same as Logic 1 EXCEPT the two marked DIFFERENT FROM LOGIC 1 below):
    - From STOPPED, holding UP/LEFT/RIGHT for config.ACTIVATION_HOLD_MS
      (fixed 3 seconds) activates FORWARD/TURN_LEFT/TURN_RIGHT.
    - Brief eye flickers under config.MOTION_FLICKER_TOLERANCE_MS are
      tolerated without resetting an activation hold in progress.
    - While FORWARD: CENTER keeps driving forward (does not stop).
      *** DIFFERENT FROM LOGIC 1 ***: holding LEFT/RIGHT for the same
      3-second activation duration turns DIRECTLY from FORWARD -- no stop
      required first.
    - While TURNING: the held direction must be maintained continuously
      (same flicker tolerance).
      *** DIFFERENT FROM LOGIC 1 ***: releasing to CENTER (past tolerance)
      goes DIRECTLY back to FORWARD -- no stop, no re-hold delay needed.
    - A long blink or double blink always forces STOPPED immediately,
      from any state, and clears any activation hold in progress. This is
      the ONLY way to reach a full stop while moving in Logic 2.
    - A beep plays the instant any movement activates (including a direct
      forward->turn activation).
    - Speed ramps up gradually after activation (speed_ramp.py).

Run this to test Logic 2 on the real wheelchair. Run main_logic1.py
separately to test the alternative (strict) design, then compare.

Usage:
    python main_logic2.py
"""

import time
import traceback

import numpy as np

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

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

CALIBRATION_FILE = "calibration_data.json"

# ---------------------------------------------------------------------
# Beep cue -- slightly different pitch from Logic 1 so you can tell which
# script is running just by ear during side-by-side testing.
# ---------------------------------------------------------------------

_BEEP_FREQ_HZ = 990
_BEEP_DURATION_S = 0.12
_BEEP_SAMPLE_RATE = 44100
_beep_sound = None
_beep_init_attempted = False


def _init_beep() -> None:
    global _beep_sound, _beep_init_attempted
    if _beep_init_attempted:
        return
    _beep_init_attempted = True
    if not _PYGAME_AVAILABLE:
        print("[main_logic2.py] NOTE: pygame not installed -- activation beep disabled.")
        return
    try:
        pygame.mixer.init(frequency=_BEEP_SAMPLE_RATE, size=-16, channels=1)
        t = np.linspace(0, _BEEP_DURATION_S, int(_BEEP_SAMPLE_RATE * _BEEP_DURATION_S), False)
        wave = np.sin(_BEEP_FREQ_HZ * t * 2 * np.pi)
        fade_len = max(1, int(0.01 * _BEEP_SAMPLE_RATE))
        envelope = np.ones_like(wave)
        envelope[:fade_len] = np.linspace(0.0, 1.0, fade_len)
        envelope[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
        wave = wave * envelope
        audio = (wave * 32767 * 0.5).astype(np.int16)
        _beep_sound = pygame.sndarray.make_sound(audio)
    except Exception as e:
        print(f"[main_logic2.py] NOTE: Beep unavailable ({e}).")
        _beep_sound = None


def _play_beep() -> None:
    _init_beep()
    if _beep_sound is not None:
        try:
            _beep_sound.play()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Motion state machine
# ---------------------------------------------------------------------

class MotionState:
    STOPPED = "STOPPED"
    FORWARD = "FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"


_ACTIVATION_TARGET = {
    "UP": MotionState.FORWARD,
    "LEFT": MotionState.TURN_LEFT,
    "RIGHT": MotionState.TURN_RIGHT,
}
_TURN_REQUIRED_DIRECTION = {
    MotionState.TURN_LEFT: "LEFT",
    MotionState.TURN_RIGHT: "RIGHT",
}


class GazeMotionFSM:
    """LOGIC 2 -- smooth: forward<->turn transitions don't require a full stop."""

    def __init__(self, activation_hold_ms, flicker_tolerance_ms, on_activate=None):
        self.state = MotionState.STOPPED
        self._activation_hold_ms = activation_hold_ms
        self._flicker_tolerance_ms = flicker_tolerance_ms
        self._on_activate = on_activate

        self._activation_candidate = None
        self._activation_start = None
        self._activation_off_since = None

        self._turn_off_since = None

    def _reset_activation(self):
        self._activation_candidate = None
        self._activation_start = None
        self._activation_off_since = None

    def _update_activation_hold(self, raw_direction, now, allowed):
        candidate = raw_direction if raw_direction in allowed else None

        if candidate == self._activation_candidate and candidate is not None:
            self._activation_off_since = None
            return

        if candidate is None:
            if self._activation_candidate is None:
                return
            if self._activation_off_since is None:
                self._activation_off_since = now
            elif (now - self._activation_off_since) * 1000 > self._flicker_tolerance_ms:
                self._reset_activation()
            return

        self._activation_candidate = candidate
        self._activation_start = now
        self._activation_off_since = None

    def _activation_elapsed_ms(self, now):
        if self._activation_start is None:
            return 0
        return (now - self._activation_start) * 1000

    def activation_progress(self, now):
        """Returns (candidate_direction, elapsed_ms, required_ms) for HUD display."""
        if self._activation_candidate is None:
            return None, 0, self._activation_hold_ms
        return self._activation_candidate, self._activation_elapsed_ms(now), self._activation_hold_ms

    def update(self, raw_direction, blink_stop_triggered):
        now = time.time()

        if blink_stop_triggered:
            self._reset_activation()
            self._turn_off_since = None
            self.state = MotionState.STOPPED
            return self.state

        if self.state == MotionState.STOPPED:
            self._update_activation_hold(raw_direction, now, allowed=("UP", "LEFT", "RIGHT"))
            if (self._activation_candidate is not None
                    and self._activation_elapsed_ms(now) >= self._activation_hold_ms):
                new_state = _ACTIVATION_TARGET[self._activation_candidate]
                self._reset_activation()
                self.state = new_state
                if self._on_activate:
                    self._on_activate(new_state)
            return self.state

        if self.state == MotionState.FORWARD:
            # DIFFERENT FROM LOGIC 1: holding LEFT/RIGHT for the same
            # activation duration turns DIRECTLY from forward, no stop.
            self._update_activation_hold(raw_direction, now, allowed=("LEFT", "RIGHT"))
            if (self._activation_candidate is not None
                    and self._activation_elapsed_ms(now) >= self._activation_hold_ms):
                new_state = _ACTIVATION_TARGET[self._activation_candidate]
                self._reset_activation()
                self.state = new_state
                if self._on_activate:
                    self._on_activate(new_state)
            return self.state

        if self.state in (MotionState.TURN_LEFT, MotionState.TURN_RIGHT):
            required = _TURN_REQUIRED_DIRECTION[self.state]
            if raw_direction == required:
                self._turn_off_since = None
            else:
                if self._turn_off_since is None:
                    self._turn_off_since = now
                elif (now - self._turn_off_since) * 1000 > self._flicker_tolerance_ms:
                    # DIFFERENT FROM LOGIC 1: releasing the turn goes
                    # DIRECTLY back to FORWARD -- no stop, no re-hold delay.
                    self._turn_off_since = None
                    self._reset_activation()
                    self.state = MotionState.FORWARD
            return self.state

        return self.state


# ---------------------------------------------------------------------
# Command mapping
# ---------------------------------------------------------------------

STATE_TO_COMMAND = {
    MotionState.STOPPED: "S",
    MotionState.FORWARD: "F",
    MotionState.TURN_LEFT: "L",
    MotionState.TURN_RIGHT: "R",
}


def log(message: str) -> None:
    if config.DEBUG:
        print(f"[main_logic2.py] {message}")


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

    motion_fsm = GazeMotionFSM(
        activation_hold_ms=config.ACTIVATION_HOLD_MS,
        flicker_tolerance_ms=config.MOTION_FLICKER_TOLERANCE_MS,
        on_activate=lambda new_state: (_play_beep(), log(f"ACTIVATED -> {new_state}")),
    )

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

    log(f"Entering main control loop (LOGIC 2 -- smooth). "
        f"Activation hold: {config.ACTIVATION_HOLD_MS}ms, "
        f"flicker tolerance: {config.MOTION_FLICKER_TOLERANCE_MS}ms.")

    prev_time = time.time()
    last_command = "-"
    no_face_warned = False

    running = True
    while running:
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
            blink_stop_event = blink_detector.get_stop_event() or blink_detector.is_holding_stop()

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

            if hard_stop:
                motion_fsm.state = MotionState.STOPPED
                motion_fsm._reset_activation()
                command = "S"
                if fail_safe_result.triggered:
                    log(f"HARD STOP: fail-safe ({fail_safe_result.reason}).")
                else:
                    log("HARD STOP: joystick emergency button.")
            elif arbiter_decision.source == "JOYSTICK":
                command = joystick_to_command(arbiter_decision.direction_or_command)
                motion_fsm.state = MotionState.STOPPED
                motion_fsm._reset_activation()
            elif blink_state != BlinkState.OPEN and not blink_stop_event:
                command = STATE_TO_COMMAND[motion_fsm.state]
            else:
                new_state = motion_fsm.update(raw_direction, blink_stop_event)
                command = STATE_TO_COMMAND[new_state]

            if command != last_command:
                log(f"Command changed: {last_command} -> {command}  "
                    f"(motion_state={motion_fsm.state}, direction={raw_direction}, "
                    f"source={arbiter_decision.source}, blink={blink_state})")
                last_command = command

            state_name = motion_fsm.state if motion_fsm.state != MotionState.STOPPED else "STOP"
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
                "Trial_ID": "logic2",
                "Horizontal_Ratio": eye_features.horizontal_ratio,
                "Vertical_Ratio": eye_features.vertical_ratio,
                "EAR": eye_features.ear,
                "Eye_Direction": raw_direction,
                "Dwell_Confirmed": motion_fsm.state != MotionState.STOPPED,
                "Blink_Duration_ms": None,
                "Input_Source": arbiter_decision.source,
                "FSM_State": motion_fsm.state,
                "Wheelchair_Command": command,
                "Left_PWM": left_pwm,
                "Right_PWM": right_pwm,
                "Fail_Safe_Triggered": fail_safe_result.reason,
                "FPS": fps,
            })

            display.render(frame, motion_fsm.state, raw_direction,
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
