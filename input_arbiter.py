"""
input_arbiter.py

Decides, each control loop cycle, whether gaze or joystick has control
authority. Priority: emergency button > joystick active > joystick
neutral-hold window > gaze.
"""

import time
import config


class ControlSource:
    GAZE = "GAZE"
    JOYSTICK = "JOYSTICK"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class InputArbiter:
    def __init__(self):
        self._last_joystick_active_time = None
        self._last_joystick_command = None

    def resolve(self, gaze_direction, joystick_command: "JoystickCommand",
                face_detected: bool) -> "ArbiterDecision":
        now = time.time()

        if joystick_command.emergency_pressed:
            return ArbiterDecision(ControlSource.EMERGENCY_STOP, None)

        if not joystick_command.in_deadzone:
            self._last_joystick_active_time = now
            self._last_joystick_command = joystick_command
            return ArbiterDecision(ControlSource.JOYSTICK, joystick_command)

        if self._last_joystick_active_time is not None:
            elapsed_ms = (now - self._last_joystick_active_time) * 1000
            if elapsed_ms < config.JOYSTICK_NEUTRAL_HOLD_MS:
                return ArbiterDecision(ControlSource.JOYSTICK, self._last_joystick_command)

        if not face_detected:
            return ArbiterDecision(ControlSource.GAZE, None)

        return ArbiterDecision(ControlSource.GAZE, gaze_direction)

    def joystick_override_just_ended(self) -> bool:
        if self._last_joystick_active_time is None:
            return False
        elapsed_ms = (time.time() - self._last_joystick_active_time) * 1000
        return config.JOYSTICK_NEUTRAL_HOLD_MS <= elapsed_ms < config.JOYSTICK_NEUTRAL_HOLD_MS + 100


class ArbiterDecision:
    def __init__(self, source: str, direction_or_command):
        self.source = source
        self.direction_or_command = direction_or_command
