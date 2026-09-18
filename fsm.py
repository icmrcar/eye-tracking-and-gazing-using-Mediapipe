"""
fsm.py

LEGACY / NOT CURRENTLY USED. This was the original dwell-based finite
state machine. main.py, gaze_test.py, main_logic1.py, and main_logic2.py
all implement their own direct command logic and do NOT import this file.
Kept only for reference. Safe to delete if it causes confusion.

States: IDLE, FORWARD, TURN_LEFT, TURN_RIGHT, STOP
"""


class FSMState:
    IDLE = "IDLE"
    FORWARD = "FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"


class FiniteStateMachine:
    def __init__(self):
        self.state = FSMState.IDLE

    def update(self, arbiter_decision: "ArbiterDecision", blink_stop_triggered: bool,
               fail_safe_triggered: bool) -> str:
        if fail_safe_triggered:
            self.state = FSMState.STOP
            return self.state

        if arbiter_decision.source == "EMERGENCY_STOP":
            self.state = FSMState.STOP
            return self.state

        if blink_stop_triggered:
            self.state = FSMState.STOP
            return self.state

        if arbiter_decision.source == "JOYSTICK":
            self._update_from_joystick(arbiter_decision.direction_or_command)
        elif arbiter_decision.source == "GAZE":
            self._update_from_gaze(arbiter_decision.direction_or_command)

        return self.state

    def _update_from_joystick(self, joystick_command) -> None:
        if joystick_command is None or joystick_command.in_deadzone:
            return

        if joystick_command.y < -0.5:
            self.state = FSMState.FORWARD
        elif joystick_command.x < -0.5:
            self.state = FSMState.TURN_LEFT
        elif joystick_command.x > 0.5:
            self.state = FSMState.TURN_RIGHT

    def _update_from_gaze(self, direction) -> None:
        if direction is None:
            return

        if self.state == FSMState.IDLE:
            if direction == "UP":
                self.state = FSMState.FORWARD

        elif self.state == FSMState.FORWARD:
            if direction == "LEFT":
                self.state = FSMState.TURN_LEFT
            elif direction == "RIGHT":
                self.state = FSMState.TURN_RIGHT

        elif self.state in (FSMState.TURN_LEFT, FSMState.TURN_RIGHT):
            if direction == "CENTER":
                self.state = FSMState.FORWARD

        elif self.state == FSMState.STOP:
            if direction == "UP":
                self.state = FSMState.FORWARD

    def get_state(self) -> str:
        return self.state
