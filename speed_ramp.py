"""
speed_ramp.py

Converts a wheelchair state into a gradually ramped PWM sequence rather
than an instant jump to full speed.
"""

import config

_TARGET_PWM_TABLE = {
    "STOP": (0, 0),
    "IDLE": (0, 0),
    "FORWARD": (config.FORWARD_PWM, config.FORWARD_PWM),
    "TURN_LEFT": (config.TURN_SLOW_PWM, config.TURN_FAST_PWM),
    "TURN_RIGHT": (config.TURN_FAST_PWM, config.TURN_SLOW_PWM),
}


class SpeedRamper:
    def __init__(self):
        self._current_left_pwm = 0.0
        self._current_right_pwm = 0.0

    def compute_target_pwm(self, fsm_state: str) -> tuple:
        return _TARGET_PWM_TABLE.get(fsm_state, (0, 0))

    def step(self, fsm_state: str, is_emergency: bool = False) -> tuple:
        if is_emergency:
            self._current_left_pwm = 0.0
            self._current_right_pwm = 0.0
            return 0, 0

        target_left, target_right = self.compute_target_pwm(fsm_state)

        steps_total = max(1, config.RAMP_DURATION_MS // config.RAMP_STEP_MS)
        left_increment = (target_left - self._current_left_pwm) / steps_total
        right_increment = (target_right - self._current_right_pwm) / steps_total

        self._current_left_pwm = self._advance(self._current_left_pwm, target_left, left_increment)
        self._current_right_pwm = self._advance(self._current_right_pwm, target_right, right_increment)

        return int(round(self._current_left_pwm)), int(round(self._current_right_pwm))

    @staticmethod
    def _advance(current: float, target: float, increment: float) -> float:
        new_value = current + increment
        if increment >= 0 and new_value > target:
            return target
        if increment < 0 and new_value < target:
            return target
        return new_value
