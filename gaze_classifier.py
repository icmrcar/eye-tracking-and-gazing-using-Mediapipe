"""
gaze_classifier.py

Classifies raw eye direction (UP/LEFT/RIGHT/CENTER) from smoothed ratios,
applies the diagonal priority rule (vertical over horizontal), and applies
flicker-tolerant dwell-time confirmation for every direction.

NOTE: DOWN was removed as a classified direction -- looking down causes the
eyelid to partially occlude the iris, which fights with blink detection.
STOP is triggered by blink_detector.py (long blink or double blink)
instead of a gaze direction.
"""

import time
import config
from eye_features import EyeFeatures


class GazeDirection:
    UP = "UP"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    CENTER = "CENTER"


_DWELL_TIMES_MS = {
    GazeDirection.UP: config.DWELL_TIME_UP_MS,
    GazeDirection.LEFT: config.DWELL_TIME_LEFT_MS,
    GazeDirection.RIGHT: config.DWELL_TIME_RIGHT_MS,
    GazeDirection.CENTER: 0,
}


class GazeClassifier:
    def __init__(self):
        self._candidate_direction = GazeDirection.CENTER
        self._dwell_start_time = None
        self._off_frame_count = 0
        self._last_confirmed_direction = GazeDirection.CENTER

    def classify_raw_direction(self, eye_features: EyeFeatures) -> str:
        """
        Returns CENTER if calibration thresholds have not been set yet.
        UP takes priority over LEFT/RIGHT per the diagonal priority rule.
        """
        if config.VERTICAL_RATIO_UP_THRESHOLD is None or config.HORIZONTAL_RATIO_LEFT_THRESHOLD is None:
            return GazeDirection.CENTER

        v = eye_features.vertical_ratio
        h = eye_features.horizontal_ratio

        if self._is_up(v):
            return GazeDirection.UP
        if self._is_left(h):
            return GazeDirection.LEFT
        if self._is_right(h):
            return GazeDirection.RIGHT
        return GazeDirection.CENTER

    @staticmethod
    def _is_up(v: float) -> bool:
        threshold = config.VERTICAL_RATIO_UP_THRESHOLD
        return v < threshold if config.VERTICAL_UP_IS_LOWER else v > threshold

    @staticmethod
    def _is_left(h: float) -> bool:
        threshold = config.HORIZONTAL_RATIO_LEFT_THRESHOLD
        return h < threshold if config.HORIZONTAL_LEFT_IS_LOWER else h > threshold

    @staticmethod
    def _is_right(h: float) -> bool:
        threshold = config.HORIZONTAL_RATIO_RIGHT_THRESHOLD
        return h > threshold if config.HORIZONTAL_LEFT_IS_LOWER else h < threshold

    def update_dwell(self, raw_direction: str):
        """
        Flicker-tolerant dwell: a single off-direction frame does not reset
        the timer; only config.DWELL_FLICKER_TOLERANCE_FRAMES consecutive
        off-frames resets it. Returns the confirmed direction once dwell
        time is met, level-triggered thereafter.
        """
        now = time.time()

        if raw_direction == self._candidate_direction:
            self._off_frame_count = 0
        else:
            self._off_frame_count += 1
            if self._off_frame_count > config.DWELL_FLICKER_TOLERANCE_FRAMES:
                self._candidate_direction = raw_direction
                self._dwell_start_time = now
                self._off_frame_count = 0

        if self._dwell_start_time is None:
            self._dwell_start_time = now

        required_ms = _DWELL_TIMES_MS.get(self._candidate_direction, 0)
        elapsed_ms = (now - self._dwell_start_time) * 1000

        if elapsed_ms >= required_ms:
            self._last_confirmed_direction = self._candidate_direction
            return self._candidate_direction

        return None

    def reset(self) -> None:
        """
        Resets dwell-tracking state to a fresh CENTER baseline. Called by
        main.py on the rising edge of a blink-triggered STOP, so dwell
        progress accumulated before the blink can't let the chair instantly
        resume the moment STOP clears.
        """
        self._candidate_direction = GazeDirection.CENTER
        self._dwell_start_time = None
        self._off_frame_count = 0

    def get_confirmed_direction(self, eye_features: EyeFeatures):
        raw_direction = self.classify_raw_direction(eye_features)
        return self.update_dwell(raw_direction)
