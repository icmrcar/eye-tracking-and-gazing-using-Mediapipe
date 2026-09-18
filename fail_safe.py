"""
fail_safe.py

Checks face-loss and serial-loss conditions BEFORE the command decision
each cycle, so fail-safe can force STOP and override everything else.
"""

import config


class FailSafeChecker:
    def __init__(self):
        self._consecutive_no_face_frames = 0

    def check(self, face_detected: bool, serial_connected: bool,
              last_serial_ack_time) -> "FailSafeResult":
        if face_detected:
            self._consecutive_no_face_frames = 0
        else:
            self._consecutive_no_face_frames += 1

        if self._consecutive_no_face_frames >= config.FACE_LOST_FRAME_THRESHOLD:
            return FailSafeResult(True, "FACE_LOST")

        if not serial_connected:
            return FailSafeResult(True, "SERIAL_LOST")

        return FailSafeResult(False, None)


class FailSafeResult:
    def __init__(self, triggered: bool, reason: str = None):
        self.triggered = triggered
        self.reason = reason
