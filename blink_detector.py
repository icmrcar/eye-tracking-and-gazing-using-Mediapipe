"""
blink_detector.py

Tracks EAR-based eye closure to detect two independent STOP triggers:
    1. Long blink   -- eyes held closed past config.DELIBERATE_BLINK_MIN_MS
    2. Double blink  -- two natural blinks in quick succession, within
                        config.DOUBLE_BLINK_WINDOW_MS of each other

Closures shorter than config.MIN_BLINK_DURATION_MS are ignored entirely as
landmark/camera noise. Closures between MIN_BLINK_DURATION_MS and
config.NATURAL_BLINK_MAX_MS are eligible for double-blink pairing; anything
longer (but still short of DELIBERATE_BLINK_MIN_MS) is reported but not
paired, since it doesn't look like the quick double-tap gesture.

Either trigger sets a one-shot "stop event" that fsm/main.py uses to force
STOP. is_holding_stop() is an extra safety net for as long as eyes remain
closed past the deliberate threshold.
"""

import time
import config


class BlinkState:
    OPEN = "OPEN"
    CLOSING = "CLOSING"                    # eyes closed, not yet long enough to classify
    NATURAL_BLINK = "NATURAL_BLINK"        # short blink, ignored on its own
    DELIBERATE_BLINK = "DELIBERATE_BLINK"  # long blink just ended (reopen edge)
    HOLDING_CLOSED = "HOLDING_CLOSED"      # eyes still closed past deliberate threshold


class BlinkDetector:
    def __init__(self):
        self._closure_start_time = None
        self._current_state = BlinkState.OPEN

        self._last_natural_blink_end_time = None
        self._stop_event_pending = False

    def update(self, ear: float) -> str:
        now = time.time()
        is_closed = ear < config.EAR_CLOSED_THRESHOLD

        if is_closed:
            if self._closure_start_time is None:
                self._closure_start_time = now

            duration_ms = (now - self._closure_start_time) * 1000

            if duration_ms >= config.DELIBERATE_BLINK_MIN_MS:
                if self._current_state != BlinkState.HOLDING_CLOSED:
                    self._stop_event_pending = True
                    if config.DEBUG:
                        print("[blink_detector.py] Long blink detected -> STOP triggered.")
                self._current_state = BlinkState.HOLDING_CLOSED
            else:
                # Eyes ARE closed right now, so this must NOT be reported
                # as OPEN -- callers rely on CLOSING to hold the last
                # command instead of reading gaze from a partially
                # occluded iris mid-blink.
                self._current_state = BlinkState.CLOSING
        else:
            if self._closure_start_time is not None:
                duration_ms = (now - self._closure_start_time) * 1000

                if duration_ms >= config.DELIBERATE_BLINK_MIN_MS:
                    self._current_state = BlinkState.DELIBERATE_BLINK
                elif duration_ms < config.MIN_BLINK_DURATION_MS:
                    # Too short to be a real blink -- landmark/camera noise.
                    # Ignore completely, revert straight to OPEN.
                    self._current_state = BlinkState.OPEN
                    self._closure_start_time = None
                    return self._current_state
                else:
                    self._current_state = BlinkState.NATURAL_BLINK
                    if duration_ms <= config.NATURAL_BLINK_MAX_MS:
                        self._check_double_blink(now)

                self._closure_start_time = None
            else:
                self._current_state = BlinkState.OPEN

        return self._current_state

    def _check_double_blink(self, now: float) -> None:
        if self._last_natural_blink_end_time is not None:
            gap_ms = (now - self._last_natural_blink_end_time) * 1000
            if gap_ms <= config.DOUBLE_BLINK_WINDOW_MS:
                self._stop_event_pending = True
                self._last_natural_blink_end_time = None
                if config.DEBUG:
                    print(f"[blink_detector.py] Double blink detected "
                          f"(gap={gap_ms:.0f}ms) -> STOP triggered.")
                return

        self._last_natural_blink_end_time = now

    def get_stop_event(self) -> bool:
        """Returns True exactly once, on the frame a STOP should trigger, then resets."""
        triggered = self._stop_event_pending
        self._stop_event_pending = False
        return triggered

    def is_holding_stop(self) -> bool:
        """True while eyes remain closed past the deliberate threshold."""
        return self._current_state == BlinkState.HOLDING_CLOSED
