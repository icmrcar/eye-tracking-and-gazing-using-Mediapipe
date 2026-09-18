"""
eye_features.py

Computes Horizontal Ratio, Vertical Ratio, and Eye Aspect Ratio (EAR) from
face/iris landmarks, and applies short-window smoothing.

Landmark ordering convention (matches face_tracking.py index groups):
    eye_landmarks:  [corner_A, top_1, top_2, corner_B, bottom_1, bottom_2]
    iris_landmarks: [center, boundary_1, boundary_2, boundary_3, boundary_4]
"""

import math
from collections import deque
from typing import Optional
import config


def _euclidean_2d(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


class EyeFeatureExtractor:
    def __init__(self):
        self._h_ratio_buffer = deque(maxlen=config.SMOOTHING_WINDOW_FRAMES)
        self._v_ratio_buffer = deque(maxlen=config.SMOOTHING_WINDOW_FRAMES)
        self._ear_buffer = deque(maxlen=config.SMOOTHING_WINDOW_FRAMES)

        self._last_smoothed = EyeFeatures(horizontal_ratio=0.5, vertical_ratio=0.5, ear=0.3)

        self._debug_call_count = 0
        self._debug_print_every_n = 20

    def compute_horizontal_ratio(self, eye_landmarks, iris_landmarks) -> float:
        """
        IMPORTANT: normalizes against the iris's *usable travel range*
        (eye_width - iris_diameter), not the full corner-to-corner eye
        width -- the iris physically can't reach the corners, so
        normalizing against the full width makes normal eye-only
        horizontal movement look like near-zero signal.
        """
        corner_a_x = eye_landmarks[0][0]
        corner_b_x = eye_landmarks[3][0]

        min_x, max_x = min(corner_a_x, corner_b_x), max(corner_a_x, corner_b_x)
        eye_width = max_x - min_x
        if eye_width == 0:
            return 0.5

        iris_center_x = iris_landmarks[0][0]
        iris_boundary_xs = [pt[0] for pt in iris_landmarks[1:5]]
        iris_diameter = max(iris_boundary_xs) - min(iris_boundary_xs)

        usable_span = eye_width - iris_diameter
        if usable_span <= 0:
            result = _clip01((iris_center_x - min_x) / eye_width)
        else:
            travel_min_x = min_x + iris_diameter / 2.0
            result = _clip01((iris_center_x - travel_min_x) / usable_span)

        if config.DEBUG:
            self._debug_call_count += 1
            if self._debug_call_count == 1 or self._debug_call_count % self._debug_print_every_n == 0:
                print(f"[eye_features.py][DEBUG] H calc #{self._debug_call_count}: "
                      f"corner_a_x={corner_a_x:.1f} corner_b_x={corner_b_x:.1f} "
                      f"eye_width={eye_width:.1f} iris_center_x={iris_center_x:.1f} "
                      f"iris_diameter={iris_diameter:.1f} usable_span={usable_span:.1f} "
                      f"-> ratio={result:.4f}")

        return result

    def compute_vertical_ratio(self, eye_landmarks, iris_landmarks) -> float:
        top_y = (eye_landmarks[1][1] + eye_landmarks[2][1]) / 2.0
        bottom_y = (eye_landmarks[4][1] + eye_landmarks[5][1]) / 2.0

        eyelid_gap = bottom_y - top_y
        if eyelid_gap == 0:
            return 0.5

        iris_center_y = iris_landmarks[0][1]
        iris_boundary_ys = [pt[1] for pt in iris_landmarks[1:5]]
        iris_diameter = max(iris_boundary_ys) - min(iris_boundary_ys)

        usable_span = eyelid_gap - iris_diameter
        if usable_span <= 0:
            return _clip01((iris_center_y - top_y) / eyelid_gap)

        travel_min_y = top_y + iris_diameter / 2.0
        return _clip01((iris_center_y - travel_min_y) / usable_span)

    def compute_ear(self, eye_landmarks) -> float:
        corner_a, top_1, top_2, corner_b, bottom_1, bottom_2 = eye_landmarks

        vertical_1 = _euclidean_2d(top_1, bottom_2)
        vertical_2 = _euclidean_2d(top_2, bottom_1)
        horizontal = _euclidean_2d(corner_a, corner_b)

        if horizontal == 0:
            return 0.3

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def get_smoothed_features(self, face_tracking_result) -> "EyeFeatures":
        """
        IMPORTANT: while the eye is closed (this frame's raw EAR below
        config.EAR_CLOSED_THRESHOLD), the H/V ratio buffers are NOT
        updated -- only the EAR buffer is. This prevents blink-corrupted
        samples from contaminating the smoothing window for several frames
        after the eye reopens (the root cause of gaze occasionally
        misreading as "UP" right after a blink).
        """
        if not face_tracking_result.face_detected:
            return self._last_smoothed

        left_eye = face_tracking_result.left_eye_landmarks
        right_eye = face_tracking_result.right_eye_landmarks
        left_iris = face_tracking_result.left_iris_landmarks
        right_iris = face_tracking_result.right_iris_landmarks

        ear_left = self.compute_ear(left_eye)
        ear_right = self.compute_ear(right_eye)
        raw_ear = (ear_left + ear_right) / 2.0

        self._ear_buffer.append(raw_ear)
        smoothed_ear = sum(self._ear_buffer) / len(self._ear_buffer)

        eye_is_open_this_frame = raw_ear >= config.EAR_CLOSED_THRESHOLD

        if eye_is_open_this_frame:
            h_left = self.compute_horizontal_ratio(left_eye, left_iris)
            h_right = self.compute_horizontal_ratio(right_eye, right_iris)
            v_left = self.compute_vertical_ratio(left_eye, left_iris)
            v_right = self.compute_vertical_ratio(right_eye, right_iris)

            raw_h = (h_left + h_right) / 2.0
            raw_v = (v_left + v_right) / 2.0

            self._h_ratio_buffer.append(raw_h)
            self._v_ratio_buffer.append(raw_v)

        if len(self._h_ratio_buffer) > 0:
            smoothed_h = sum(self._h_ratio_buffer) / len(self._h_ratio_buffer)
            smoothed_v = sum(self._v_ratio_buffer) / len(self._v_ratio_buffer)
        else:
            smoothed_h = self._last_smoothed.horizontal_ratio
            smoothed_v = self._last_smoothed.vertical_ratio

        self._last_smoothed = EyeFeatures(smoothed_h, smoothed_v, smoothed_ear, raw_ear=raw_ear)
        return self._last_smoothed

    def get_raw_features_unsmoothed(self, face_tracking_result) -> Optional["EyeFeatures"]:
        """Convenience method for calibration.py: this frame's raw ratios."""
        if not face_tracking_result.face_detected:
            return None

        left_eye = face_tracking_result.left_eye_landmarks
        right_eye = face_tracking_result.right_eye_landmarks
        left_iris = face_tracking_result.left_iris_landmarks
        right_iris = face_tracking_result.right_iris_landmarks

        h_left = self.compute_horizontal_ratio(left_eye, left_iris)
        h_right = self.compute_horizontal_ratio(right_eye, right_iris)
        v_left = self.compute_vertical_ratio(left_eye, left_iris)
        v_right = self.compute_vertical_ratio(right_eye, right_iris)
        ear_left = self.compute_ear(left_eye)
        ear_right = self.compute_ear(right_eye)

        return EyeFeatures(
            horizontal_ratio=(h_left + h_right) / 2.0,
            vertical_ratio=(v_left + v_right) / 2.0,
            ear=(ear_left + ear_right) / 2.0,
        )


class EyeFeatures:
    def __init__(self, horizontal_ratio: float, vertical_ratio: float, ear: float, raw_ear: float = None):
        self.horizontal_ratio = horizontal_ratio
        self.vertical_ratio = vertical_ratio
        self.ear = ear
        # Unsmoothed, this-frame-only EAR. blink_detector.py uses THIS, not
        # `ear`, for closure-duration timing -- smoothing would distort how
        # a closure's duration is measured.
        self.raw_ear = raw_ear if raw_ear is not None else ear
