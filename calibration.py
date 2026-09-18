"""
calibration.py

Runs the initial startup calibration routine (user looks at defined
reference points: center, left, right, up) to set the threshold values in
config.py, and runs event-triggered re-centering after joystick override
periods end.

NOTE: DOWN was removed as a calibrated direction -- looking down partially
occludes the iris with the eyelid, which fights with blink detection. STOP
is triggered by blink_detector.py instead -- only CENTER, LEFT, RIGHT, UP
are calibrated here.

Features:
    - Step counter, beep cue before each direction's countdown, numeric
      countdown, live sample counter + live H/V readout
    - Outlier rejection (statistical) + per-direction retry (up to 3x)
    - Separation-quality check (LEFT<->RIGHT, CENTER<->each extreme) with
      specific per-direction warnings
    - Head-pose stability enforcement (solvePnP): sample collection
      actively PAUSES whenever the head drifts from the CENTER-step
      baseline by more than config.HEAD_POSE_TOLERANCE_DEG -- the timer
      freezes and no samples are collected until the head is stable again
"""

import json
import math
import os
import time

import cv2
import numpy as np

import config
from face_tracking import draw_head_pose_overlay

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

# --- Beep sound (plays right when a new direction step begins, before the
# countdown, as an audible cue to prepare/change eye direction) ---
_BEEP_FREQ_HZ = 880
_BEEP_DURATION_S = 0.15
_BEEP_SAMPLE_RATE = 44100
_beep_sound = None
_beep_init_attempted = False


def _init_beep() -> None:
    global _beep_sound, _beep_init_attempted
    if _beep_init_attempted:
        return
    _beep_init_attempted = True

    if not _PYGAME_AVAILABLE:
        print("[calibration.py] NOTE: pygame not installed -- beep cue disabled.")
        return

    try:
        pygame.mixer.init(frequency=_BEEP_SAMPLE_RATE, size=-16, channels=1)

        t = np.linspace(0, _BEEP_DURATION_S,
                         int(_BEEP_SAMPLE_RATE * _BEEP_DURATION_S), False)
        wave = np.sin(_BEEP_FREQ_HZ * t * 2 * np.pi)

        fade_len = max(1, int(0.01 * _BEEP_SAMPLE_RATE))
        envelope = np.ones_like(wave)
        envelope[:fade_len] = np.linspace(0.0, 1.0, fade_len)
        envelope[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
        wave = wave * envelope

        audio = (wave * 32767 * 0.5).astype(np.int16)
        _beep_sound = pygame.sndarray.make_sound(audio)
    except Exception as e:
        print(f"[calibration.py] NOTE: Beep cue unavailable ({e}). "
              f"Calibration will continue without sound.")
        _beep_sound = None


def _play_beep() -> None:
    _init_beep()
    if _beep_sound is not None:
        try:
            _beep_sound.play()
        except Exception:
            pass


# Order matters: CENTER first so we always have a baseline reference.
CALIBRATION_SEQUENCE = ["CENTER", "LEFT", "RIGHT", "UP"]

PREP_SECONDS = 3.0
SAMPLE_SECONDS = 2.5
MIN_SAMPLES_REQUIRED = 8
MAX_RETRIES_PER_DIRECTION = 3
OUTLIER_STD_MULTIPLIER = 2.0

MIN_SEPARATION = 0.08
MIN_CENTER_SEPARATION = MIN_SEPARATION / 2.0
MAX_ASYMMETRY_RATIO = 2.5

MAX_CAPTURE_WALLCLOCK_MULTIPLIER = 4

WINDOW_NAME = "Calibration"


class Calibration:
    def __init__(self):
        self._reference_points = {}
        self._sample_counts = {}
        self._recenter_pending = False
        self._center_head_pose = None

    # ------------------------------------------------------------------
    # Initial calibration
    # ------------------------------------------------------------------

    def run_initial_calibration(self, camera, face_tracker, eye_feature_extractor) -> bool:
        cv2.namedWindow(WINDOW_NAME)

        total_steps = len(CALIBRATION_SEQUENCE)
        for step_index, direction in enumerate(CALIBRATION_SEQUENCE, start=1):
            attempt = 1
            while True:
                head_pose_baseline = self._center_head_pose if direction != "CENTER" else None

                cancelled, samples = self._run_direction_capture(
                    camera, face_tracker, eye_feature_extractor,
                    direction, step_index, total_steps, attempt, head_pose_baseline,
                )

                if cancelled:
                    print("[calibration.py] Calibration cancelled by user (ESC).")
                    cv2.destroyWindow(WINDOW_NAME)
                    return False

                head_stable_samples = self._filter_by_head_pose(samples, head_pose_baseline)
                rejected_for_head_movement = len(samples) - len(head_stable_samples)

                clean_samples = self._reject_outliers(head_stable_samples)

                if len(clean_samples) >= MIN_SAMPLES_REQUIRED:
                    avg_h = sum(s[0] for s in clean_samples) / len(clean_samples)
                    avg_v = sum(s[1] for s in clean_samples) / len(clean_samples)
                    self._reference_points[direction] = (avg_h, avg_v)
                    self._sample_counts[direction] = len(clean_samples)

                    extra_note = ""
                    if rejected_for_head_movement > 0:
                        extra_note = f", {rejected_for_head_movement} rejected for head movement"
                    print(f"[calibration.py] {direction}: h_ratio={avg_h:.3f}, "
                          f"v_ratio={avg_v:.3f} "
                          f"({len(clean_samples)} valid samples, "
                          f"{len(head_stable_samples) - len(clean_samples)} outliers rejected"
                          f"{extra_note})")

                    if direction == "CENTER":
                        self._establish_head_pose_baseline(samples)

                    break

                print(f"[calibration.py] WARNING: Only {len(clean_samples)}/"
                      f"{MIN_SAMPLES_REQUIRED} valid samples for {direction} "
                      f"(attempt {attempt}/{MAX_RETRIES_PER_DIRECTION})"
                      + (f" -- {rejected_for_head_movement} were rejected for head "
                         f"movement, try holding your head still." if rejected_for_head_movement > 0 else "."))

                if attempt >= MAX_RETRIES_PER_DIRECTION:
                    print(f"[calibration.py] ERROR: {direction} failed after "
                          f"{MAX_RETRIES_PER_DIRECTION} attempts. Aborting calibration.")
                    cv2.destroyWindow(WINDOW_NAME)
                    return False

                cancelled = self._show_retry_prompt(camera, direction)
                if cancelled:
                    print("[calibration.py] Calibration cancelled by user (ESC).")
                    cv2.destroyWindow(WINDOW_NAME)
                    return False

                attempt += 1

        cv2.destroyWindow(WINDOW_NAME)

        separations_ok = self._check_separation()
        if not separations_ok:
            print("[calibration.py] ERROR: Calibration separation too small for "
                  "reliable direction classification (see WARNINGs above for "
                  "which specific direction(s) failed). Please re-run "
                  "calibration -- look further toward each extreme, and make "
                  "sure you're looking straight ahead (not already off to one "
                  "side) during the CENTER step.")
            return False

        self._derive_thresholds()
        print("[calibration.py] Calibration complete.")
        return True

    def _run_direction_capture(self, camera, face_tracker, eye_feature_extractor,
                                direction, step_index, total_steps, attempt,
                                head_pose_baseline):
        """
        Returns:
            (cancelled: bool, samples: list of (h_ratio, v_ratio, yaw, pitch))
        """
        _play_beep()

        # --- Countdown phase ---
        countdown_start = time.time()
        while True:
            elapsed = time.time() - countdown_start
            remaining = PREP_SECONDS - elapsed
            if remaining <= 0:
                break

            success, frame = camera.read_frame()
            if success:
                head_pose = None
                tracking_result = None
                if head_pose_baseline is not None:
                    tracking_result = face_tracker.process_frame(frame)
                    if tracking_result.face_detected and tracking_result.head_yaw is not None:
                        head_pose = (tracking_result.head_yaw, tracking_result.head_pitch)
                if tracking_result is not None:
                    draw_head_pose_overlay(frame, tracking_result)

                countdown_number = max(1, math.ceil(remaining))
                self._draw_prompt(
                    frame, direction, step_index, total_steps, attempt,
                    sub_text=f"Get ready... {countdown_number}",
                    sample_count=None, live_features=None,
                    head_pose=head_pose, head_pose_baseline=head_pose_baseline,
                )
                cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return True, []

        # --- Sample collection phase (pauses on head instability) ---
        samples = []
        stable_elapsed = 0.0
        last_tick = time.time()
        was_paused = False

        capture_deadline = time.time() + (SAMPLE_SECONDS * MAX_CAPTURE_WALLCLOCK_MULTIPLIER)

        while stable_elapsed < SAMPLE_SECONDS:
            if time.time() > capture_deadline:
                if config.DEBUG:
                    print(f"[calibration.py] {direction}: capture timed out waiting "
                          f"for a stable head position.")
                break

            success, frame = camera.read_frame()
            if not success:
                continue

            tracking_result = face_tracker.process_frame(frame)
            raw_features = eye_feature_extractor.get_raw_features_unsmoothed(tracking_result)
            draw_head_pose_overlay(frame, tracking_result)

            head_pose = None
            if tracking_result.face_detected and tracking_result.head_yaw is not None:
                head_pose = (tracking_result.head_yaw, tracking_result.head_pitch)

            head_is_stable = self._is_head_stable(head_pose, head_pose_baseline)

            now = time.time()
            frame_dt = now - last_tick
            last_tick = now

            if head_is_stable:
                was_paused = False
                stable_elapsed += frame_dt

                if raw_features is not None:
                    yaw = head_pose[0] if head_pose is not None else None
                    pitch = head_pose[1] if head_pose is not None else None
                    samples.append((raw_features.horizontal_ratio, raw_features.vertical_ratio,
                                     yaw, pitch))
            else:
                if not was_paused and config.DEBUG:
                    print(f"[calibration.py] {direction}: PAUSED -- head moved, "
                          f"hold still to continue capturing.")
                was_paused = True

            self._draw_prompt(
                frame, direction, step_index, total_steps, attempt,
                sub_text="PAUSED -- hold head still..." if was_paused else "Hold your gaze...",
                sample_count=len(samples),
                live_features=raw_features,
                head_pose=head_pose,
                head_pose_baseline=head_pose_baseline,
                capture_paused=was_paused,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return True, samples

        return False, samples

    @staticmethod
    def _is_head_stable(head_pose, head_pose_baseline) -> bool:
        if head_pose_baseline is None:
            return True

        if head_pose is None:
            return False

        baseline_yaw, baseline_pitch = head_pose_baseline
        yaw, pitch = head_pose
        return (abs(yaw - baseline_yaw) <= config.HEAD_POSE_TOLERANCE_DEG
                and abs(pitch - baseline_pitch) <= config.HEAD_POSE_TOLERANCE_DEG)

    def _filter_by_head_pose(self, samples, head_pose_baseline):
        if head_pose_baseline is None:
            return samples

        baseline_yaw, baseline_pitch = head_pose_baseline
        kept = []
        for h, v, yaw, pitch in samples:
            if yaw is None or pitch is None:
                kept.append((h, v, yaw, pitch))
                continue
            if (abs(yaw - baseline_yaw) <= config.HEAD_POSE_TOLERANCE_DEG
                    and abs(pitch - baseline_pitch) <= config.HEAD_POSE_TOLERANCE_DEG):
                kept.append((h, v, yaw, pitch))
        return kept

    def _establish_head_pose_baseline(self, center_samples):
        valid = [(yaw, pitch) for _, _, yaw, pitch in center_samples
                 if yaw is not None and pitch is not None]

        if not valid:
            print("[calibration.py] NOTE: Head pose estimation unavailable "
                  "during CENTER step -- head-stability filtering will be "
                  "skipped for the rest of calibration.")
            self._center_head_pose = None
            return

        avg_yaw = sum(y for y, _ in valid) / len(valid)
        avg_pitch = sum(p for _, p in valid) / len(valid)
        self._center_head_pose = (avg_yaw, avg_pitch)
        print(f"[calibration.py] Head pose baseline set: "
              f"yaw={avg_yaw:.1f}deg, pitch={avg_pitch:.1f}deg "
              f"(tolerance: +/-{config.HEAD_POSE_TOLERANCE_DEG:.0f}deg)")

    def _show_retry_prompt(self, camera, direction) -> bool:
        prompt_end = time.time() + 2.0
        while time.time() < prompt_end:
            success, frame = camera.read_frame()
            if success:
                h, w = frame.shape[:2]
                cv2.putText(frame, f"Not enough samples for {direction}.",
                            (30, h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Retrying this step...",
                            (30, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return True

        return False

    @staticmethod
    def _draw_prompt(frame, direction, step_index, total_steps, attempt,
                      sub_text, sample_count, live_features,
                      head_pose=None, head_pose_baseline=None, capture_paused=False):
        h, w = frame.shape[:2]

        step_text = f"Step {step_index}/{total_steps}"
        if attempt > 1:
            step_text += f"  (attempt {attempt}/{MAX_RETRIES_PER_DIRECTION})"
        cv2.putText(frame, step_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2, cv2.LINE_AA)

        if capture_paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 150), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "PAUSED -- CAPTURE FROZEN", (w // 2 - 140, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Look {direction}", (30, h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, sub_text, (30, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if sample_count is not None:
            counter_color = (0, 255, 0) if sample_count >= MIN_SAMPLES_REQUIRED else (0, 165, 255)
            cv2.putText(frame, f"Samples: {sample_count}", (30, h // 2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, counter_color, 2, cv2.LINE_AA)

            bar_x, bar_y, bar_w, bar_h = 30, h // 2 + 60, 200, 12
            filled = int(min(1.0, sample_count / (MIN_SAMPLES_REQUIRED * 2)) * bar_w)
            bar_color = (0, 100, 200) if capture_paused else counter_color
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), bar_color, -1)

        if live_features is not None:
            readout = f"H:{live_features.horizontal_ratio:.3f}  V:{live_features.vertical_ratio:.3f}"
            cv2.putText(frame, readout, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2, cv2.LINE_AA)

        if head_pose is not None:
            yaw, pitch = head_pose
            if head_pose_baseline is not None:
                baseline_yaw, baseline_pitch = head_pose_baseline
                yaw_dev = abs(yaw - baseline_yaw)
                pitch_dev = abs(pitch - baseline_pitch)
                stable = (yaw_dev <= config.HEAD_POSE_TOLERANCE_DEG
                          and pitch_dev <= config.HEAD_POSE_TOLERANCE_DEG)
                head_color = (0, 200, 0) if stable else (0, 0, 255)
                head_label = "HEAD STABLE" if stable else "HOLD HEAD STILL"
                cv2.putText(frame, head_label, (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, head_color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Yaw:{yaw:.1f} (dev {yaw_dev:.1f}) "
                                    f"Pitch:{pitch:.1f} (dev {pitch_dev:.1f})",
                            (10, h - 42), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, head_color, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Yaw:{yaw:.1f}  Pitch:{pitch:.1f} (baseline)",
                            (10, h - 42), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1, cv2.LINE_AA)
        elif head_pose_baseline is not None and capture_paused:
            cv2.putText(frame, "HOLD HEAD STILL (pose lost)", (10, h - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "ESC to cancel", (w - 150, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 150, 150), 1, cv2.LINE_AA)

    @staticmethod
    def _reject_outliers(samples):
        if len(samples) < 4:
            return samples

        h_values = [s[0] for s in samples]
        v_values = [s[1] for s in samples]

        h_mean = sum(h_values) / len(h_values)
        v_mean = sum(v_values) / len(v_values)
        h_std = (sum((x - h_mean) ** 2 for x in h_values) / len(h_values)) ** 0.5
        v_std = (sum((y - v_mean) ** 2 for y in v_values) / len(v_values)) ** 0.5

        h_std = max(h_std, 1e-6)
        v_std = max(v_std, 1e-6)

        clean = [
            s for s in samples
            if abs(s[0] - h_mean) <= OUTLIER_STD_MULTIPLIER * h_std
            and abs(s[1] - v_mean) <= OUTLIER_STD_MULTIPLIER * v_std
        ]
        return clean

    def _check_separation(self) -> bool:
        center_h, center_v = self._reference_points["CENTER"]
        left_h, _ = self._reference_points["LEFT"]
        right_h, _ = self._reference_points["RIGHT"]
        _, up_v = self._reference_points["UP"]

        h_separation = abs(right_h - left_h)

        center_left_sep = abs(center_h - left_h)
        center_right_sep = abs(center_h - right_h)
        center_up_sep = abs(center_v - up_v)

        print("[calibration.py] --- Separation summary ---")
        print(f"    LEFT <-> RIGHT horizontal separation: {h_separation:.3f} "
              f"(minimum required: {MIN_SEPARATION:.3f})")
        print(f"    CENTER <-> LEFT:  {center_left_sep:.3f}   "
              f"CENTER <-> RIGHT: {center_right_sep:.3f}   "
              f"(minimum required: {MIN_CENTER_SEPARATION:.3f} each)")
        print(f"    CENTER <-> UP:    {center_up_sep:.3f}   "
              f"(minimum required: {MIN_CENTER_SEPARATION:.3f})")

        ok = True

        if h_separation < MIN_SEPARATION:
            print("    WARNING: LEFT/RIGHT separation too small -- horizontal "
                  "direction detection will be unreliable.")
            ok = False

        if center_left_sep < MIN_CENTER_SEPARATION:
            print("    WARNING: CENTER is too close to LEFT.")
            ok = False
        if center_right_sep < MIN_CENTER_SEPARATION:
            print("    WARNING: CENTER is too close to RIGHT.")
            ok = False
        if center_up_sep < MIN_CENTER_SEPARATION:
            print("    WARNING: CENTER is too close to UP.")
            ok = False

        if ok:
            h_ratio_asym = max(center_left_sep, center_right_sep) / max(min(center_left_sep, center_right_sep), 1e-6)
            if h_ratio_asym > MAX_ASYMMETRY_RATIO:
                print(f"    NOTE: CENTER is much closer to one horizontal side "
                      f"than the other ({h_ratio_asym:.1f}x).")

        return ok

    def _derive_thresholds(self):
        center_h, center_v = self._reference_points["CENTER"]
        left_h, _ = self._reference_points["LEFT"]
        right_h, _ = self._reference_points["RIGHT"]
        _, up_v = self._reference_points["UP"]

        config.HORIZONTAL_RATIO_LEFT_THRESHOLD = (center_h + left_h) / 2.0
        config.HORIZONTAL_RATIO_RIGHT_THRESHOLD = (center_h + right_h) / 2.0
        config.VERTICAL_RATIO_UP_THRESHOLD = (center_v + up_v) / 2.0

        config.HORIZONTAL_LEFT_IS_LOWER = left_h < center_h
        config.VERTICAL_UP_IS_LOWER = up_v < center_v

        print(f"[calibration.py] Thresholds derived: "
              f"H_LEFT={config.HORIZONTAL_RATIO_LEFT_THRESHOLD:.3f}, "
              f"H_RIGHT={config.HORIZONTAL_RATIO_RIGHT_THRESHOLD:.3f}, "
              f"V_UP={config.VERTICAL_RATIO_UP_THRESHOLD:.3f}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_calibration(self, path: str) -> None:
        data = {
            "reference_points": self._reference_points,
            "sample_counts": self._sample_counts,
            "thresholds": {
                "HORIZONTAL_RATIO_LEFT_THRESHOLD": config.HORIZONTAL_RATIO_LEFT_THRESHOLD,
                "HORIZONTAL_RATIO_RIGHT_THRESHOLD": config.HORIZONTAL_RATIO_RIGHT_THRESHOLD,
                "VERTICAL_RATIO_UP_THRESHOLD": config.VERTICAL_RATIO_UP_THRESHOLD,
                "HORIZONTAL_LEFT_IS_LOWER": config.HORIZONTAL_LEFT_IS_LOWER,
                "VERTICAL_UP_IS_LOWER": config.VERTICAL_UP_IS_LOWER,
            },
        }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[calibration.py] Calibration saved to {path}")

    def load_calibration(self, path: str) -> bool:
        if not os.path.exists(path):
            print(f"[calibration.py] No existing calibration file at {path}")
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            self._reference_points = data["reference_points"]
            self._sample_counts = data.get("sample_counts", {})
            thresholds = data["thresholds"]

            config.HORIZONTAL_RATIO_LEFT_THRESHOLD = thresholds["HORIZONTAL_RATIO_LEFT_THRESHOLD"]
            config.HORIZONTAL_RATIO_RIGHT_THRESHOLD = thresholds["HORIZONTAL_RATIO_RIGHT_THRESHOLD"]
            config.VERTICAL_RATIO_UP_THRESHOLD = thresholds["VERTICAL_RATIO_UP_THRESHOLD"]
            config.HORIZONTAL_LEFT_IS_LOWER = thresholds["HORIZONTAL_LEFT_IS_LOWER"]
            config.VERTICAL_UP_IS_LOWER = thresholds["VERTICAL_UP_IS_LOWER"]

            print(f"[calibration.py] Calibration loaded from {path}")
            return True
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[calibration.py] ERROR: Failed to load calibration file: {e}")
            return False

    # ------------------------------------------------------------------
    # Event-based re-centering
    # ------------------------------------------------------------------

    def trigger_recenter_if_needed(self, joystick_override_just_ended: bool) -> bool:
        if joystick_override_just_ended:
            self._recenter_pending = True

        if self._recenter_pending:
            self._recenter_pending = False
            return True

        return False
