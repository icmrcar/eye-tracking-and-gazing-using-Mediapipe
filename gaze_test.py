"""
gaze_test.py

Standalone, simplified vision-pipeline test -- NO FSM module, NO joystick,
NO serial/ESP32. Just:

    Open camera -> Calibrate (or load saved calibration) -> Detect eyes ->
    Classify instant direction -> Map to F/L/R/S -> Show dashboard

Direction mapping (instant, no dwell):
    Look UP -> F, LEFT -> L, RIGHT -> R, CENTER -> -

STOP (S) is triggered by a long blink or a double blink, and latches until
you look UP again with eyes stably open.

Usage:
    python gaze_test.py

Press 'q' to quit, 'c' to force re-calibration, 'r' to reset blink counters.
"""

import time
import traceback

import cv2

import config
from camera import Camera
from face_tracking import FaceTracker, draw_head_pose_overlay
from eye_features import EyeFeatureExtractor
from calibration import Calibration
from gaze_classifier import GazeClassifier, GazeDirection
from blink_detector import BlinkDetector, BlinkState

CALIBRATION_FILE = "calibration_data.json"

DIRECTION_TO_COMMAND = {
    GazeDirection.UP: "F",
    GazeDirection.LEFT: "L",
    GazeDirection.RIGHT: "R",
    GazeDirection.CENTER: "-",
}

COMMAND_COLORS = {
    "F": (0, 200, 0),
    "L": (0, 165, 255),
    "R": (0, 165, 255),
    "S": (0, 0, 255),
    "-": (200, 200, 200),
}

BLINK_STATE_COLORS = {
    BlinkState.OPEN: (200, 200, 200),
    BlinkState.CLOSING: (0, 165, 255),
    BlinkState.NATURAL_BLINK: (0, 200, 255),
    BlinkState.DELIBERATE_BLINK: (0, 0, 255),
    BlinkState.HOLDING_CLOSED: (0, 0, 255),
}


def log(message: str) -> None:
    if config.DEBUG:
        print(f"[gaze_test.py] {message}")


def draw_landmarks(frame, tracking_result):
    if not tracking_result.face_detected:
        return
    for x, y, z in tracking_result.left_eye_landmarks + tracking_result.right_eye_landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    for x, y, z in tracking_result.left_iris_landmarks + tracking_result.right_iris_landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
    draw_head_pose_overlay(frame, tracking_result)


def draw_ratio_bar(frame, x, y, width, height, value, low_threshold, high_threshold, label):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (90, 90, 90), 1)

    def to_px(v):
        return x + int(max(0.0, min(1.0, v)) * width)

    if low_threshold is not None:
        px = to_px(low_threshold)
        cv2.line(frame, (px, y), (px, y + height), (255, 0, 0), 1)
    if high_threshold is not None:
        px = to_px(high_threshold)
        cv2.line(frame, (px, y), (px, y + height), (0, 0, 255), 1)

    marker_px = to_px(value)
    cv2.line(frame, (marker_px, y - 3), (marker_px, y + height + 3), (0, 255, 255), 2)

    cv2.putText(frame, f"{label}: {value:.3f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_dashboard(frame, tracking_result, eye_features, raw_direction, command,
                    blink_state, stopped, fps, blink_counts):
    h, w = frame.shape[:2]

    draw_landmarks(frame, tracking_result)

    banner_color = (0, 0, 150) if stopped else (0, 90, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), banner_color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    banner_text = "STOPPED -- look UP to resume" if stopped else "ACTIVE"
    cv2.putText(frame, banner_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)

    status_text = "FACE DETECTED" if tracking_result.face_detected else "NO FACE"
    status_color = (0, 200, 0) if tracking_result.face_detected else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)

    blink_color = BLINK_STATE_COLORS.get(blink_state, (255, 255, 255))
    cv2.putText(frame, f"Blink: {blink_state}", (10, 102), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, blink_color, 2, cv2.LINE_AA)

    counts_text = (f"Blinks: {blink_counts['total']}  "
                    f"Long: {blink_counts['long']}  "
                    f"Double: {blink_counts['double']}")
    cv2.putText(frame, counts_text, (10, 124), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if tracking_result.face_detected and tracking_result.head_yaw is not None:
        pose_text = (f"Head Yaw:{tracking_result.head_yaw:.1f}  "
                     f"Pitch:{tracking_result.head_pitch:.1f}  "
                     f"Roll:{tracking_result.head_roll:.1f}")
        cv2.putText(frame, pose_text, (10, 146), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

    command_color = COMMAND_COLORS.get(command, (255, 255, 255))
    box_x1, box_y1, box_x2, box_y2 = w - 110, 42, w - 15, 112
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (30, 30, 30), -1)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), command_color, 2)
    cv2.putText(frame, command, (box_x1 + 22, box_y2 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, command_color, 4, cv2.LINE_AA)

    label = "eyes must be OPEN + look UP" if stopped else (raw_direction or "-")
    cv2.putText(frame, label, (w - 280, 130), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, command_color, 2, cv2.LINE_AA)

    if eye_features is not None:
        draw_ratio_bar(
            frame, x=10, y=h - 70, width=220, height=14,
            value=eye_features.horizontal_ratio,
            low_threshold=config.HORIZONTAL_RATIO_LEFT_THRESHOLD,
            high_threshold=config.HORIZONTAL_RATIO_RIGHT_THRESHOLD,
            label="Horizontal",
        )
        draw_ratio_bar(
            frame, x=10, y=h - 30, width=220, height=14,
            value=eye_features.vertical_ratio,
            low_threshold=config.VERTICAL_RATIO_UP_THRESHOLD,
            high_threshold=None,
            label="Vertical (UP only)",
        )
        cv2.putText(frame, f"EAR: {eye_features.ear:.3f}", (250, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, "'q' quit  'c' recalibrate  'r' reset counts", (w - 320, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)


def run_calibration(camera, face_tracker, eye_extractor, calibration):
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

    log("Opening camera...")
    if not camera.open():
        log("ERROR: Camera failed to open. Check config.CAMERA_INDEX and permissions.")
        return

    log("Initializing MediaPipe Face Mesh...")
    face_tracker.initialize()

    if calibration.load_calibration(CALIBRATION_FILE):
        log(f"Loaded existing calibration from {CALIBRATION_FILE}.")
    else:
        log("No existing calibration found.")
        if not run_calibration(camera, face_tracker, eye_extractor, calibration):
            camera.release()
            face_tracker.close()
            return

    log("Entering main loop. Press 'q' to quit, 'c' to recalibrate, 'r' to reset blink counts.")

    window_name = "Gaze Test Dashboard"
    cv2.namedWindow(window_name)

    prev_time = time.time()
    last_command = "-"
    no_face_warned = False
    stopped = False
    blink_counts = {"total": 0, "long": 0, "double": 0}

    while True:
        try:
            success, frame = camera.read_frame()
            if not success:
                log("WARNING: Frame read failed, skipping this cycle.")
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

            newly_stopped = False
            if blink_stop_event and not stopped:
                stopped = True
                newly_stopped = True
                log(f"STOP triggered by blink (state={blink_state}). Look UP to resume.")

            if stopped:
                command = "S"
                if not newly_stopped and blink_state == BlinkState.OPEN \
                        and raw_direction == GazeDirection.UP:
                    stopped = False
                    command = DIRECTION_TO_COMMAND[GazeDirection.UP]
                    log("Resumed: UP detected with eyes stably open, STOP latch cleared.")
            elif blink_state != BlinkState.OPEN:
                command = last_command
            else:
                command = DIRECTION_TO_COMMAND.get(raw_direction, "-")

            if command != last_command:
                log(f"Command changed: {last_command} -> {command}  "
                    f"(direction={raw_direction}, blink={blink_state}, "
                    f"H={eye_features.horizontal_ratio:.3f}, "
                    f"V={eye_features.vertical_ratio:.3f})")
                last_command = command

            now = time.time()
            frame_time = now - prev_time
            fps = 1.0 / frame_time if frame_time > 0 else 0.0
            prev_time = now

            if fps < config.FPS_WARNING_THRESHOLD:
                log(f"WARNING: Low FPS ({fps:.1f}). Consider lowering FRAME_WIDTH/"
                    f"FRAME_HEIGHT in config.py.")

            draw_dashboard(frame, tracking_result, eye_features, raw_direction, command,
                            blink_state, stopped, fps, blink_counts)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                run_calibration(camera, face_tracker, eye_extractor, calibration)
                stopped = False
                last_command = "-"
            elif key == ord('r'):
                blink_counts = {"total": 0, "long": 0, "double": 0}
                log("Blink counters reset.")

        except Exception:
            log("ERROR: Exception during frame processing:")
            traceback.print_exc()
            continue

    camera.release()
    face_tracker.close()
    cv2.destroyAllWindows()
    log("Shutdown complete.")


if __name__ == "__main__":
    main()
