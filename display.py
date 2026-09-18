"""
display.py

Multi-panel live debugging dashboard, styled after a reference layout:
    - MASTER FEED: full camera view with eye/iris markers drawn on it
    - LEFT EYE / RIGHT EYE: zoomed-in crops of each eye, markers rescaled
    - GAZE POSITION: a small 2D plot showing where the gaze currently sits
      between LEFT / RIGHT / UP (this system has no reverse-gaze command,
      so the reference layout's "BACK" direction is replaced with CENTER)
    - STATUS PANEL: command, confidence, face/serial status, blink count,
      eye-openness, FPS, raw ratios + calibrated thresholds, per-frame
      timing, and recording state

IMPROVEMENTS over the plain text-overlay version this replaces:
    - Eye crops make it possible to SEE what the tracker sees, instead of
      trusting numbers alone -- much faster to spot a bad calibration or a
      landmark glitch.
    - The gaze quadrant plot gives an at-a-glance sense of how close the
      user is to a direction threshold, not just the final classified
      direction.
    - "CONF" is repurposed as hold-to-activate progress (0-100%) --this
      system doesn't have a per-frame classifier confidence score, but
      hold-progress is the more directly useful number here: it answers
      "how close is this gaze about to trigger a command?"
    - All fields are optional with safe defaults, so this still renders
      (in a reduced form) even if a caller doesn't pass every field.

WIRING NOTE FOR main_logic1.py / main_logic2.py:
    render() has new optional parameters: eye_features, blink_count,
    hold_progress_pct, serial_connected, frame_count, recording. Passing
    these gets you the full dashboard; omitting them just leaves those
    fields blank/zeroed rather than crashing. See the bottom of this file
    for the exact snippet to add to your main loop's Display.render() call.
"""

import time
import cv2
import numpy as np

# ---- Layout constants ----
RIGHT_COL_W = 350
EYE_PANEL_H = 120
GAZE_PANEL_H = 120
STATUS_PANEL_H = 200
CANVAS_H = EYE_PANEL_H * 2 + GAZE_PANEL_H + STATUS_PANEL_H  # right column drives total height

BG = (18, 18, 18)
GOLD = (0, 170, 220)      # BGR -- reads as amber/gold on screen
GREEN = (0, 220, 0)
RED = (0, 0, 220)
WHITE = (235, 235, 235)
GRAY = (130, 130, 130)
BLUE = (220, 140, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel_border(canvas, x, y, w, h, label, color=GOLD):
    cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), color, 1)
    cv2.putText(canvas, label, (x + 8, y + 20), FONT, 0.5, color, 1, cv2.LINE_AA)


def _fit_master_feed(frame, target_h):
    """
    Resizes frame to exactly target_h tall, deriving the width from the
    frame's own aspect ratio -- this fills the master panel completely
    with no letterbox bars, since the width is computed to match rather
    than assumed. Returns (resized_frame, scale_factor).
    """
    fh, fw = frame.shape[:2]
    scale = target_h / fh
    target_w = int(fw * scale)
    resized = cv2.resize(frame, (target_w, target_h))
    return resized, scale


def _draw_marker(img, x, y, size=7, color=GREEN, thickness=2):
    """Small '+' style marker matching the reference screenshot's eye markers."""
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)
    cv2.circle(img, (x - size, y), 2, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + size, y), 2, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y - size), 2, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y + size), 2, color, -1, cv2.LINE_AA)


def _eye_crop_panel(frame, eye_landmarks, iris_landmarks, panel_w, panel_h, label):
    """Crops a padded bounding box around one eye + iris, scales it up to
    fill the panel, and redraws the iris marker at the correct scaled
    position -- this is what makes the zoomed-in eye view genuinely useful
    for spotting tracking problems, not just decorative."""
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (10, 10, 10)

    if eye_landmarks is None or iris_landmarks is None:
        cv2.putText(panel, "NO EYE DATA", (panel_w // 2 - 70, panel_h // 2), FONT, 0.55, GRAY, 1, cv2.LINE_AA)
        _panel_border(panel, 0, 0, panel_w, panel_h, label)
        return panel

    xs = [p[0] for p in eye_landmarks]
    ys = [p[1] for p in eye_landmarks]
    pad_x = (max(xs) - min(xs)) * 0.6 + 8
    pad_y = (max(ys) - min(ys)) * 0.9 + 8
    x0, x1 = max(0, min(xs) - pad_x), max(xs) + pad_x
    y0, y1 = max(0, min(ys) - pad_y), max(ys) + pad_y

    fh, fw = frame.shape[:2]
    x0i, x1i = int(max(0, x0)), int(min(fw, x1))
    y0i, y1i = int(max(0, y0)), int(min(fh, y1))
    if x1i <= x0i or y1i <= y0i:
        cv2.putText(panel, "NO EYE DATA", (panel_w // 2 - 70, panel_h // 2), FONT, 0.55, GRAY, 1, cv2.LINE_AA)
        _panel_border(panel, 0, 0, panel_w, panel_h, label)
        return panel

    crop = frame[y0i:y1i, x0i:x1i]
    ch, cw = crop.shape[:2]
    scale = min(panel_w / cw, (panel_h - 24) / ch)
    new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized_crop = cv2.resize(crop, (new_w, new_h))

    y_off = 24 + ((panel_h - 24) - new_h) // 2
    x_off = (panel_w - new_w) // 2
    panel[y_off:y_off + new_h, x_off:x_off + new_w] = resized_crop

    def to_panel_xy(px, py):
        return (int((px - x0i) * scale) + x_off, int((py - y0i) * scale) + y_off)

    ix, iy = to_panel_xy(iris_landmarks[0][0], iris_landmarks[0][1])
    _draw_marker(panel, ix, iy, size=max(6, int(8 * scale / 2)), color=GREEN, thickness=2)

    _panel_border(panel, 0, 0, panel_w, panel_h, label)
    return panel


def _gaze_quadrant_panel(panel_w, panel_h, h_ratio, v_ratio, gaze_direction):
    """
    2D indicator: horizontal axis = LEFT <-> RIGHT (eye_features.horizontal_ratio),
    vertical axis = CENTER <-> UP/FORWARD (eye_features.vertical_ratio).
    Unlike the reference image, there is no BACK direction in this system
    -- looking down is not a classified direction (see gaze_classifier.py),
    so the lower half is simply labelled CENTER.
    """
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (10, 10, 10)
    _panel_border(panel, 0, 0, panel_w, panel_h, "GAZE POSITION")

    top = 30
    bottom = panel_h - 10
    left = 15
    right = panel_w - 15
    cx = (left + right) // 2
    cy = (top + bottom) // 2

    cv2.line(panel, (left, cy), (right, cy), GOLD, 1, cv2.LINE_AA)
    cv2.line(panel, (cx, top), (cx, bottom), (90, 90, 90), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (cx - 45, cy - 45), (cx + 45, cy + 45), (90, 90, 90), 1)

    cv2.putText(panel, "UP (FWD)", (cx - 34, top - 6), FONT, 0.4, WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, "LEFT", (left, cy - 6), FONT, 0.4, WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, "RIGHT", (right - 40, cy - 6), FONT, 0.4, WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, "CENTER", (cx - 28, bottom + 0), FONT, 0.4, GRAY, 1, cv2.LINE_AA)

    if h_ratio is not None and v_ratio is not None:
        px = int(left + h_ratio * (right - left))
        py = int(bottom - v_ratio * (bottom - top))
        px = max(left, min(right, px))
        py = max(top, min(bottom, py))
        dot_color = GOLD if gaze_direction in (None, "CENTER") else GREEN
        cv2.circle(panel, (px, py), 7, dot_color, -1, cv2.LINE_AA)
        cv2.circle(panel, (px, py), 7, WHITE, 1, cv2.LINE_AA)

    return panel


def _right_text(panel, text, y, panel_w, scale, color, thickness=1, margin=14):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(panel, text, (panel_w - tw - margin, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _status_panel(panel_w, panel_h, *, command, hold_progress_pct, face_detected,
                   frame_count, serial_connected, blink_count, ear, fps,
                   h_ratio, v_ratio, up_threshold, left_threshold, right_threshold,
                   proc_time_ms, control_source, recording):
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (10, 10, 10)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), GOLD, 1)

    cmd_color = GREEN if command == "F" else RED if command in ("S", "-") else BLUE
    cv2.putText(panel, f"CMD: {command}", (14, 30), FONT, 0.8, cmd_color, 2, cv2.LINE_AA)
    serial_txt, serial_color = ("Serial: OK", GREEN) if serial_connected else ("Serial: LOST", RED)
    _right_text(panel, serial_txt, 26, panel_w, 0.5, serial_color)

    y = 58
    face_txt = "FACE: TRACKING LOCKED" if face_detected else "FACE: LOST"
    face_color = GREEN if face_detected else RED
    cv2.putText(panel, face_txt, (14, y), FONT, 0.48, face_color, 1, cv2.LINE_AA)
    _right_text(panel, f"HOLD: {hold_progress_pct:3.0f}%", y, panel_w, 0.48, WHITE)

    y += 26
    cv2.putText(panel, f"track-run: {frame_count}f", (14, y), FONT, 0.46, GRAY, 1, cv2.LINE_AA)
    _right_text(panel, f"Blinks: {blink_count}", y, panel_w, 0.46, WHITE)

    y += 26
    cv2.putText(panel, f"EyeOpen: {ear:.3f}", (14, y), FONT, 0.46, WHITE, 1, cv2.LINE_AA)
    _right_text(panel, f"FPS(avg): {fps:.0f}", y, panel_w, 0.46, WHITE)

    y += 26
    cv2.putText(panel, f"H-ratio: {h_ratio:.3f}", (14, y), FONT, 0.46, WHITE, 1, cv2.LINE_AA)
    _right_text(panel, f"V-ratio: {v_ratio:.3f}", y, panel_w, 0.46, WHITE)

    y += 24
    up_txt = f"UP thr:{up_threshold:.3f}" if up_threshold is not None else "UP thr: --"
    l_txt = f"L thr:{left_threshold:.3f}" if left_threshold is not None else "L thr: --"
    r_txt = f"R thr:{right_threshold:.3f}" if right_threshold is not None else "R thr: --"
    cv2.putText(panel, up_txt, (14, y), FONT, 0.42, GRAY, 1, cv2.LINE_AA)
    cv2.putText(panel, l_txt, (170, y), FONT, 0.42, GRAY, 1, cv2.LINE_AA)
    _right_text(panel, r_txt, y, panel_w, 0.42, GRAY)

    y += 24
    cv2.putText(panel, f"Proc: {proc_time_ms:.1f}ms", (14, y), FONT, 0.44, GRAY, 1, cv2.LINE_AA)
    _right_text(panel, f"Source: {control_source}", y, panel_w, 0.44, GRAY)

    y += 30
    rec_txt, rec_color = ("REC: ON", RED) if recording else ("REC: OFF", GRAY)
    cv2.putText(panel, rec_txt, (14, y), FONT, 0.52, rec_color, 1, cv2.LINE_AA)
    if recording:
        cv2.circle(panel, (95, y - 6), 5, RED, -1, cv2.LINE_AA)

    return panel


class Display:
    def __init__(self, window_name: str = "Eye-Gaze Wheelchair Control"):
        self.window_name = window_name
        self._start_time = time.time()

    def render(self, frame, fsm_state: str, gaze_direction: str,
               control_source: str, command: str, fps: float,
               tracking_result=None, eye_features=None,
               blink_count: int = 0, hold_progress_pct: float = 0.0,
               serial_connected: bool = True, frame_count: int = 0,
               recording: bool = True, proc_time_ms: float = 0.0) -> None:
        if frame is None:
            return

        master, scale = _fit_master_feed(frame.copy(), CANVAS_H)
        x_off, y_off = 0, 0
        master_w = master.shape[1]

        face_detected = bool(tracking_result and tracking_result.face_detected)
        if face_detected:
            for iris in (tracking_result.left_iris_landmarks, tracking_result.right_iris_landmarks):
                px = int(iris[0][0] * scale) + x_off
                py = int(iris[0][1] * scale) + y_off
                _draw_marker(master, px, py, size=10, color=GREEN, thickness=2)

        _panel_border(master, 0, 0, master_w, CANVAS_H, "MASTER FEED")

        left_eye = tracking_result.left_eye_landmarks if face_detected else None
        right_eye = tracking_result.right_eye_landmarks if face_detected else None
        left_iris = tracking_result.left_iris_landmarks if face_detected else None
        right_iris = tracking_result.right_iris_landmarks if face_detected else None

        left_eye_panel = _eye_crop_panel(frame, left_eye, left_iris, RIGHT_COL_W, EYE_PANEL_H, "LEFT EYE")
        right_eye_panel = _eye_crop_panel(frame, right_eye, right_iris, RIGHT_COL_W, EYE_PANEL_H, "RIGHT EYE")

        h_ratio = eye_features.horizontal_ratio if eye_features else 0.5
        v_ratio = eye_features.vertical_ratio if eye_features else 0.5
        ear = eye_features.ear if eye_features else 0.0
        gaze_panel = _gaze_quadrant_panel(RIGHT_COL_W, GAZE_PANEL_H, h_ratio, v_ratio, gaze_direction)

        import config
        status = _status_panel(
            RIGHT_COL_W, STATUS_PANEL_H,
            command=command, hold_progress_pct=hold_progress_pct,
            face_detected=face_detected, frame_count=frame_count,
            serial_connected=serial_connected, blink_count=blink_count, ear=ear,
            fps=fps, h_ratio=h_ratio, v_ratio=v_ratio,
            up_threshold=config.VERTICAL_RATIO_UP_THRESHOLD,
            left_threshold=config.HORIZONTAL_RATIO_LEFT_THRESHOLD,
            right_threshold=config.HORIZONTAL_RATIO_RIGHT_THRESHOLD,
            proc_time_ms=proc_time_ms, control_source=control_source,
            recording=recording,
        )

        right_col = np.vstack([left_eye_panel, right_eye_panel, gaze_panel, status])
        canvas = np.hstack([master, right_col])

        cv2.imshow(self.window_name, canvas)

    def check_exit_key(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        return key == ord('q')

    def close(self) -> None:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# WIRING SNIPPET for main_logic1.py / main_logic2.py's main loop, where
# display.render(...) is currently called. Add the new keyword args to
# get the full dashboard (all are optional -- safe to add incrementally):
#
#   frame_count += 1
#   loop_start = time.time()
#   ... existing per-frame processing ...
#   proc_time_ms = (time.time() - loop_start) * 1000
#
#   display.render(
#       frame, motion_fsm.state, raw_direction, arbiter_decision.source,
#       command, fps, tracking_result=tracking_result,
#       eye_features=eye_features,
#       blink_count=blink_detector_total_count,   # tally this yourself, e.g. in blink_detector.py or main loop
#       hold_progress_pct=(elapsed_ms / required_ms * 100) if required_ms else 0,
#       serial_connected=serial_comm.is_connected(),
#       frame_count=frame_count,
#       recording=data_logger is not None,
#       proc_time_ms=proc_time_ms,
#   )
# ---------------------------------------------------------------------
