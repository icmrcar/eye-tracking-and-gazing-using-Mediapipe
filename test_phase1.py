"""
test_phase1.py

Standalone test for Phase 1 only: camera capture + MediaPipe face/iris
landmark detection + display overlay with landmark drawing.

Usage:
    python test_phase1.py

Press 'q' to quit.
"""

import cv2
import time

from camera import Camera
from face_tracking import FaceTracker


def draw_landmarks(frame, tracking_result):
    if not tracking_result.face_detected:
        return frame

    for x, y, z in tracking_result.left_eye_landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    for x, y, z in tracking_result.right_eye_landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    for x, y, z in tracking_result.left_iris_landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
    for x, y, z in tracking_result.right_iris_landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

    return frame


def main():
    camera = Camera()
    face_tracker = FaceTracker()

    if not camera.open():
        print("Camera failed to open. Check config.CAMERA_INDEX and permissions.")
        return

    face_tracker.initialize()

    print("Phase 1 test running. Press 'q' to quit.")

    prev_time = time.time()
    while True:
        success, frame = camera.read_frame()
        if not success:
            print("Frame read failed, skipping...")
            continue

        tracking_result = face_tracker.process_frame(frame)
        frame = draw_landmarks(frame, tracking_result)

        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now

        status_text = "FACE DETECTED" if tracking_result.face_detected else "NO FACE"
        status_color = (0, 200, 0) if tracking_result.face_detected else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Phase 1 Test - Camera + Face/Iris Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    face_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
