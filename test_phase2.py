"""
test_phase2.py

Standalone test for Phase 2: camera + face tracking + eye feature ratios
(horizontal, vertical, EAR) + full calibration routine.

Usage:
    python test_phase2.py

Press 'q' to quit the live-readout phase.
"""

import cv2

from camera import Camera
from face_tracking import FaceTracker
from eye_features import EyeFeatureExtractor
from calibration import Calibration

CALIBRATION_FILE = "calibration_data.json"


def main():
    camera = Camera()
    face_tracker = FaceTracker()
    eye_extractor = EyeFeatureExtractor()
    calibration = Calibration()

    if not camera.open():
        print("Camera failed to open.")
        return
    face_tracker.initialize()

    if not calibration.load_calibration(CALIBRATION_FILE):
        print("Running initial calibration...")
        success = calibration.run_initial_calibration(camera, face_tracker, eye_extractor)
        if not success:
            print("Calibration failed. Exiting.")
            camera.release()
            face_tracker.close()
            return
        calibration.save_calibration(CALIBRATION_FILE)

    print("\nLive readout. Press 'q' to quit.\n")
    while True:
        success, frame = camera.read_frame()
        if not success:
            continue

        tracking_result = face_tracker.process_frame(frame)
        features = eye_extractor.get_smoothed_features(tracking_result)

        status = "FACE" if tracking_result.face_detected else "NO FACE"
        text = (f"{status} | H:{features.horizontal_ratio:.3f} "
                f"V:{features.vertical_ratio:.3f} EAR:{features.ear:.3f}")
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Phase 2 Test - Eye Features", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    face_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
