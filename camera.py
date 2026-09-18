"""
camera.py

Handles webcam initialization and per-frame capture.
No gaze/vision logic lives here -- this module's only job is to reliably
hand raw frames to face_tracking.py.
"""

import cv2
import config


class Camera:
    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"[camera.py] ERROR: Could not open camera at index {self.camera_index}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        success, _ = self.cap.read()
        if not success:
            print("[camera.py] ERROR: Camera opened but failed to read a test frame")
            self.cap.release()
            return False

        print(f"[camera.py] Camera {self.camera_index} opened successfully "
              f"({config.FRAME_WIDTH}x{config.FRAME_HEIGHT} @ {config.TARGET_FPS}fps target)")
        return True

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None

        success, frame = self.cap.read()
        if not success:
            return False, None

        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            print("[camera.py] Camera released")
