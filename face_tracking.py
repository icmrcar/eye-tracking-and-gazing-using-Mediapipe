"""
face_tracking.py

Wraps MediaPipe Face Mesh initialization and per-frame landmark/iris
extraction. Returns a structured landmark result or a "no face detected"
signal that fail_safe.py consumes.

Also estimates head pose (yaw/pitch/roll) using six stable facial landmarks
(nose tip, chin, eye corners, mouth corners) via cv2.solvePnP against a
generic 3D face model, and provides drawing helpers (bounding box, the 6
pose landmarks, and a 3D orientation axis) for the live dashboard.
"""

import math

import cv2
import numpy as np
import mediapipe as mp


# --- MediaPipe Face Mesh landmark index groups ---
# Order for both sets: [outer_corner, top_1, top_2, inner_corner, bottom_1, bottom_2]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Iris landmarks are only populated when refine_landmarks=True.
# Index 0 in each group is the iris center; 1-4 are the boundary points.
RIGHT_IRIS_IDX = [468, 469, 470, 471, 472]
LEFT_IRIS_IDX = [473, 474, 475, 476, 477]

# --- Head pose landmarks (nose tip, chin, eye corners, mouth corners) ---
HEAD_POSE_LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 263,
    "right_eye_corner": 33,
    "left_mouth_corner": 291,
    "right_mouth_corner": 61,
}

# Generic 3D face model points corresponding to the landmarks above.
_HEAD_POSE_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -330.0, -65.0),   # chin
    (-225.0, 170.0, -135.0),  # left eye corner
    (225.0, 170.0, -135.0),   # right eye corner
    (-150.0, -150.0, -125.0),  # left mouth corner
    (150.0, -150.0, -125.0),   # right mouth corner
], dtype=np.float64)


class FaceTracker:
    def __init__(self):
        self.face_mesh = None
        self._mp_face_mesh = mp.solutions.face_mesh

    def initialize(self) -> None:
        self.face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,   # required for iris landmarks (468-477)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[face_tracking.py] MediaPipe Face Mesh initialized (iris refinement ON)")

    def process_frame(self, frame):
        """
        Returns:
            FaceTrackingResult with face_detected, landmarks, eye/iris
            landmark subsets, and head_yaw/head_pitch/head_roll +
            head_rotation_vector/head_translation_vector (or None if pose
            estimation failed this frame).
        """
        if self.face_mesh is None:
            raise RuntimeError("FaceTracker.initialize() must be called before process_frame()")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return FaceTrackingResult(face_detected=False)

        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        def to_pixel_coords(indices):
            return [(face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z)
                    for i in indices]

        left_eye_landmarks = to_pixel_coords(LEFT_EYE_IDX)
        right_eye_landmarks = to_pixel_coords(RIGHT_EYE_IDX)
        left_iris_landmarks = to_pixel_coords(LEFT_IRIS_IDX)
        right_iris_landmarks = to_pixel_coords(RIGHT_IRIS_IDX)

        head_pose = self.compute_head_pose(face_landmarks, w, h)
        if head_pose is not None:
            head_yaw, head_pitch, head_roll, head_rvec, head_tvec = head_pose
        else:
            head_yaw = head_pitch = head_roll = head_rvec = head_tvec = None

        return FaceTrackingResult(
            face_detected=True,
            landmarks=face_landmarks,
            left_eye_landmarks=left_eye_landmarks,
            right_eye_landmarks=right_eye_landmarks,
            left_iris_landmarks=left_iris_landmarks,
            right_iris_landmarks=right_iris_landmarks,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            head_roll=head_roll,
            head_rotation_vector=head_rvec,
            head_translation_vector=head_tvec,
        )

    def compute_head_pose(self, face_landmarks, frame_width: int, frame_height: int):
        """
        Returns:
            (yaw, pitch, roll, rotation_vector, translation_vector), or
            None if pose could not be estimated this frame.
        """
        image_points = np.array([
            (face_landmarks[HEAD_POSE_LANDMARK_IDX["nose_tip"]].x * frame_width,
             face_landmarks[HEAD_POSE_LANDMARK_IDX["nose_tip"]].y * frame_height),
            (face_landmarks[HEAD_POSE_LANDMARK_IDX["chin"]].x * frame_width,
             face_landmarks[HEAD_POSE_LANDMARK_IDX["chin"]].y * frame_height),
            (face_landmarks[HEAD_POSE_LANDMARK_IDX["left_eye_corner"]].x * frame_width,
             face_landmarks[HEAD_POSE_LANDMARK_IDX["left_eye_corner"]].y * frame_height),
            (face_landmarks[HEAD_POSE_LANDMARK_IDX["right_eye_corner"]].x * frame_width,
             face_landmarks[HEAD_POSE_LANDMARK_IDX["right_eye_corner"]].y * frame_height),
            (face_landmarks[HEAD_POSE_LANDMARK_IDX["left_mouth_corner"]].x * frame_width,
             face_landmarks[HEAD_POSE_LANDMARK_IDX["left_mouth_corner"]].y * frame_height),
            (face_landmarks[HEAD_POSE_LANDMARK_IDX["right_mouth_corner"]].x * frame_width,
             face_landmarks[HEAD_POSE_LANDMARK_IDX["right_mouth_corner"]].y * frame_height),
        ], dtype=np.float64)

        focal_length = frame_width
        center = (frame_width / 2.0, frame_height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            _HEAD_POSE_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = 0.0

        return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll),
                rotation_vector, translation_vector)

    def close(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()
            print("[face_tracking.py] MediaPipe Face Mesh closed")


class FaceTrackingResult:
    def __init__(self, face_detected, landmarks=None,
                 left_eye_landmarks=None, right_eye_landmarks=None,
                 left_iris_landmarks=None, right_iris_landmarks=None,
                 head_yaw=None, head_pitch=None, head_roll=None,
                 head_rotation_vector=None, head_translation_vector=None):
        self.face_detected = face_detected
        self.landmarks = landmarks
        self.left_eye_landmarks = left_eye_landmarks
        self.right_eye_landmarks = right_eye_landmarks
        self.left_iris_landmarks = left_iris_landmarks
        self.right_iris_landmarks = right_iris_landmarks
        self.head_yaw = head_yaw
        self.head_pitch = head_pitch
        self.head_roll = head_roll
        self.head_rotation_vector = head_rotation_vector
        self.head_translation_vector = head_translation_vector


def get_face_bounding_box(tracking_result, frame_width, frame_height):
    """Pixel-space (x_min, y_min, x_max, y_max) bounding box of the face, or None."""
    if tracking_result.landmarks is None:
        return None
    landmarks_list = list(tracking_result.landmarks)[:468]
    xs = [lm.x * frame_width for lm in landmarks_list]
    ys = [lm.y * frame_height for lm in landmarks_list]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def get_head_pose_landmark_points(tracking_result, frame_width, frame_height):
    """Pixel-space points for the 6 landmarks used in head pose estimation, or None."""
    if tracking_result.landmarks is None:
        return None
    pts = []
    for key in ("nose_tip", "chin", "left_eye_corner", "right_eye_corner",
                "left_mouth_corner", "right_mouth_corner"):
        idx = HEAD_POSE_LANDMARK_IDX[key]
        lm = tracking_result.landmarks[idx]
        pts.append((int(lm.x * frame_width), int(lm.y * frame_height)))
    return pts


def get_head_pose_axis_points(tracking_result, frame_width, frame_height, axis_length=80.0):
    """
    Projects a 3D axis (origin=nose tip, X/Y/Z endpoints) into pixel space
    for drawing a visual pitch/yaw/roll gizmo. Returns
    [origin, x_end, y_end, z_end], or None if pose wasn't estimated.
    """
    if tracking_result.head_rotation_vector is None or tracking_result.head_translation_vector is None:
        return None

    focal_length = frame_width
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    axis_3d = np.array([
        (0.0, 0.0, 0.0),
        (axis_length, 0.0, 0.0),
        (0.0, axis_length, 0.0),
        (0.0, 0.0, axis_length),
    ], dtype=np.float64)

    image_points, _ = cv2.projectPoints(
        axis_3d, tracking_result.head_rotation_vector,
        tracking_result.head_translation_vector, camera_matrix, dist_coeffs,
    )
    pts = image_points.reshape(-1, 2)
    return [(int(p[0]), int(p[1])) for p in pts]


def draw_head_pose_overlay(frame, tracking_result):
    """
    Draws the face bounding box, the 6 head-pose landmark points, and a 3D
    axis gizmo (red=X, green=Y, blue=Z) at the nose tip directly onto
    frame, in place. No-op if face not detected or pose wasn't estimated.
    """
    if not tracking_result.face_detected:
        return

    h, w = frame.shape[:2]

    bbox = get_face_bounding_box(tracking_result, w, h)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    pose_points = get_head_pose_landmark_points(tracking_result, w, h)
    if pose_points is not None:
        for px, py in pose_points:
            cv2.circle(frame, (px, py), 3, (255, 0, 255), -1)

    axis_points = get_head_pose_axis_points(tracking_result, w, h)
    if axis_points is not None:
        origin, x_end, y_end, z_end = axis_points
        cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 2, tipLength=0.3)   # X: red
        cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), 2, tipLength=0.3)   # Y: green
        cv2.arrowedLine(frame, origin, z_end, (255, 0, 0), 2, tipLength=0.3)   # Z: blue
