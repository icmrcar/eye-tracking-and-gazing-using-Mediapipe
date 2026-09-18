import cv2
import mediapipe as mp
import numpy as np
import os
import time

# -----------------------------
# Dataset folders
# -----------------------------
labels = ["left", "right", "up", "down", "center"]

for label in labels:
    os.makedirs(f"dataset/p1/day02/{label}", exist_ok=True)

current_label = None

# -----------------------------
# Mediapipe
# -----------------------------
mp_face = mp.solutions.face_mesh

LEFT_EYE = [33,133,160,159,158,157,173,246]
RIGHT_EYE = [362,263,387,386,385,384,398,466]

cap = cv2.VideoCapture(0)

save_interval = 1.0
last_save = time.time()

with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        both_eyes = None

        if results.multi_face_landmarks:

            h, w, _ = frame.shape

            lm = results.multi_face_landmarks[0]

            left_pts = []
            right_pts = []

            for idx in LEFT_EYE:
                x = int(lm.landmark[idx].x * w)
                y = int(lm.landmark[idx].y * h)
                left_pts.append((x, y))

            for idx in RIGHT_EYE:
                x = int(lm.landmark[idx].x * w)
                y = int(lm.landmark[idx].y * h)
                right_pts.append((x, y))

            left_pts = np.array(left_pts)
            right_pts = np.array(right_pts)

            lx, ly, lw, lh = cv2.boundingRect(left_pts)
            rx, ry, rw, rh = cv2.boundingRect(right_pts)

            pad = 20

            left_eye = frame[
                max(0, ly-pad):ly+lh+pad,
                max(0, lx-pad):lx+lw+pad
            ]

            right_eye = frame[
                max(0, ry-pad):ry+rh+pad,
                max(0, rx-pad):rx+rw+pad
            ]

            if left_eye.size and right_eye.size:

                left_eye = cv2.resize(left_eye, (120, 60))
                right_eye = cv2.resize(right_eye, (120, 60))

                both_eyes = np.hstack((left_eye, right_eye))

                cv2.imshow("Both Eyes", both_eyes)

        # Save image
        if both_eyes is not None and current_label is not None:

            if time.time() - last_save >= save_interval:

                folder = f"dataset/p1/day02/{current_label}"

                count = len(os.listdir(folder))

                filename = os.path.join(
                    folder,
                    f"{current_label}_{count:06d}.jpg"
                )

                cv2.imwrite(filename, both_eyes)

                print("Saved:", filename)

                last_save = time.time()

        info = f"Label: {current_label}"
        cv2.putText(frame,
                    info,
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('l'):
            current_label = "left"

        elif key == ord('r'):
            current_label = "right"

        elif key == ord('u'):
            current_label = "up"

        elif key == ord('d'):
            current_label = "down"

        elif key == ord('c'):
            current_label = "center"

        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()