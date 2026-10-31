import cv2
import mediapipe as mp
import math
from collections import deque, Counter

mp_face_mesh = mp.solutions.face_mesh


def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


history = deque(maxlen=5)

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:

        success, frame = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        gaze = "CENTER"

        if results.multi_face_landmarks:

            h, w, _ = frame.shape

            face = results.multi_face_landmarks[0]

            # =========================
            # LEFT EYE
            # =========================

            left_outer = face.landmark[33]
            left_inner = face.landmark[133]

            left_top = face.landmark[159]
            left_bottom = face.landmark[145]

            left_iris = face.landmark[468]

            # =========================
            # RIGHT EYE
            # =========================

            right_outer = face.landmark[362]
            right_inner = face.landmark[263]

            right_top = face.landmark[386]
            right_bottom = face.landmark[374]

            right_iris = face.landmark[473]

            # =========================
            # Horizontal Gaze
            # =========================

            left_ratio = (
                (left_iris.x - left_outer.x)
                /
                (left_inner.x - left_outer.x + 1e-6)
            )

            right_ratio = (
                (right_iris.x - right_outer.x)
                /
                (right_inner.x - right_outer.x + 1e-6)
            )

            horizontal_ratio = (
                left_ratio +
                right_ratio
            ) / 2

            # =========================
            # Eye Aspect Ratio (EAR)
            # =========================

            left_width = distance(
                left_outer,
                left_inner
            )

            left_height = distance(
                left_top,
                left_bottom
            )

            right_width = distance(
                right_outer,
                right_inner
            )

            right_height = distance(
                right_top,
                right_bottom
            )

            left_ear = (
                left_height /
                (left_width + 1e-6)
            )

            right_ear = (
                right_height /
                (right_width + 1e-6)
            )

            ear = (
                left_ear +
                right_ear
            ) / 2

            # =========================
            # Classification
            # =========================

            if ear < 0.18:
                gaze = "DOWN"

            elif horizontal_ratio < 0.42:
                gaze = "LEFT"

            elif horizontal_ratio > 0.58:
                gaze = "RIGHT"

            else:
                gaze = "CENTER"

            # =========================
            # Smoothing
            # =========================

            history.append(gaze)

            gaze = Counter(
                history
            ).most_common(1)[0][0]

            # =========================
            # Draw landmarks
            # =========================

            draw_points = [
                33,133,159,145,468,
                362,263,386,374,473
            ]

            for idx in draw_points:

                lm = face.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0,255,0),
                    -1
                )

            # Iris

            cv2.circle(
                frame,
                (
                    int(left_iris.x * w),
                    int(left_iris.y * h)
                ),
                5,
                (0,0,255),
                -1
            )

            cv2.circle(
                frame,
                (
                    int(right_iris.x * w),
                    int(right_iris.y * h)
                ),
                5,
                (0,0,255),
                -1
            )

            # Debug

            cv2.putText(
                frame,
                f"H:{horizontal_ratio:.2f}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )

            cv2.putText(
                frame,
                f"EAR:{ear:.2f}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )

        cv2.putText(
            frame,
            f"GAZE: {gaze}",
            (20,130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

        cv2.imshow(
            "FaceMesh Gaze Test",
            frame
        )

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()