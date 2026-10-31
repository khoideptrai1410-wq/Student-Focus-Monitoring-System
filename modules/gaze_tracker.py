import cv2
import mediapipe as mp
import math


class GazeTracker:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Lưu landmarks để HeadPose sử dụng sau này
        self.last_landmarks = None

    def eye_aspect_ratio(self, top, bottom, left, right):

        vertical = math.dist(top, bottom)
        horizontal = math.dist(left, right)

        return vertical / (horizontal + 1e-6)

    def detect(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self.last_landmarks = None
            return "NO_FACE", 0

        face = results.multi_face_landmarks[0]

        # ===== Lưu landmarks =====
        self.last_landmarks = face.landmark

        left_outer = face.landmark[33]
        left_inner = face.landmark[133]
        left_iris = face.landmark[468]

        right_outer = face.landmark[362]
        right_inner = face.landmark[263]
        right_iris = face.landmark[473]

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

        h_ratio = (
            left_ratio +
            right_ratio
        ) / 2

        left_ear = self.eye_aspect_ratio(
            (
                face.landmark[159].x,
                face.landmark[159].y
            ),
            (
                face.landmark[145].x,
                face.landmark[145].y
            ),
            (
                face.landmark[33].x,
                face.landmark[33].y
            ),
            (
                face.landmark[133].x,
                face.landmark[133].y
            )
        )

        # ===== Detect DOWN =====
        if left_ear < 0.18:
            return "DOWN", 0.95

        # ===== Detect LEFT =====
        if h_ratio < 0.42:
            return "LEFT", 0.95

        # ===== Detect RIGHT =====
        if h_ratio > 0.58:
            return "RIGHT", 0.95

        # ===== Detect CENTER =====
        return "CENTER", 0.95