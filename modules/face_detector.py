import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):

        self.mp_face_detection = mp.solutions.face_detection

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_detection.process(rgb)

        faces = []

        if results.detections:

            h, w, _ = frame.shape

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # tránh âm
                x = max(0, x)
                y = max(0, y)

                faces.append((x, y, bw, bh))

        return faces