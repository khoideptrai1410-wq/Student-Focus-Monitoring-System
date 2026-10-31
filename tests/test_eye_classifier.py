import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
import cv2

from ultralytics import YOLO
from modules.face_detector import FaceDetector

model = YOLO("models/eye_classifier.pt")

detector = FaceDetector()

cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = detector.detect_faces(frame)

    for (x, y, w, h) in faces:

        # ROI giống lúc tạo dataset
        eye_roi = frame[
            y + int(h * 0.25): y + int(h * 0.45),
            x + int(w * 0.18): x + int(w * 0.82)
        ]

        if eye_roi.size == 0:
            continue

        eye_roi = cv2.resize(
            eye_roi,
            (224, 112)
        )

        results = model.predict(
            eye_roi,
            verbose=False
        )

        result = results[0]

        cls_id = int(
            result.probs.top1
        )

        label = result.names[
            cls_id
        ]

        confidence = float(
            result.probs.top1conf
        )

        # vẽ ROI mắt trên frame
        cv2.rectangle(
            frame,
            (
                x + int(w * 0.18),
                y + int(h * 0.22)
            ),
            (
                x + int(w * 0.82),
                y + int(h * 0.42)
            ),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Eye ROI",
            eye_roi
        )

    cv2.imshow(
        "Eye Classifier Test",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()