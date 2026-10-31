
import sys
from pathlib import Path

import cv2

# Xac dinh thu muc goc cua du an
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Them thu muc goc vao Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.person_detector import PersonDetector


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Khong mo duoc camera!")
        return

    print("Dang chup anh tu webcam...")

    ret, frame = camera.read()
    camera.release()

    if not ret or frame is None:
        print("ERROR: Khong doc duoc frame!")
        return

    detector = PersonDetector()

    # Phat hien nguoi
    persons = detector.detect(frame)

    print("So nguoi:", len(persons))

    display = frame.copy()

    # Ve bounding box
    for index, person in enumerate(persons):
        x1, y1, x2, y2, confidence = person

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"Person {confidence:.2f}"

        cv2.putText(
            display,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Chon nguoi chinh
    person = detector.get_primary_person(persons)

    if person is not None:
        roi = detector.crop_person(frame, person)

        if roi is not None:
            print("ROI shape:", roi.shape)
            cv2.imshow("Person ROI", roi)
        else:
            print("Khong cat duoc Person ROI!")
    else:
        print("Khong phat hien nguoi!")

    cv2.imshow("YOLO Person Detection", display)

    print("Nhan ESC de dong cua so.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
