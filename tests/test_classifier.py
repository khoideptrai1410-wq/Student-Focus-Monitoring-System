from ultralytics import YOLO
import cv2

model = YOLO("models/focus_classifier.pt")

cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    probs = results[0].probs

    cls = probs.top1

    label = results[0].names[cls]

    conf = float(probs.top1conf)

    cv2.putText(
        frame,
        f"{label} {conf:.2f}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Focus Classifier",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()