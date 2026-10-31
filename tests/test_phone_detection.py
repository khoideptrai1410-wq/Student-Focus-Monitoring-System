
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Xac dinh thu muc goc du an
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():

    # Mo camera va chup mot frame
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Khong mo duoc camera!")
        return

    print("Hay cam dien thoai truoc camera.")

    input("Nhan ENTER de chup anh...")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Khong doc duoc frame!")
        return

    # Luu anh goc de kiem tra
    output_dir = PROJECT_ROOT / "evaluation"
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(
        str(output_dir / "phone_debug.jpg"),
        frame
    )

    # Tai YOLO
    model_path = PROJECT_ROOT / "yolov8n.pt"
    model = YOLO(str(model_path))

    # Cac cau hinh thu nghiem
    configurations = [
        (416, 0.25),
        (640, 0.25),
        (640, 0.10),
        (416, 0.10)
    ]

    for imgsz, confidence in configurations:

        print("\n================================")
        print("IMAGE SIZE:", imgsz)
        print("CONFIDENCE:", confidence)
        print("================================")

        results = model.predict(
            source=frame,
            classes=[67],
            conf=confidence,
            imgsz=imgsz,
            device="cpu",
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                score = float(box.conf[0].item())

                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().tolist()
                )

                detections.append(score)

                print(
                    "PHONE DETECTED:",
                    round(score, 4),
                    "BOX:",
                    (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    )
                )

        if not detections:
            print("KHONG PHAT HIEN DUOC DIEN THOAI")

        # Hien thi ket qua
        annotated = results[0].plot()

        cv2.imshow(
            f"YOLO Phone {imgsz} conf {confidence}",
            annotated
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nDa hoan thanh kiem tra.")
    print("Anh goc: evaluation/phone_debug.jpg")


if __name__ == "__main__":
    main()
