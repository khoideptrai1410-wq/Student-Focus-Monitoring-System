"""Collect YOLO phone-candidate crops and label them with keyboard shortcuts.

Controls:
  P: save selected candidate as PHONE
  N: save selected candidate as NOT_PHONE
  S: select next candidate when multiple boxes exist
  Q: quit

Images are saved under data/phone_classifier/raw/{PHONE,NOT_PHONE}.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Collect labeled phone candidates")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights path")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO candidate confidence")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO inference image size")
    parser.add_argument(
        "--out-dir",
        default="data/phone_classifier/raw",
        help="Output directory containing PHONE and NOT_PHONE folders",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.out_dir)
    phone_dir = output_root / "PHONE"
    not_phone_dir = output_root / "NOT_PHONE"
    phone_dir.mkdir(parents=True, exist_ok=True)
    not_phone_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {model_path.resolve()}")

    model = YOLO(str(model_path))

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index and permissions.")

    print("Phone candidate collector started.")
    print("P = PHONE | N = NOT_PHONE | S = next candidate | Q = quit")
    print(f"Saving images to: {output_root.resolve()}")

    selected_index = 0
    saved_counts = {
        "PHONE": len(list(phone_dir.glob("*.jpg"))),
        "NOT_PHONE": len(list(not_phone_dir.glob("*.jpg"))),
    }

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue

            results = model.predict(
                source=frame,
                classes=[67],
                conf=args.conf,
                imgsz=args.imgsz,
                device="cpu",
                verbose=False,
            )

            candidates = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0].item())
                    if x2 > x1 and y2 > y1:
                        candidates.append((x1, y1, x2, y2, conf))

            if candidates:
                selected_index %= len(candidates)
            else:
                selected_index = 0

            display = frame.copy()
            for index, candidate in enumerate(candidates):
                x1, y1, x2, y2, conf = candidate
                color = (0, 255, 255) if index == selected_index else (255, 160, 0)
                thickness = 3 if index == selected_index else 1
                cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
                label = f"Candidate {index + 1} | YOLO {conf:.2f}"
                cv2.putText(
                    display, label, (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
                )

            status = (
                f"Candidates: {len(candidates)} | Selected: {selected_index + 1 if candidates else 0}"
            )
            cv2.rectangle(display, (0, 0), (display.shape[1], 82), (25, 25, 25), -1)
            cv2.putText(display, status, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display, "P: PHONE   N: NOT_PHONE   S: NEXT   Q: QUIT",
                        (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Phone Candidate Collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and candidates:
                selected_index = (selected_index + 1) % len(candidates)
            elif key in (ord("p"), ord("n")):
                if not candidates:
                    print("No YOLO phone candidate in this frame; nothing saved.")
                    continue

                candidate = candidates[selected_index]
                x1, y1, x2, y2, conf = candidate
                height, width = frame.shape[:2]
                pad_x = int((x2 - x1) * 0.12)
                pad_y = int((y2 - y1) * 0.12)
                left = max(0, x1 - pad_x)
                top = max(0, y1 - pad_y)
                right = min(width, x2 + pad_x)
                bottom = min(height, y2 + pad_y)
                crop = frame[top:bottom, left:right]
                if crop.size == 0:
                    print("Candidate crop is empty; skipped.")
                    continue

                label = "PHONE" if key == ord("p") else "NOT_PHONE"
                saved_counts[label] += 1
                filename = (
                    f"{label.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
                    f"{saved_counts[label]:05d}.jpg"
                )
                destination = (phone_dir if label == "PHONE" else not_phone_dir) / filename
                if cv2.imwrite(str(destination), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    print(
                        f"Saved {label}: {destination} "
                        f"(YOLO confidence={conf:.2f}; totals={saved_counts})"
                    )
                else:
                    print(f"Could not save image: {destination}")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Collection finished. Final counts:", saved_counts)


if __name__ == "__main__":
    main()
