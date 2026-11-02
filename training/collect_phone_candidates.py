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
import numpy as np
from ultralytics import YOLO


def perceptual_hash(image):
    """Return a compact hash for detecting visually near-identical crops."""
    if image is None or image.size == 0:
        raise ValueError("Cannot hash an empty image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    low_frequency = cv2.dct(np.float32(resized))[:8, :8].flatten()
    median = np.median(low_frequency[1:])

    value = 0
    for bit, coefficient in enumerate(low_frequency[1:]):
        if coefficient > median:
            value |= 1 << bit
    return value


def load_existing_hashes(directory):
    """Load hashes only, keeping image memory use independent of dataset size."""
    hashes = []
    for path in sorted(directory.glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None:
            print(f"Could not read existing image for duplicate check: {path}")
            continue
        hashes.append(perceptual_hash(image))
    return hashes


def is_near_duplicate(image_hash, existing_hashes, min_hash_distance):
    if min_hash_distance == 0:
        return False
    return any(
        (image_hash ^ existing_hash).bit_count() < min_hash_distance
        for existing_hash in existing_hashes
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Collect labeled phone candidates")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights path")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO candidate confidence")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO inference image size")
    parser.add_argument(
        "--min-hash-distance",
        type=int,
        default=6,
        help=(
            "Skip a crop if its perceptual hash is closer than this many bits "
            "to an image already saved in the same class; use 0 to disable"
        ),
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=100,
        help="Stop saving to a class after this many images; use 0 for no limit",
    )
    parser.add_argument(
        "--max-crop-size",
        type=int,
        default=512,
        help="Resize saved crops so their longest side is at most this many pixels",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for saved crops (1-100)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/phone_classifier/raw",
        help="Output directory containing PHONE and NOT_PHONE folders",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.min_hash_distance <= 63:
        raise ValueError("min-hash-distance must be between 0 and 63.")
    if args.max_images_per_class < 0:
        raise ValueError("max-images-per-class cannot be negative.")
    if args.max_crop_size < 0:
        raise ValueError("max-crop-size cannot be negative.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be between 1 and 100.")

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
    print(
        "Near-duplicate filtering: "
        f"{'disabled' if args.min_hash_distance == 0 else f'< {args.min_hash_distance} hash bits'}"
    )
    print(
        "Image budget per class: "
        f"{args.max_images_per_class or 'unlimited'}"
    )

    selected_index = 0
    saved_counts = {
        "PHONE": len(list(phone_dir.glob("*.jpg"))),
        "NOT_PHONE": len(list(not_phone_dir.glob("*.jpg"))),
    }
    duplicate_counts = {"PHONE": 0, "NOT_PHONE": 0}
    hashes_by_label = {
        "PHONE": load_existing_hashes(phone_dir),
        "NOT_PHONE": load_existing_hashes(not_phone_dir),
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
                if (
                    args.max_images_per_class
                    and saved_counts[label] >= args.max_images_per_class
                ):
                    print(
                        f"{label} image limit reached "
                        f"({args.max_images_per_class}); no more images saved."
                    )
                    continue

                crop_hash = perceptual_hash(crop)
                if is_near_duplicate(
                    crop_hash,
                    hashes_by_label[label],
                    args.min_hash_distance,
                ):
                    duplicate_counts[label] += 1
                    print(
                        f"Skipped near-duplicate {label} crop "
                        f"(skipped totals={duplicate_counts})"
                    )
                    continue

                save_crop = crop
                if args.max_crop_size:
                    crop_height, crop_width = crop.shape[:2]
                    longest_side = max(crop_height, crop_width)
                    if longest_side > args.max_crop_size:
                        scale = args.max_crop_size / longest_side
                        save_crop = cv2.resize(
                            crop,
                            (
                                max(1, round(crop_width * scale)),
                                max(1, round(crop_height * scale)),
                            ),
                            interpolation=cv2.INTER_AREA,
                        )

                saved_counts[label] += 1
                filename = (
                    f"{label.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
                    f"{saved_counts[label]:05d}.jpg"
                )
                destination = (phone_dir if label == "PHONE" else not_phone_dir) / filename
                if cv2.imwrite(
                    str(destination),
                    save_crop,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                ):
                    hashes_by_label[label].append(crop_hash)
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
    print("Near-duplicate crops skipped:", duplicate_counts)


if __name__ == "__main__":
    main()
