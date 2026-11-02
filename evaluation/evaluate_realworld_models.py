"""Evaluate PhoneClassifier V1 and Colab Candidate on a separate real-world crop set.

Expected dataset structure:
    D:\\PhoneClassifierRealWorldTest\\
        PHONE\\
        NOT_PHONE\\

Run from the project root:
    .venv\\Scripts\\python.exe evaluation\\evaluate_realworld_models.py

The script does not modify or overwrite any model checkpoint.
"""

from pathlib import Path
import argparse
import sys
import time

import cv2


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.phone_classifier import PhoneClassifier  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET = Path(r"D:\PhoneClassifierRealWorldTest")
DEFAULT_MODELS = {
    "V1": ROOT / "models" / "phone_classifier.pth",
    "Candidate": ROOT / "models" / "phone_classifier_colab_candidate.pth",
}


def calculate_metrics(tp: int, fp: int, tn: int, fn: int):
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return accuracy, precision, recall, f1


def get_images(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def evaluate_one_model(name: str, model_path: Path, dataset_dir: Path, threshold: float):
    print("\n" + "=" * 72)
    print(f"MODEL: {name}")
    print(f"CHECKPOINT: {model_path}")
    print("=" * 72)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint for {name}: {model_path}\n"
            "Place the checkpoint in the project's models folder before running."
        )

    classifier = PhoneClassifier(model_path=str(model_path), threshold=threshold)
    if not classifier.load_model():
        raise RuntimeError(f"Could not load checkpoint: {model_path}")

    counts = {"PHONE": 0, "NOT_PHONE": 0}
    tp = fp = tn = fn = skipped = 0
    durations_ms = []

    for actual_label in ("PHONE", "NOT_PHONE"):
        image_paths = get_images(dataset_dir / actual_label)
        counts[actual_label] = len(image_paths)

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                skipped += 1
                print(f"[SKIP] Cannot read: {image_path.name}")
                continue

            start = time.perf_counter()
            prediction = classifier.predict(image)
            durations_ms.append((time.perf_counter() - start) * 1000.0)

            if prediction is None:
                skipped += 1
                print(f"[SKIP] Prediction failed: {image_path.name}")
                continue

            # Match PhoneDetector's production decision rule.
            predicted_phone = bool(prediction.get("is_phone", False))
            actual_phone = actual_label == "PHONE"

            if actual_phone and predicted_phone:
                tp += 1
            elif not actual_phone and predicted_phone:
                fp += 1
            elif not actual_phone and not predicted_phone:
                tn += 1
            else:
                fn += 1

    accuracy, precision, recall, f1 = calculate_metrics(tp, fp, tn, fn)
    average_ms = sum(durations_ms) / len(durations_ms) if durations_ms else 0.0

    print("\nDATASET:")
    print(f"PHONE images     : {counts['PHONE']}")
    print(f"NOT_PHONE images : {counts['NOT_PHONE']}")
    print(f"Threshold        : {threshold:.2f}")
    print(f"Skipped          : {skipped}")

    print("\nCONFUSION MATRIX:")
    print(f"TP (phone detected correctly) : {tp}")
    print(f"FP (false phone alert)        : {fp}")
    print(f"TN (non-phone rejected)       : {tn}")
    print(f"FN (phone missed)             : {fn}")

    print("\nMETRICS:")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1 * 100:.2f}%")
    print(f"Avg CNN inference time : {average_ms:.2f} ms/image")

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "skipped": skipped,
        "average_ms": average_ms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare CNN V1 and Colab Candidate on new real-world crops."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Folder containing PHONE and NOT_PHONE subfolders.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Phone probability threshold; default 0.50.",
    )
    parser.add_argument(
        "--v1",
        default=str(DEFAULT_MODELS["V1"]),
        help="Path to V1 checkpoint.",
    )
    parser.add_argument(
        "--candidate",
        default=str(DEFAULT_MODELS["Candidate"]),
        help="Path to Colab Candidate checkpoint.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")

    phone_images = get_images(dataset_dir / "PHONE")
    not_phone_images = get_images(dataset_dir / "NOT_PHONE")
    if not phone_images or not not_phone_images:
        raise ValueError("Both PHONE and NOT_PHONE folders must contain images.")

    print("PHONE CLASSIFIER REAL-WORLD EVALUATION")
    print(f"Dataset  : {dataset_dir.resolve()}")
    print(f"PHONE    : {len(phone_images)}")
    print(f"NOT_PHONE: {len(not_phone_images)}")
    print(f"Threshold: {args.threshold:.2f}")
    print("Note: this evaluates CNN classification on YOLO-proposed crops.")

    results = []
    results.append(
        evaluate_one_model("V1", Path(args.v1), dataset_dir, args.threshold)
    )
    results.append(
        evaluate_one_model("Candidate", Path(args.candidate), dataset_dir, args.threshold)
    )

    print("\n" + "=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)
    print(f"{'Model':<12}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Avg ms':>12}")
    for item in results:
        print(
            f"{item['name']:<12}"
            f"{item['accuracy'] * 100:>11.2f}%"
            f"{item['precision'] * 100:>11.2f}%"
            f"{item['recall'] * 100:>11.2f}%"
            f"{item['f1'] * 100:>11.2f}%"
            f"{item['average_ms']:>11.2f}"
        )
    print("\nEvaluation completed. No checkpoint was modified.")


if __name__ == "__main__":
    main()
