
from pathlib import Path
import argparse
import cv2

from modules.phone_classifier import PhoneClassifier


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TEST_DIR = ROOT / "data" / "phone_classifier" / "holdout_test"

MODELS = {
    "V1 - Model cu": ROOT / "models" / "phone_classifier.pth",
    "V2 - Model moi": ROOT / "models" / "phone_classifier_v2.pth"
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def calculate_metrics(tp, fp, tn, fn):

    total = tp + fp + tn + fn

    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1


def evaluate_model(model_name, model_path, test_dir, threshold):

    print("\n" + "=" * 65)
    print("EVALUATING:", model_name)
    print("=" * 65)

    if not model_path.is_file():
        print("Không tìm thấy model:", model_path)
        return

    classifier = PhoneClassifier(
        model_path=str(model_path),
        threshold=threshold
    )

    if not classifier.load_model():
        print("Không thể tải model.")
        return

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    skipped = 0

    for label in ["PHONE", "NOT_PHONE"]:

        folder = test_dir / label

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục: {folder}"
            )

        image_paths = sorted(
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:

            image = cv2.imread(str(image_path))

            if image is None:
                skipped += 1
                continue

            prediction = classifier.predict(image)

            if prediction is None:
                skipped += 1
                continue

            actual_phone = (label == "PHONE")

            predicted_phone = (
                prediction["phone_probability"] >= threshold
            )

            if actual_phone and predicted_phone:
                tp += 1

            elif not actual_phone and predicted_phone:
                fp += 1

            elif not actual_phone and not predicted_phone:
                tn += 1

            else:
                fn += 1

    accuracy, precision, recall, f1 = calculate_metrics(
        tp, fp, tn, fn
    )

    print("\nTHRESHOLD:", threshold)
    print("TP:", tp)
    print("FP:", fp)
    print("TN:", tn)
    print("FN:", fn)
    print("Skipped:", skipped)

    print("\nRESULT:")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1 * 100:.2f}%")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-dir",
        default=str(DEFAULT_TEST_DIR)
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60
    )

    args = parser.parse_args()

    test_dir = Path(args.test_dir)

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy test directory: {test_dir}"
        )

    print("=" * 65)
    print("       PHONE CLASSIFIER V1 vs V2")
    print("=" * 65)

    print("Test directory:", test_dir)
    print("Threshold:", args.threshold)

    for model_name, model_path in MODELS.items():

        evaluate_model(
            model_name,
            model_path,
            test_dir,
            args.threshold
        )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
