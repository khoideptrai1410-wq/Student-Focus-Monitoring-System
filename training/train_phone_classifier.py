
"""
Train MobileNetV3-Small to classify PHONE vs NOT_PHONE.

Dataset structure:
    data/phone_classifier/
        raw/
            PHONE/
            NOT_PHONE/
        holdout_val/
            PHONE/
            NOT_PHONE/
        holdout_test/
            PHONE/
            NOT_PHONE/

Important:
- raw is used only for training.
- holdout_val is used only for validation.
- holdout_test is never loaded by this script.
- Output is saved as models/phone_classifier_v2.pth.
"""

import argparse
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEFAULT_TRAIN_DIR = "data/phone_classifier/raw"
DEFAULT_VAL_DIR = "data/phone_classifier/holdout_val"
DEFAULT_OUTPUT = "models/phone_classifier_v2.pth"

PHONE_LABEL = "PHONE"
NOT_PHONE_LABEL = "NOT_PHONE"

# Threshold dùng để lựa chọn checkpoint.
# Giữ 0.50 để tương thích với cấu hình app hiện tại.
SELECTION_THRESHOLD = 0.50

# Dùng để theo dõi ảnh hưởng của việc tăng threshold.
THRESHOLDS_TO_REPORT = [
    0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90
]


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train Phone Classifier V2"
    )

    parser.add_argument(
        "--train-dir",
        default=DEFAULT_TRAIN_DIR
    )

    parser.add_argument(
        "--val-dir",
        default=DEFAULT_VAL_DIR
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=100,
        help="Use at most this many randomly selected training images per class; 0 uses all",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train without ImageNet pretrained weights"
    )

    return parser.parse_args()


# ==========================================================
# RANDOM SEED
# ==========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================
# DATA TRANSFORMS
# ==========================================================

def create_transforms():

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            INPUT_SIZE,
            scale=(0.85, 1.0),
            ratio=(0.90, 1.10)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=MEAN,
            std=STD
        )
    ])

    val_transform = transforms.Compose([

        transforms.Resize(
            (INPUT_SIZE, INPUT_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=MEAN,
            std=STD
        )
    ])

    return train_transform, val_transform


# ==========================================================
# DATASET
# ==========================================================

def load_datasets(train_dir, val_dir):

    train_path = Path(train_dir)
    val_path = Path(val_dir)

    if not train_path.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục train: {train_path.resolve()}"
        )

    if not val_path.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục validation: {val_path.resolve()}"
        )

    train_transform, val_transform = create_transforms()

    train_dataset = datasets.ImageFolder(
        str(train_path),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        str(val_path),
        transform=val_transform
    )

    train_classes = [
        name.upper() for name in train_dataset.classes
    ]

    val_classes = [
        name.upper() for name in val_dataset.classes
    ]

    expected_classes = {
        PHONE_LABEL,
        NOT_PHONE_LABEL
    }

    if set(train_classes) != expected_classes:
        raise ValueError(
            "Thư mục train phải có đúng hai lớp PHONE và NOT_PHONE. "
            f"Đang có: {train_classes}"
        )

    if set(val_classes) != expected_classes:
        raise ValueError(
            "Thư mục validation phải có đúng hai lớp PHONE và NOT_PHONE. "
            f"Đang có: {val_classes}"
        )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Thứ tự nhãn giữa train và validation không giống nhau."
        )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            "Train hoặc validation không có ảnh."
        )

    class_names = train_classes

    train_counts = {
        name.upper(): train_dataset.targets.count(index)
        for name, index in train_dataset.class_to_idx.items()
    }

    val_counts = {
        name.upper(): val_dataset.targets.count(index)
        for name, index in val_dataset.class_to_idx.items()
    }

    for label in [PHONE_LABEL, NOT_PHONE_LABEL]:

        if train_counts.get(label, 0) == 0:
            raise ValueError(
                f"Train không có ảnh lớp {label}."
            )

        if val_counts.get(label, 0) == 0:
            raise ValueError(
                f"Validation không có ảnh lớp {label}."
            )

    return (
        train_dataset,
        val_dataset,
        class_names,
        train_counts,
        val_counts
    )


def limit_training_images(dataset, class_names, max_per_class, seed):
    """Select a reproducible, balanced subset without touching validation data."""
    available_counts = {
        class_name: sum(
            1 for target in dataset.targets if target == class_index
        )
        for class_index, class_name in enumerate(class_names)
    }

    if max_per_class == 0:
        return dataset, available_counts

    rng = random.Random(seed)
    selected_indices = []
    selected_counts = {}

    for class_index, class_name in enumerate(class_names):
        class_indices = [
            index
            for index, target in enumerate(dataset.targets)
            if target == class_index
        ]
        if len(class_indices) > max_per_class:
            class_indices = rng.sample(class_indices, max_per_class)

        selected_indices.extend(class_indices)
        selected_counts[class_name] = len(class_indices)

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices), selected_counts


# ==========================================================
# MODEL
# ==========================================================

def build_model(num_classes, pretrained=True):

    if pretrained:

        weights = models.MobileNet_V3_Small_Weights.DEFAULT

        model = models.mobilenet_v3_small(
            weights=weights
        )

    else:

        model = models.mobilenet_v3_small(
            weights=None
        )

    input_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(
        input_features,
        num_classes
    )

    return model


# ==========================================================
# TRAIN ONE EPOCH
# ==========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += int(
            (predictions == labels).sum().item()
        )

        total_samples += batch_size

    average_loss = total_loss / max(1, total_samples)

    accuracy = total_correct / max(1, total_samples)

    return average_loss, accuracy


# ==========================================================
# VALIDATION
# ==========================================================

def validate(
    model,
    loader,
    criterion,
    device,
    phone_index
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    true_labels = []
    phone_probabilities = []

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            phone_probs = probabilities[:, phone_index]

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            true_labels.extend(
                labels.cpu().tolist()
            )

            phone_probabilities.extend(
                phone_probs.cpu().tolist()
            )

    average_loss = total_loss / max(1, total_samples)

    return (
        average_loss,
        true_labels,
        phone_probabilities
    )


# ==========================================================
# CLASSIFICATION METRICS
# ==========================================================

def calculate_metrics(
    true_labels,
    phone_probabilities,
    phone_index,
    threshold
):

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for true_label, probability in zip(
        true_labels,
        phone_probabilities
    ):

        actual_phone = (
            true_label == phone_index
        )

        predicted_phone = (
            probability >= threshold
        )

        if actual_phone and predicted_phone:
            tp += 1

        elif not actual_phone and predicted_phone:
            fp += 1

        elif not actual_phone and not predicted_phone:
            tn += 1

        else:
            fn += 1

    precision = tp / max(1, tp + fp)

    recall = tp / max(1, tp + fn)

    f1 = (
        2 * precision * recall
        / max(0.000001, precision + recall)
    )

    accuracy = (
        (tp + tn)
        / max(1, tp + tn + fp + fn)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


# ==========================================================
# PRINT METRICS
# ==========================================================

def print_metrics(metrics, threshold):

    print(
        f"Threshold={threshold:.2f} | "
        f"Accuracy={metrics['accuracy'] * 100:.2f}% | "
        f"Precision={metrics['precision'] * 100:.2f}% | "
        f"Recall={metrics['recall'] * 100:.2f}% | "
        f"F1={metrics['f1'] * 100:.2f}% | "
        f"TP={metrics['tp']} "
        f"FP={metrics['fp']} "
        f"TN={metrics['tn']} "
        f"FN={metrics['fn']}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    args = parse_args()

    if args.epochs < 1:
        raise ValueError("Epochs phải lớn hơn 0.")

    if args.batch_size < 1:
        raise ValueError("Batch size phải lớn hơn 0.")

    if args.max_images_per_class < 0:
        raise ValueError("Max images per class cannot be negative.")

    if args.learning_rate <= 0:
        raise ValueError("Learning rate phải lớn hơn 0.")

    set_seed(args.seed)

    print("=" * 65)
    print("       STUDENT FOCUS - PHONE CLASSIFIER V2")
    print("=" * 65)

    # ------------------------------------------------------
    # Load train and validation only.
    # holdout_test is intentionally NOT loaded.
    # ------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        class_names,
        train_counts,
        val_counts
    ) = load_datasets(
        args.train_dir,
        args.val_dir
    )

    available_train_counts = train_counts
    train_dataset, train_counts = limit_training_images(
        train_dataset,
        class_names,
        args.max_images_per_class,
        args.seed
    )

    print("\nDATASET INFORMATION")

    print("Train directory:", args.train_dir)
    print("Validation directory:", args.val_dir)

    print(
        "Available train classes:",
        available_train_counts
    )

    print(
        "Used train samples:",
        len(train_dataset),
        f"(max {args.max_images_per_class or 'all'} per class)"
    )

    print("Used train classes:", train_counts)
    print("Validation samples:", len(val_dataset))

    print("Train classes:", train_counts)
    print("Validation classes:", val_counts)

    print("Classes:", class_names)

    phone_index = class_names.index(
        PHONE_LABEL
    )

    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    # ------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # ------------------------------------------------------
    # Build model
    # ------------------------------------------------------

    print("\nLoading MobileNetV3-Small...")

    try:

        model = build_model(
            num_classes=len(class_names),
            pretrained=not args.no_pretrained
        )

    except Exception as error:

        raise RuntimeError(
            "Không thể tải MobileNetV3 pretrained. "
            "Hãy kiểm tra Internet hoặc thử --no-pretrained."
        ) from error

    model.to(device)

    # ------------------------------------------------------
    # Weighted loss
    # ------------------------------------------------------

    counts = [
        train_counts[name]
        for name in class_names
    ]

    class_counts = torch.tensor(
        counts,
        dtype=torch.float32
    )

    class_weights = (
        class_counts.sum()
        / (
            len(class_names)
            * class_counts.clamp(min=1)
        )
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_f1 = -1.0
    best_precision = -1.0
    best_checkpoint = None

    # ------------------------------------------------------
    # Training loop
    # ------------------------------------------------------

    print("\nSTART TRAINING")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, true_labels, phone_probs = validate(
            model,
            val_loader,
            criterion,
            device,
            phone_index
        )

        scheduler.step()

        metrics = calculate_metrics(
            true_labels,
            phone_probs,
            phone_index,
            SELECTION_THRESHOLD
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss={train_loss:.4f} | "
            f"Train Acc={train_accuracy * 100:.2f}% | "
            f"Val Loss={val_loss:.4f} | "
            f"Val F1 PHONE={metrics['f1'] * 100:.2f}%"
        )

        # Select checkpoint by PHONE F1.
        # If F1 is equal, prefer higher PHONE precision.

        is_better = (
            metrics["f1"] > best_f1
            or (
                abs(metrics["f1"] - best_f1) < 0.000001
                and metrics["precision"] > best_precision
            )
        )

        if is_better:

            best_f1 = metrics["f1"]

            best_precision = metrics["precision"]

            best_checkpoint = {
                "state_dict": deepcopy(model.state_dict()),

                "class_names": class_names,

                "input_size": INPUT_SIZE,

                "mean": MEAN,

                "std": STD,

                "validation_accuracy": metrics["accuracy"],

                "validation_precision_phone": metrics["precision"],

                "validation_recall_phone": metrics["recall"],

                "validation_f1_phone": metrics["f1"],

                "validation_threshold": SELECTION_THRESHOLD,

                "class_counts": train_counts,

                "validation_class_counts": val_counts,

                "epoch": epoch
            }

            torch.save(
                best_checkpoint,
                output_path
            )

            print(
                f"  [SAVED] Best V2 checkpoint: {output_path}"
            )

    # ------------------------------------------------------
    # Final validation report
    # ------------------------------------------------------

    if best_checkpoint is None:
        raise RuntimeError(
            "Training không tạo được checkpoint."
        )

    print("\n" + "=" * 65)
    print("             V2 VALIDATION REPORT")
    print("=" * 65)

    print(
        "Best epoch:",
        best_checkpoint["epoch"]
    )

    print(
        "Best validation F1 PHONE:",
        f"{best_checkpoint['validation_f1_phone'] * 100:.2f}%"
    )

    print("\nTHRESHOLD COMPARISON")

    # Re-evaluate using the best saved model.

    model.load_state_dict(
        best_checkpoint["state_dict"]
    )

    _, true_labels, phone_probs = validate(
        model,
        val_loader,
        criterion,
        device,
        phone_index
    )

    for threshold in THRESHOLDS_TO_REPORT:

        metrics = calculate_metrics(
            true_labels,
            phone_probs,
            phone_index,
            threshold
        )

        print_metrics(
            metrics,
            threshold
        )

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

    print("Checkpoint:", output_path.resolve())

    print(
        "Best validation F1 PHONE:",
        f"{best_f1 * 100:.2f}%"
    )

    print(
        "Selection threshold:",
        SELECTION_THRESHOLD
    )

    print(
        "\nLưu ý: holdout_test chưa được sử dụng. "
        "Hãy giữ nguyên tập này để đánh giá cuối cùng."
    )


if __name__ == "__main__":
    main()
