# -*- coding: utf-8 -*-
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

Cải tiến v3 (giảm false positive chuột/chai nước/bóp):
- Augmentation mạnh hơn: RandomPerspective, GaussianBlur, RandomGrayscale,
  RandomErasing để buộc model học đặc trưng cấu trúc thay vì màu sắc.
- Label Smoothing để tránh model quá tự tin với NOT_PHONE.
- Mixup augmentation trộn lẫn ảnh PHONE và NOT_PHONE, buộc ranh giới quyết định
  sắc nét hơn.
- Precision-oriented checkpoint: ưu tiên model có FP thấp (ít báo nhầm nhất).
- Tùy chọn --high-precision-mode tăng ngưỡng lựa chọn checkpoint lên 0.70.
- Tùy chọn --freeze-backbone để fine-tune chỉ classifier head (nhanh hơn).
- Tùy chọn --hard-negative-weight để phạt nặng false positive hơn.
"""

import argparse
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        description="Train Phone Classifier V2 (cải tiến giảm false positive)"
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
        default=30,
        help="Số epoch (mặc định tăng lên 30 để hội tụ tốt hơn)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=0,
        help="Dùng tối đa n ảnh mỗi lớp; 0 = dùng tất cả (mặc định đổi sang 0)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
        help="Learning rate ban đầu (mặc định tăng nhẹ để warm-up nhanh hơn)"
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

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Đóng băng backbone MobileNetV3, chỉ train classifier head. "
            "Nhanh hơn và ít overfitting khi dữ liệu ít (<200 ảnh/lớp)."
        )
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.10,
        help="Label smoothing (0.0 = tắt; 0.10 giúp tránh overconfidence)"
    )

    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.20,
        help="Mixup alpha (0.0 = tắt mixup; 0.20 trộn nhẹ để tạo ranh giới rõ ràng hơn)"
    )

    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=2.0,
        help=(
            "Nhân trọng số lớp NOT_PHONE thêm n lần so với cân bằng tự nhiên. "
            "Giá trị > 1 phạt nặng false positive (báo nhầm vật khác là điện thoại). "
            "Ví dụ: 2.0 = NOT_PHONE được coi là quan trọng gấp đôi PHONE."
        )
    )

    parser.add_argument(
        "--high-precision-mode",
        action="store_true",
        help=(
            "Chọn checkpoint tối ưu Precision thay vì F1. "
            "Dùng khi cần tuyệt đối không báo nhầm (chuột/chai nước/bóp), "
            "chấp nhận bỏ sót điện thoại hơn."
        )
    )

    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=SELECTION_THRESHOLD,
        help=(
            "Ngưỡng xác suất để chọn checkpoint tốt nhất. "
            "Nâng lên 0.65-0.70 để giảm false positive trong validation."
        )
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
# DATA TRANSFORMS (cải tiến)
# ==========================================================

def create_transforms(use_strong_augment: bool = True):
    """
    Augmentation mạnh hơn để giảm false positive:
    - RandomPerspective: giả lập góc nhìn khác nhau (cầm điện thoại nghiêng).
    - GaussianBlur: giả lập camera mờ/nhoè.
    - RandomGrayscale: buộc model học hình dạng, không chỉ màu sắc.
    - RandomErasing: che ngẫu nhiên một phần ảnh, tránh overfit màu nền.
    """

    if use_strong_augment:
        train_transform = transforms.Compose([

            transforms.RandomResizedCrop(
                INPUT_SIZE,
                scale=(0.70, 1.0),   # crop rộng hơn để học nhiều tỷ lệ
                ratio=(0.75, 1.33)   # tỷ lệ khung hình đa dạng hơn
            ),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomVerticalFlip(p=0.10),  # ảnh điện thoại ít lộn ngược

            transforms.RandomRotation(degrees=15),   # xoay nhẹ

            transforms.RandomPerspective(
                distortion_scale=0.20,
                p=0.40
            ),

            transforms.ColorJitter(
                brightness=0.30,
                contrast=0.30,
                saturation=0.20,
                hue=0.05
            ),

            transforms.RandomGrayscale(p=0.10),  # học hình dạng, không chỉ màu

            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.5)
            ),

            transforms.ToTensor(),

            transforms.Normalize(mean=MEAN, std=STD),

            # Che ngẫu nhiên một vùng nhỏ, tránh overfit chi tiết cục bộ
            transforms.RandomErasing(
                p=0.20,
                scale=(0.02, 0.10),
                ratio=(0.5, 2.0),
                value=0
            ),
        ])

    else:
        # Augmentation nhẹ (tương thích với phiên bản cũ)
        train_transform = transforms.Compose([

            transforms.RandomResizedCrop(
                INPUT_SIZE,
                scale=(0.85, 1.0),
                ratio=(0.90, 1.10)
            ),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10
            ),

            transforms.ToTensor(),

            transforms.Normalize(mean=MEAN, std=STD),
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
# MIXUP AUGMENTATION
# ==========================================================

def mixup_batch(images, labels, alpha: float, num_classes: int, device):
    """
    Mixup: trộn cặp ảnh và nhãn ngẫu nhiên.
    Giúp model học ranh giới quyết định sắc nét hơn giữa PHONE và NOT_PHONE.
    Trả về (mixed_images, soft_labels) với soft_labels là one-hot mềm.
    """
    if alpha <= 0.0:
        # Không dùng mixup; trả về one-hot cứng
        soft_labels = F.one_hot(labels, num_classes).float()
        return images, soft_labels

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=device)

    mixed_images = lam * images + (1 - lam) * images[perm]

    # Tạo soft label
    labels_one_hot = F.one_hot(labels, num_classes).float()
    labels_perm_one_hot = F.one_hot(labels[perm], num_classes).float()
    soft_labels = lam * labels_one_hot + (1 - lam) * labels_perm_one_hot

    return mixed_images, soft_labels


def soft_cross_entropy(logits, soft_labels, label_smoothing: float = 0.0):
    """
    Cross entropy với soft label (dùng cho mixup + label smoothing).
    """
    num_classes = logits.size(1)

    if label_smoothing > 0.0:
        # Áp dụng label smoothing lên soft label
        smoothed = soft_labels * (1 - label_smoothing) + label_smoothing / num_classes
    else:
        smoothed = soft_labels

    log_probs = F.log_softmax(logits, dim=1)
    loss = -(smoothed * log_probs).sum(dim=1).mean()
    return loss


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

    train_transform, val_transform = create_transforms(use_strong_augment=True)

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

def build_model(num_classes, pretrained=True, freeze_backbone=False):
    """
    Xây dựng MobileNetV3-Small với tùy chọn đóng băng backbone.

    freeze_backbone=True: chỉ train classifier head.
    Phù hợp khi dữ liệu ít (<200 ảnh/lớp) để tránh overfitting.
    """

    if pretrained:

        weights = models.MobileNet_V3_Small_Weights.DEFAULT

        model = models.mobilenet_v3_small(
            weights=weights
        )

    else:

        model = models.mobilenet_v3_small(
            weights=None
        )

    if freeze_backbone and pretrained:
        # Đóng băng toàn bộ features, chỉ train avgpool + classifier
        for param in model.features.parameters():
            param.requires_grad = False
        print("[INFO] Backbone đã bị đóng băng. Chỉ train classifier head.")

    input_features = model.classifier[3].in_features

    # Thêm Dropout cao hơn vào classifier để tránh overfitting
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(input_features, num_classes)
    )

    return model


# ==========================================================
# TRAIN ONE EPOCH (hỗ trợ mixup)
# ==========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    num_classes: int,
    mixup_alpha: float = 0.0,
    label_smoothing: float = 0.0
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Áp dụng mixup
        mixed_images, soft_labels = mixup_batch(
            images, labels, mixup_alpha, num_classes, device
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(mixed_images)

        loss = soft_cross_entropy(logits, soft_labels, label_smoothing)

        loss.backward()

        # Gradient clipping để training ổn định hơn
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Tính accuracy trên ảnh gốc (không mixup) cho dễ theo dõi
        with torch.inference_mode():
            orig_logits = model(images)
            predictions = orig_logits.argmax(dim=1)

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
    device,
    phone_index
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    true_labels = []
    phone_probabilities = []

    criterion_eval = nn.CrossEntropyLoss()

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion_eval(outputs, labels)

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

    fp_warn = " ⚠ FP CAO" if metrics["fp"] > 5 else ""

    print(
        f"Threshold={threshold:.2f} | "
        f"Acc={metrics['accuracy'] * 100:.1f}% | "
        f"Precision={metrics['precision'] * 100:.1f}% | "
        f"Recall={metrics['recall'] * 100:.1f}% | "
        f"F1={metrics['f1'] * 100:.1f}% | "
        f"TP={metrics['tp']} "
        f"FP={metrics['fp']}{fp_warn} "
        f"TN={metrics['tn']} "
        f"FN={metrics['fn']}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():
    import sys
    import io
    # Đảm bảo stdout có thể in UTF-8 trên Windows (PowerShell CP1252)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    else:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    args = parse_args()

    if args.epochs < 1:
        raise ValueError("Epochs phải lớn hơn 0.")

    if args.batch_size < 1:
        raise ValueError("Batch size phải lớn hơn 0.")

    if args.max_images_per_class < 0:
        raise ValueError("Max images per class cannot be negative.")

    if args.learning_rate <= 0:
        raise ValueError("Learning rate phải lớn hơn 0.")

    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("Label smoothing phải trong khoảng [0, 1).")

    if args.mixup_alpha < 0.0:
        raise ValueError("Mixup alpha không được âm.")

    if args.hard_negative_weight < 1.0:
        raise ValueError("hard-negative-weight phải >= 1.0.")

    selection_threshold = args.selection_threshold

    set_seed(args.seed)

    print("=" * 70)
    print("    STUDENT FOCUS - PHONE CLASSIFIER V2 (cải tiến giảm FP)")
    print("=" * 70)

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

    not_phone_index = class_names.index(
        NOT_PHONE_LABEL
    )

    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    # ------------------------------------------------------
    # Cấu hình training
    # ------------------------------------------------------

    print("\nTRAINING CONFIGURATION")
    print(f"  Epochs             : {args.epochs}")
    print(f"  Learning rate      : {args.learning_rate}")
    print(f"  Label smoothing    : {args.label_smoothing}")
    print(f"  Mixup alpha        : {args.mixup_alpha}")
    print(f"  Hard negative wt   : {args.hard_negative_weight}x")
    print(f"  Freeze backbone    : {args.freeze_backbone}")
    print(f"  High precision mode: {args.high_precision_mode}")
    print(f"  Selection threshold: {selection_threshold:.2f}")

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
            pretrained=not args.no_pretrained,
            freeze_backbone=args.freeze_backbone
        )

    except Exception as error:

        raise RuntimeError(
            "Không thể tải MobileNetV3 pretrained. "
            "Hãy kiểm tra Internet hoặc thử --no-pretrained."
        ) from error

    model.to(device)

    # ------------------------------------------------------
    # Weighted loss - phạt nặng NOT_PHONE hơn để giảm FP
    # ------------------------------------------------------

    counts = [
        train_counts[name]
        for name in class_names
    ]

    class_counts = torch.tensor(
        counts,
        dtype=torch.float32
    )

    # Cân bằng tự nhiên
    class_weights = (
        class_counts.sum()
        / (
            len(class_names)
            * class_counts.clamp(min=1)
        )
    )

    # Phạt nặng thêm NOT_PHONE: tăng chi phí khi model báo nhầm vật khác là điện thoại
    if args.hard_negative_weight > 1.0:
        class_weights[not_phone_index] *= args.hard_negative_weight
        print(
            f"\n[INFO] NOT_PHONE weight nhân thêm {args.hard_negative_weight}x "
            f"-> FP (báo nhầm vật khác) sẽ bị phạt nặng hơn."
        )

    # Criterion cho các batch không dùng mixup (val/eval)
    criterion_train_plain = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing
    )
    _ = criterion_train_plain  # chỉ dùng để khởi tạo; mixup dùng soft_cross_entropy

    # Optimizer: tách learning rate cho backbone và head khi không freeze
    if args.freeze_backbone:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=0.01
        )
    else:
        # Backbone học chậm hơn 10x để không phá vỡ pretrained features
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.learning_rate * 0.1},
                {"params": head_params,    "lr": args.learning_rate}
            ],
            weight_decay=0.01
        )
        print(
            f"[INFO] Differential LR: backbone={args.learning_rate * 0.1:.6f}, "
            f"head={args.learning_rate:.6f}"
        )

    # Cosine Annealing với warm restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,           # restart mỗi 10 epoch
        T_mult=1,
        eta_min=1e-6
    )

    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_score = -1.0      # F1 hoặc Precision tùy mode
    best_f1 = -1.0
    best_precision = -1.0
    best_fp = float("inf")
    best_checkpoint = None

    # ------------------------------------------------------
    # Training loop
    # ------------------------------------------------------

    print("\nSTART TRAINING")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            num_classes=len(class_names),
            mixup_alpha=args.mixup_alpha,
            label_smoothing=args.label_smoothing
        )

        val_loss, true_labels, phone_probs = validate(
            model,
            val_loader,
            device,
            phone_index
        )

        scheduler.step()

        metrics = calculate_metrics(
            true_labels,
            phone_probs,
            phone_index,
            selection_threshold
        )

        fp_indicator = f" [FP={metrics['fp']}]" if metrics['fp'] > 0 else " [FP=0 ✓]"

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"TrainLoss={train_loss:.4f} | "
            f"TrainAcc={train_accuracy * 100:.1f}% | "
            f"ValLoss={val_loss:.4f} | "
            f"F1={metrics['f1'] * 100:.1f}% | "
            f"Prec={metrics['precision'] * 100:.1f}%{fp_indicator}"
        )

        # --------------------------------------------------
        # Lựa chọn checkpoint tốt nhất
        # --------------------------------------------------

        if args.high_precision_mode:
            # Ưu tiên Precision cao nhất (ít FP nhất)
            # Nếu Precision bằng nhau, chọn F1 cao hơn
            current_score = metrics["precision"]
            is_better = (
                current_score > best_score
                or (
                    abs(current_score - best_score) < 0.000001
                    and metrics["f1"] > best_f1
                )
            )
        else:
            # Ưu tiên F1 (cân bằng Precision-Recall)
            # Nếu F1 bằng nhau, chọn Precision cao hơn (ít FP hơn)
            current_score = metrics["f1"]
            is_better = (
                current_score > best_score
                or (
                    abs(current_score - best_score) < 0.000001
                    and metrics["precision"] > best_precision
                )
            )

        if is_better:

            best_score = current_score
            best_f1 = metrics["f1"]
            best_precision = metrics["precision"]
            best_fp = metrics["fp"]

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

                "validation_fp": metrics["fp"],

                "validation_threshold": selection_threshold,

                "class_counts": train_counts,

                "validation_class_counts": val_counts,

                "epoch": epoch,

                # Lưu cấu hình training để tái hiện
                "training_config": {
                    "label_smoothing": args.label_smoothing,
                    "mixup_alpha": args.mixup_alpha,
                    "hard_negative_weight": args.hard_negative_weight,
                    "freeze_backbone": args.freeze_backbone,
                    "high_precision_mode": args.high_precision_mode,
                }
            }

            torch.save(
                best_checkpoint,
                output_path
            )

            mode_label = "Precision" if args.high_precision_mode else "F1"
            print(
                f"  [SAVED] Best checkpoint (epoch {epoch}): "
                f"{mode_label}={current_score * 100:.2f}% | "
                f"FP={metrics['fp']} -> {output_path}"
            )

    # ------------------------------------------------------
    # Final validation report
    # ------------------------------------------------------

    if best_checkpoint is None:
        raise RuntimeError(
            "Training không tạo được checkpoint."
        )

    print("\n" + "=" * 70)
    print("                  VALIDATION REPORT")
    print("=" * 70)

    print(
        "Best epoch:",
        best_checkpoint["epoch"]
    )

    print(
        "Best val F1 PHONE:",
        f"{best_checkpoint['validation_f1_phone'] * 100:.2f}%"
    )

    print(
        "Best val Precision PHONE:",
        f"{best_checkpoint['validation_precision_phone'] * 100:.2f}%"
    )

    print(
        "Best val FP (false positive):",
        best_checkpoint["validation_fp"]
    )

    print("\nTHRESHOLD COMPARISON (best checkpoint)")
    print("-" * 70)

    # Re-evaluate using the best saved model.

    model.load_state_dict(
        best_checkpoint["state_dict"]
    )

    _, true_labels, phone_probs = validate(
        model,
        val_loader,
        device,
        phone_index
    )

    best_precision_threshold = 0.50
    best_precision_val = 0.0

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

        # Tìm threshold cho Precision tốt nhất với Recall >= 0.5
        if (
            metrics["precision"] > best_precision_val
            and metrics["recall"] >= 0.50
        ):
            best_precision_val = metrics["precision"]
            best_precision_threshold = threshold

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print("Checkpoint:", output_path.resolve())

    print(
        "Best val F1 PHONE:",
        f"{best_f1 * 100:.2f}%"
    )

    print(
        "Best val Precision PHONE:",
        f"{best_precision * 100:.2f}%"
    )

    print(
        "Best val FP (báo nhầm trên validation):",
        best_fp
    )

    print(
        "Selection threshold:",
        selection_threshold
    )

    print()
    print(
        f"💡 GỢI Ý: Nếu vẫn còn báo nhầm chuột/chai nước/bóp trong thực tế:"
    )
    print(
        f"   1. Thêm nhiều ảnh chuột/chai nước/bóp vào NOT_PHONE, train lại."
    )
    print(
        f"   2. Tăng threshold trong app.py hoặc phone_detector.py lên "
        f"{best_precision_threshold:.2f} (threshold tốt nhất trên validation)."
    )
    print(
        f"   3. Dùng --high-precision-mode để chọn checkpoint ưu tiên Precision."
    )
    print(
        f"   4. Tăng --hard-negative-weight (hiện tại {args.hard_negative_weight}) "
        f"để phạt FP nặng hơn."
    )

    print(
        "\nLưu ý: holdout_test chưa được sử dụng. "
        "Hãy giữ nguyên tập này để đánh giá cuối cùng."
    )


if __name__ == "__main__":
    main()
