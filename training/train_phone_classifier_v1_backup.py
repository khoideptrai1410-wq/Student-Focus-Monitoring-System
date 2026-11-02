"""Train a MobileNetV3-Small PHONE vs NOT_PHONE image classifier.

Expected input:
  data/phone_classifier/raw/PHONE/*
  data/phone_classifier/raw/NOT_PHONE/*

The script creates a stratified train/validation split from the raw images and
writes models/phone_classifier.pth.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Phone/Not-Phone CNN")
    parser.add_argument("--data-dir", default="data/phone_classifier/raw")
    parser.add_argument("--output", default="models/phone_classifier.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download/use ImageNet pretrained MobileNetV3 weights",
    )
    return parser.parse_args()


def make_stratified_split(targets, val_ratio, seed):
    rng = random.Random(seed)
    indices_by_class = defaultdict(list)
    for index, label in enumerate(targets):
        indices_by_class[int(label)].append(index)

    train_indices = []
    val_indices = []
    for label, indices in indices_by_class.items():
        rng.shuffle(indices)
        if len(indices) < 5:
            raise ValueError(
                "Each class needs at least 5 images to create a useful "
                "train/validation split. Collect more images first."
            )
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_model(num_classes, pretrained):
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)
            batch_count = labels.size(0)
            total_loss += float(loss.item()) * batch_count
            total_correct += int((predictions == labels).sum().item())
            total_count += batch_count

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    if not 0.05 <= args.val_ratio <= 0.40:
        raise ValueError("val-ratio must be between 0.05 and 0.40.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_dir)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_root.resolve()}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.85, 1.0), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_dataset_all = datasets.ImageFolder(str(data_root), transform=train_transform)
    val_dataset_all = datasets.ImageFolder(str(data_root), transform=val_transform)
    class_names = [name.upper() for name in train_dataset_all.classes]

    if set(class_names) != {"PHONE", "NOT_PHONE"}:
        raise ValueError(
            "Dataset must contain exactly these two class folders: "
            "PHONE and NOT_PHONE. Found: " + str(class_names)
        )

    counts = {
        class_names[index]: train_dataset_all.targets.count(index)
        for index in range(len(class_names))
    }
    print("Dataset counts:", counts)
    if min(counts.values()) < 5:
        raise ValueError("Collect at least 5 images per class before training.")

    train_indices, val_indices = make_stratified_split(
        train_dataset_all.targets, args.val_ratio, args.seed
    )
    train_loader = DataLoader(
        Subset(train_dataset_all, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(val_dataset_all, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_indices)} | Validation images: {len(val_indices)}")

    try:
        model = build_model(len(class_names), pretrained=not args.no_pretrained)
    except Exception as error:
        if not args.no_pretrained:
            raise RuntimeError(
                "Could not load pretrained MobileNetV3 weights. Check internet "
                "connection, or retry with --no-pretrained (less recommended)."
            ) from error
        raise

    model.to(device)

    # Balance the two classes if the collected image counts differ.
    train_targets = [train_dataset_all.targets[index] for index in train_indices]
    class_counts = torch.bincount(
        torch.tensor(train_targets, dtype=torch.long),
        minlength=len(class_names),
    ).float()
    class_weights = class_counts.sum() / (len(class_names) * class_counts.clamp(min=1))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    best_val_accuracy = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint = {
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "input_size": INPUT_SIZE,
                "mean": MEAN,
                "std": STD,
                "validation_accuracy": best_val_accuracy,
                "class_counts": counts,
            }
            torch.save(checkpoint, output_path)
            print(f"[SAVED] Best checkpoint: {output_path.resolve()}")

    print(f"Training finished. Best validation accuracy: {best_val_accuracy:.3f}")
    print("Checkpoint:", output_path.resolve())
    print(
        "Important: test with a separate camera session and new backgrounds/items. "
        "A high validation score on a random split does not guarantee performance "
        "on unseen objects or lighting conditions."
    )


if __name__ == "__main__":
    main()
