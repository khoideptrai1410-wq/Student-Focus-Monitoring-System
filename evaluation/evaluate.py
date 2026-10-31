import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(__file__)

GROUND_TRUTH = os.path.join(BASE_DIR, "ground_truth.csv")
PREDICTION = os.path.join(BASE_DIR, "prediction.csv")

# ==========================================================
# Load CSV
# ==========================================================

gt = pd.read_csv(GROUND_TRUTH)
pred = pd.read_csv(PREDICTION)

# Merge theo Time
data = pd.merge(
    gt,
    pred,
    on="Time",
    how="inner"
)

y_true = data["Label"]
y_pred = data["Prediction"]

# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision_weighted = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

# ==========================================================
# Console
# ==========================================================

print("=" * 60)
print("      STUDENT FOCUS MONITORING SYSTEM")
print("              EVALUATION")
print("=" * 60)

print(f"Total Samples        : {len(y_true)}")
print()

print(f"Accuracy             : {accuracy * 100:.2f}%")

print()

print(f"Weighted Precision   : {precision_weighted * 100:.2f}%")
print(f"Weighted Recall      : {recall_weighted * 100:.2f}%")
print(f"Weighted F1-score    : {f1_weighted * 100:.2f}%")

print()

print(f"Macro Precision      : {precision_macro * 100:.2f}%")
print(f"Macro Recall         : {recall_macro * 100:.2f}%")
print(f"Macro F1-score       : {f1_macro * 100:.2f}%")

# ==========================================================
# Save Metrics
# ==========================================================

with open(
    os.path.join(BASE_DIR, "metrics.txt"),
    "w",
    encoding="utf8"
) as f:

    f.write("=" * 60 + "\n")
    f.write("STUDENT FOCUS MONITORING SYSTEM\n")
    f.write("EVALUATION RESULT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total Samples        : {len(y_true)}\n\n")

    f.write(f"Accuracy             : {accuracy * 100:.2f}%\n\n")

    f.write(f"Weighted Precision   : {precision_weighted * 100:.2f}%\n")
    f.write(f"Weighted Recall      : {recall_weighted * 100:.2f}%\n")
    f.write(f"Weighted F1-score    : {f1_weighted * 100:.2f}%\n\n")

    f.write(f"Macro Precision      : {precision_macro * 100:.2f}%\n")
    f.write(f"Macro Recall         : {recall_macro * 100:.2f}%\n")
    f.write(f"Macro F1-score       : {f1_macro * 100:.2f}%\n")

# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(
    y_true,
    y_pred,
    zero_division=0
)

print("\n")
print(report)

with open(
    os.path.join(BASE_DIR, "classification_report.txt"),
    "w",
    encoding="utf8"
) as f:

    f.write(report)

# ==========================================================
# Labels
# ==========================================================

labels = sorted(
    list(
        set(y_true) | set(y_pred)
    )
)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

fig, ax = plt.subplots(figsize=(9, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    cmap="Blues",
    values_format="d",
    xticks_rotation=35,
    colorbar=False,
    ax=ax
)

plt.title(
    "Confusion Matrix",
    fontsize=16,
    fontweight="bold"
)

plt.grid(False)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

# ==========================================================
# Normalized Confusion Matrix
# ==========================================================

cm_norm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels,
    normalize="true"
)

fig, ax = plt.subplots(figsize=(9, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_norm,
    display_labels=labels
)

disp.plot(
    cmap="Blues",
    values_format=".2f",
    xticks_rotation=35,
    colorbar=False,
    ax=ax
)

plt.title(
    "Normalized Confusion Matrix",
    fontsize=16,
    fontweight="bold"
)

plt.grid(False)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "confusion_matrix_normalized.png"
    ),
    dpi=300
)

plt.show()

print("\n")
print("=" * 60)
print("Evaluation completed successfully.")
print("=" * 60)
print("Generated files:")
print("- metrics.txt")
print("- classification_report.txt")
print("- confusion_matrix.png")
print("- confusion_matrix_normalized.png")
print("=" * 60)