from pathlib import Path
from typing import Any, Dict, Optional


class PhoneClassifier:
    """MobileNetV3-Small classifier that verifies YOLO phone candidates.

    Expected checkpoint format is produced by training/train_phone_classifier.py.
    Class folders must be named PHONE and NOT_PHONE.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.50,
        device: Optional[str] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else project_root / "models" / "phone_classifier.pth"
        )
        self.threshold = float(threshold)
        self.requested_device = device

        self.model = None
        self.device = None
        self.class_names = []
        self.transform = None
        self._load_attempted = False
        self._load_error = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load_model(self) -> bool:
        """Load model weights once. Returns False if no trained checkpoint exists."""
        if self.ready:
            return True

        if self._load_attempted:
            return False
        self._load_attempted = True

        if not self.model_path.is_file():
            self._load_error = f"Checkpoint not found: {self.model_path}"
            print(
                "[WARNING] Phone detection is disabled until a trained CNN "
                f"checkpoint is available: {self.model_path}"
            )
            return False

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            if self.requested_device:
                self.device = torch.device(self.requested_device)
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

            try:
                checkpoint = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                # Compatibility with older PyTorch versions.
                checkpoint = torch.load(self.model_path, map_location=self.device)

            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                raise ValueError(
                    "Invalid checkpoint. Train the model with "
                    "training/train_phone_classifier.py."
                )

            self.class_names = checkpoint.get("class_names", ["NOT_PHONE", "PHONE"])
            self.class_names = [str(name).upper() for name in self.class_names]
            if "PHONE" not in self.class_names or "NOT_PHONE" not in self.class_names:
                raise ValueError(
                    "Checkpoint must contain both PHONE and NOT_PHONE classes."
                )

            model = models.mobilenet_v3_small(weights=None)
            input_features = model.classifier[3].in_features

            # Hỗ trợ cả hai dạng classifier head:
            # - V2 cũ: nn.Linear đơn giản
            # - V3 mới: nn.Sequential(Dropout, Linear) để giảm overfitting
            state_keys = checkpoint["state_dict"].keys()
            has_sequential_head = any(
                "classifier.3.1.weight" in k for k in state_keys
            )
            if has_sequential_head:
                model.classifier[3] = nn.Sequential(
                    nn.Dropout(p=0.30),
                    nn.Linear(input_features, len(self.class_names))
                )
            else:
                model.classifier[3] = nn.Linear(input_features, len(self.class_names))

            model.load_state_dict(checkpoint["state_dict"])
            model.to(self.device)
            model.eval()
            self.model = model

            input_size = int(checkpoint.get("input_size", 224))
            mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
            std = checkpoint.get("std", [0.229, 0.224, 0.225])
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            print(
                "[INFO] Phone CNN loaded: "
                f"{self.model_path} | device={self.device} | "
                f"threshold={self.threshold:.2f}"
            )
            return True

        except Exception as error:
            self._load_error = str(error)
            self.model = None
            print(f"[ERROR] Could not load Phone CNN: {error}")
            return False

    def predict(self, image_bgr: Any) -> Optional[Dict[str, Any]]:
        """Classify one BGR image crop and return label/probability details."""
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return None

        if not self.load_model():
            return None

        try:
            import cv2
            import torch
            from PIL import Image

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

            best_index = int(torch.argmax(probabilities).item())
            label = self.class_names[best_index]
            probability = float(probabilities[best_index].item())
            phone_index = self.class_names.index("PHONE")
            phone_probability = float(probabilities[phone_index].item())

            return {
                "label": label,
                "confidence": probability,
                "phone_probability": phone_probability,
                "is_phone": (
                    label == "PHONE" and phone_probability >= self.threshold
                ),
            }

        except Exception as error:
            print(f"[WARNING] Phone CNN prediction failed: {error}")
            return None
