from pathlib import Path
from typing import List, Optional, Tuple

from modules.phone_classifier import PhoneClassifier


class PhoneDetector:
    """Detect phone candidates with YOLOv8 and verify them with MobileNetV3.

    YOLO class 67 (COCO: cell phone) proposes candidate bounding boxes.
    If a trained PhoneClassifier checkpoint is available, each candidate crop
    is verified by the CNN. If the checkpoint is not available yet, this class
    falls back to YOLO detections so the existing phone-detection feature keeps
    working while the training dataset is being collected.

    Detection tuple format:
        (x1, y1, x2, y2, confidence)

    confidence is YOLO confidence during fallback, and CNN phone probability
    after successful CNN verification.
    """

    PHONE_CLASS_ID = 67

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.25,
        model=None,
        classifier_path: Optional[str] = None,
        classifier_threshold: float = 0.80,
        classifier=None,
        use_classifier_if_available: bool = True,
    ) -> None:

        self.confidence = float(confidence)
        self.model = model
        self.use_classifier_if_available = bool(
            use_classifier_if_available
        )

        project_root = Path(__file__).resolve().parent.parent

        self.model_path = (
            Path(model_path)
            if model_path is not None
            else project_root / "yolov8n.pt"
        )

        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = PhoneClassifier(
                model_path=classifier_path,
                threshold=classifier_threshold,
            )

        self._fallback_warning_shown = False

    @property
    def classifier_ready(self) -> bool:
        """Kiểm tra CNN đã tải trọng số thành công hay chưa."""

        return bool(
            getattr(self.classifier, "ready", False)
        )

    def load_model(self) -> None:
        """Load YOLO nếu chưa được truyền model dùng chung."""

        if self.model is not None:
            return

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy YOLO model: {self.model_path}"
            )

        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))

        print("[INFO] YOLO Phone Detector Loaded")

    def load_classifier(self) -> bool:
        """Thử tải CNN. Nếu chưa có trọng số thì dùng YOLO."""

        if not self.use_classifier_if_available:
            return False

        try:
            return bool(
                self.classifier.load_model()
            )

        except Exception as error:
            print(
                "[WARNING] Không tải được Phone CNN; "
                "tạm dùng kết quả YOLO. "
                f"Chi tiết: {error}"
            )

            return False

    def parse_results(
        self,
        results
    ) -> List[Tuple[int, int, int, int, float]]:
        """Lấy các ứng viên cell phone từ kết quả YOLO."""

        candidates = []

        if results is None:
            return candidates

        if not isinstance(results, (list, tuple)):
            results = [results]

        for result in results:

            boxes = getattr(result, "boxes", None)

            if boxes is None:
                continue

            for box in boxes:

                try:
                    class_id = int(
                        box.cls[0].item()
                    )

                    if class_id != self.PHONE_CLASS_ID:
                        continue

                    yolo_confidence = float(
                        box.conf[0].item()
                    )

                    if yolo_confidence < self.confidence:
                        continue

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    candidates.append(
                        (
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                            yolo_confidence,
                        )
                    )

                except (
                    AttributeError,
                    IndexError,
                    TypeError,
                    ValueError
                ) as error:

                    print(
                        "[WARNING] Bỏ qua kết quả YOLO không hợp lệ: "
                        f"{error}"
                    )

        return candidates

    @staticmethod
    def _crop_candidate(
        frame,
        candidate,
        padding: float = 0.12
    ):
        """Cắt vùng vật thể YOLO và thêm một ít vùng xung quanh."""

        if frame is None or getattr(frame, "size", 0) == 0:
            return None

        height, width = frame.shape[:2]

        x1, y1, x2, y2, _ = candidate

        box_width = int(x2) - int(x1)
        box_height = int(y2) - int(y1)

        if box_width <= 1 or box_height <= 1:
            return None

        pad_x = int(box_width * padding)
        pad_y = int(box_height * padding)

        left = max(0, int(x1) - pad_x)
        top = max(0, int(y1) - pad_y)

        right = min(width, int(x2) + pad_x)
        bottom = min(height, int(y2) + pad_y)

        if right <= left or bottom <= top:
            return None

        crop = frame[top:bottom, left:right]

        if crop.size == 0:
            return None

        return crop.copy()

    def detect_phone(self, frame, results=None):
        """YOLO phát hiện và CNN xác minh điện thoại.

        Khi chưa có CNN:
            Trả về các ứng viên YOLO để chức năng không bị tắt.

        Khi đã có CNN:
            Chỉ trả về ứng viên được CNN xác nhận là PHONE.
        """

        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        # Nếu không có kết quả YOLO được truyền vào,
        # tự chạy YOLO độc lập.
        if results is None:

            self.load_model()

            results = self.model.predict(
                source=frame,
                classes=[self.PHONE_CLASS_ID],
                conf=self.confidence,
                imgsz=416,
                device="cpu",
                verbose=False,
            )

        # Bước 1: Lấy ứng viên điện thoại từ YOLO.
        candidates = self.parse_results(results)

        if not candidates:
            return []

        # Bước 2: Nếu CNN chưa sẵn sàng,
        # tạm thời sử dụng YOLO như trước.
        if not self.load_classifier():

            if not self._fallback_warning_shown:

                print(
                    "[INFO] Phone CNN chưa sẵn sàng; "
                    "đang tạm dùng YOLOv8 để nhận diện điện thoại."
                )

                print(
                    "[INFO] Hãy huấn luyện và đặt file "
                    "models/phone_classifier.pth "
                    "để bật xác minh CNN."
                )

                self._fallback_warning_shown = True

            return candidates

        # Bước 3: CNN xác minh từng ứng viên.
        confirmed = []

        for candidate in candidates:

            crop = self._crop_candidate(
                frame,
                candidate
            )

            if crop is None:
                continue

            prediction = self.classifier.predict(crop)

            if prediction is None:
                # CNN lỗi khi dự đoán thì không xác nhận
                # vật thể là điện thoại.
                continue

            if not prediction.get("is_phone", False):
                continue

            x1, y1, x2, y2, _ = candidate

            cnn_confidence = float(
                prediction.get("phone_probability", 0.0)
            )

            confirmed.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    cnn_confidence
                )
            )

        return confirmed