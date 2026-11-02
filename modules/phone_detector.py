from pathlib import Path
from typing import List, Optional, Tuple

from modules.phone_classifier import PhoneClassifier


class PhoneDetector:
    """Detect phone candidates with YOLOv8 and verify them with MobileNetV3.

    YOLO class 67 (COCO: cell phone) proposes candidate bounding boxes.
    Each candidate crop must be verified by a trained PhoneClassifier. If the
    classifier is unavailable, candidates are rejected rather than treated as
    confirmed phones.

    Pipeline:
        1. YOLO đề xuất bounding box với class 67 (cell phone).
        2. Bộ lọc hình học (aspect ratio + diện tích tối thiểu) loại bỏ bình
           nước, chuột, bóp ví ngay lập tức — không cần CNN.
        3. CNN xác minh từng ứng viên còn lại.

    Điện thoại có tỉ lệ w/h đặc trưng:
        - Portrait  : w/h ≈ 0.40 – 0.65
        - Landscape : w/h ≈ 1.55 – 2.50
    Vật bị nhầm thường ngoài vùng này:
        - Bình nước : w/h ≈ 0.15 – 0.35  (bị lọc)
        - Chuột     : w/h ≈ 0.65 – 1.10  (bị lọc)
        - Bóp ví    : w/h ≈ 1.10 – 1.55  (bị lọc)

    Detection tuple format:
        (x1, y1, x2, y2, confidence)

    Candidate tuples contain YOLO confidence; confirmed tuples contain CNN
    phone probability.
    """

    PHONE_CLASS_ID = 67

    # Khoảng tỉ lệ w/h hợp lệ cho điện thoại.
    # Portrait: 0.40–0.65 | Landscape: 1.55–2.50
    # Dải (0.40, 2.50) bao phủ cả hai chiều giữ nguyên.
    _ASPECT_MIN = 0.40
    _ASPECT_MAX = 2.50

    # Dải bị cấm giữa portrait và landscape (0.65–1.55):
    # điện thoại gần vuông không tồn tại — đây thường là chuột hoặc bóp.
    _ASPECT_FORBIDDEN_LOW  = 0.65
    _ASPECT_FORBIDDEN_HIGH = 1.55

    # Diện tích bounding box tối thiểu (pixel²).
    # Loại bỏ những box quá nhỏ không thể là điện thoại thật.
    _MIN_BOX_AREA = 1500

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.25,
        model=None,
        classifier_path: Optional[str] = None,
        classifier_threshold: float = 0.65,  # tăng từ 0.50 → 0.65 để giảm FP
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
        """Thử tải CNN để xác minh ứng viên YOLO."""

        if not self.use_classifier_if_available:
            return False

        try:
            return bool(
                self.classifier.load_model()
            )

        except Exception as error:
            print(f"[WARNING] Không tải được Phone CNN: {error}")

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

    def _passes_geometry_filter(
        self,
        candidate: Tuple[int, int, int, int, float]
    ) -> bool:
        """Lọc ứng viên bằng hình học bounding box TRƯỚC khi đưa vào CNN.

        Điện thoại có tỉ lệ w/h (aspect ratio) rất đặc trưng:
            - Portrait  : 0.40 – 0.65  (cao hơn rộng)
            - Landscape : 1.55 – 2.50  (rộng hơn cao)
        Vùng 0.65 – 1.55 (gần vuông) KHÔNG phải điện thoại:
            - Chuột máy tính : ~0.75 – 1.10
            - Bóp/ví         : ~1.10 – 1.50
        Bình nước có w/h < 0.40 (rất cao so với rộng).

        Trả về True nếu ứng viên có thể là điện thoại, False nếu loại.
        """
        x1, y1, x2, y2, _ = candidate

        box_w = int(x2) - int(x1)
        box_h = int(y2) - int(y1)

        # Loại box quá nhỏ
        area = box_w * box_h
        if area < self._MIN_BOX_AREA:
            return False

        if box_h == 0:
            return False

        aspect = box_w / box_h

        # Loại nếu ngoài dải hợp lệ tổng thể
        if aspect < self._ASPECT_MIN or aspect > self._ASPECT_MAX:
            return False

        # Loại vùng "gần vuông" — điện thoại không có tỉ lệ này
        if self._ASPECT_FORBIDDEN_LOW <= aspect <= self._ASPECT_FORBIDDEN_HIGH:
            return False

        return True

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
        """Only return YOLO candidates confirmed by the trained phone CNN."""

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

        # Bước 2: Lọc hình học — loại bình nước/chuột/bóp không cần CNN.
        geo_passed = []
        for candidate in candidates:
            if self._passes_geometry_filter(candidate):
                geo_passed.append(candidate)
            else:
                x1, y1, x2, y2, conf = candidate
                w = x2 - x1
                h = y2 - y1
                ratio = w / h if h > 0 else 0
                print(
                    f"[FILTER] Loai ung vien YOLO (hinh hoc): "
                    f"w={w} h={h} ratio={ratio:.2f} area={w*h} "
                    f"yolo_conf={conf:.2f}"
                )

        if not geo_passed:
            return []

        # A YOLO proposal alone is not reliable enough to report a phone.
        if not self.load_classifier():

            if not self._fallback_warning_shown:

                print(
                    "[WARNING] Phone CNN unavailable; ignoring YOLO "
                    "candidates to avoid false positives. Train "
                    "models/phone_classifier.pth to enable verification."
                )

                self._fallback_warning_shown = True

            return []

        # Bước 3: CNN xác minh từng ứng viên còn lại (đã qua lọc hình học).
        confirmed = []

        for candidate in geo_passed:

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