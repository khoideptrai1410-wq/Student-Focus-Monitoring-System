
from pathlib import Path


class PersonDetector:
    """
    Phat hien nguoi bang YOLOv8.
    Ho tro cat Person ROI de chuan bi anh cho CNN.
    """

    PERSON_CLASS_ID = 0

    def __init__(self, model_path=None, confidence=0.25, model=None):
        self.confidence = float(confidence)
        self.model = model

        if model_path is None:
            self.model_path = (
                Path(__file__).resolve().parent.parent / "yolov8n.pt"
            )
        else:
            self.model_path = Path(model_path)

    def load_model(self):
        if self.model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay YOLO model: {self.model_path}"
            )

        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        print("[INFO] YOLOv8 Person Detector Loaded")

    def parse_results(self, results):
        """Lay bounding box cua nguoi tu ket qua YOLO."""

        persons = []

        for result in results or []:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())

                if class_id != self.PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                if confidence < self.confidence:
                    continue
                persons.append(
                    (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        confidence
                    )
                )

        return persons

    def detect(self, frame, results=None):
        """
        Phat hien nguoi.
        Neu app.py da chay YOLO thi tai su dung results.
        """

        if frame is None or frame.size == 0:
            return []

        if results is None:
            self.load_model()

            results = self.model.predict(
                source=frame,
                classes=[self.PERSON_CLASS_ID],
                conf=self.confidence,
                imgsz=416,
                device="cpu",
                verbose=False
            )

        return self.parse_results(results)

    def crop_person(self, frame, person, padding=0.08):
        """
        Cat vung anh Person ROI.
        Tra ve anh BGR chua resize.
        """

        if frame is None or frame.size == 0 or person is None:
            return None

        h, w = frame.shape[:2]

        x1, y1, x2, y2, _ = person

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            return None

        pad_x = int(box_width * padding)
        pad_y = int(box_height * padding)

        x1 = max(0, int(x1) - pad_x)
        y1 = max(0, int(y1) - pad_y)

        x2 = min(w, int(x2) + pad_x)
        y2 = min(h, int(y2) + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        return roi.copy()

    def get_primary_person(self, persons):
        """
        Chon nguoi co bounding box lon nhat.
        Phu hop voi kich ban mot sinh vien truoc camera.
        """

        if not persons:
            return None

        return max(
            persons,
            key=lambda person: (
                (person[2] - person[0])
                * (person[3] - person[1])
            )
        )
