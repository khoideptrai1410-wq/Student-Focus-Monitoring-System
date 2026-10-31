import unittest
from types import SimpleNamespace

import numpy as np

from modules.phone_detector import PhoneDetector


class FakeValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCoordinates:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def tolist(self):
        return self.coordinates


class FakeBox:
    def __init__(self, class_id=67, confidence=0.95):
        self.cls = [FakeValue(class_id)]
        self.conf = [FakeValue(confidence)]
        self.xyxy = [FakeCoordinates([10, 10, 50, 60])]


class FakeClassifier:
    def __init__(self, ready, prediction=None):
        self.ready = ready
        self.prediction = prediction
        self.predict_calls = 0

    def load_model(self):
        return self.ready

    def predict(self, image):
        self.predict_calls += 1
        return self.prediction


class PhoneDetectorTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.results = [SimpleNamespace(boxes=[FakeBox()])]

    def test_rejects_yolo_candidate_when_classifier_is_unavailable(self):
        classifier = FakeClassifier(ready=False)
        detector = PhoneDetector(classifier=classifier)

        detections = detector.detect_phone(self.frame, results=self.results)

        self.assertEqual(detections, [])
        self.assertEqual(classifier.predict_calls, 0)

    def test_returns_candidate_only_when_classifier_confirms_phone(self):
        classifier = FakeClassifier(
            ready=True,
            prediction={"is_phone": True, "phone_probability": 0.93},
        )
        detector = PhoneDetector(classifier=classifier)

        detections = detector.detect_phone(self.frame, results=self.results)

        self.assertEqual(detections, [(10, 10, 50, 60, 0.93)])
        self.assertEqual(classifier.predict_calls, 1)

    def test_rejects_candidate_when_classifier_says_not_phone(self):
        classifier = FakeClassifier(
            ready=True,
            prediction={"is_phone": False, "phone_probability": 0.12},
        )
        detector = PhoneDetector(classifier=classifier)

        detections = detector.detect_phone(self.frame, results=self.results)

        self.assertEqual(detections, [])
        self.assertEqual(classifier.predict_calls, 1)


if __name__ == "__main__":
    unittest.main()
