import sys
import time
import logging
import threading
import csv
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from modules.face_detector import FaceDetector
from modules.phone_detector import PhoneDetector
from modules.gaze_tracker import GazeTracker
from modules.head_pose import HeadPose as HeadPoseEstimator
from modules.pattern_classifier import FocusAnalyzer
from modules.person_detector import PersonDetector
from modules.report_generator import ReportGenerator
from modules.feature_extractor import FeatureExtractor


try:
    from modules.violation_logger import ViolationLogger
except ImportError:
    class ViolationLogger:
        def __init__(self):
            self.logs = []

        def log_violation(self, timestamp, violation_type, focus_score):
            self.logs.append({
                "timestamp": timestamp,
                "type": violation_type,
                "focus_score": focus_score
            })

        def get_logs(self):
            return self.logs


GAZE_TEXT_MAP = {
    "LEFT": "Looking Left",
    "RIGHT": "Looking Right",
    "DOWN": "Looking Down",
    "CENTER": "Looking Forward",
    "FORWARD": "Looking Forward",
    "CLOSED": "Eyes Closed",
    "EYES_CLOSED": "Eyes Closed",
    "NO_FACE": "Unknown",
    "UNKNOWN": "Unknown"
}


class SystemMonitorApp:

    def __init__(self):

        self.logger = logging.getLogger("SystemMonitorApp")
        self.cap = None

        # Camera thread
        self.capture_thread = None
        self.capture_running = False

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_id = 0

        # YOLO worker
        self.yolo_model = None
        self.yolo_thread = None
        self.yolo_running = False

        self.yolo_condition = threading.Condition()
        self.yolo_pending_frame = None

        self.detection_lock = threading.Lock()
        self.yolo_error = None

        # Configuration
        self.YOLO_EVERY_N_FRAMES = 3
        self.YOLO_IMAGE_SIZE = 416
        self.VISION_EVERY_N_FRAMES = 4

        #Nguong chung cho yolo va phat hien nguoi
        self.YOLO_CONFIDENCE = 0.25

        # Confidence rieng cho dien thoai
        self.PHONE_CONFIDENCE = 0.25
        self.PHONE_CLASSIFIER_THRESHOLD = 0.50

        # Modules
        self.face_detector = None
        self.phone_detector = None
        self.gaze_tracker = None
        self.head_pose_estimator = None
        self.focus_analyzer = None
        self.feature_extractor = None
        self.person_detector = None
        self.violation_logger = None

        self.report = ReportGenerator()

        # Session
        self.start_time = time.time()

        self.focus_score = 100.0
        self.violation_count = 0
        self.focus_status = "Focused"

        # Statistics
        self.total_frames = 0
        self.total_fps_sum = 0.0
        self.total_focus_sum = 0.0

        # Violation counts
        self.phone_violations = 0
        self.looking_left_count = 0
        self.looking_right_count = 0
        self.looking_down_count = 0
        self.eyes_closed_count = 0
        self.no_face_count = 0
        self.no_student_count = 0

        # Timing
        self.prev_time = time.time()
        self.fps_value = 0.0
        self.frame_count = 0

        self.current_gaze = "CENTER"
        self.cached_gaze = "UNKNOWN"

        self.gaze_start_time = time.time()
        self.gaze_logged = False

        self.last_feature_log = 0

        # Detection cache
        self.faces = []
        self.phones = []
        self.persons = []

        # Violation state
        self.phone_violation_active = False

        self.no_student_start_time = None
        self.no_student_logged = False

        self.no_face_start_time = None
        self.no_face_logged = False

        # Head pose
        self.head_direction = "Forward"
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

        # Colors BGR
        self.COLOR_BG = (30, 30, 30)
        self.COLOR_CARD_BG = (255, 255, 255)
        self.COLOR_PRIMARY = (235, 99, 37)
        self.COLOR_SUCCESS = (94, 197, 34)
        self.COLOR_WARNING = (11, 158, 245)
        self.COLOR_DANGER = (68, 68, 239)

        # Evaluation
        os.makedirs("evaluation", exist_ok=True)

        self.prediction_file = "evaluation/prediction.csv"
        self.last_save_time = time.time()

        with open(
            self.prediction_file,
            "w",
            newline="",
            encoding="utf8"
        ) as f:

            writer = csv.writer(f)
            writer.writerow(["Time", "Prediction"])

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def initialize(self):

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )

        self.logger.info("Initializing optimized monitoring system...")

        self.face_detector = FaceDetector()
        self.gaze_tracker = GazeTracker()
        self.head_pose_estimator = HeadPoseEstimator()

        self.focus_analyzer = FocusAnalyzer()
        self.violation_logger = ViolationLogger()
        self.feature_extractor = FeatureExtractor()

        # Load YOLO only once
        model_path = Path(__file__).resolve().parent / "yolov8n.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay YOLO weights: {model_path}"
            )

        self.yolo_model = YOLO(str(model_path))

        # Both detectors share the same YOLO model
        self.person_detector = PersonDetector(
            model=self.yolo_model,
            confidence=self.YOLO_CONFIDENCE
        )

        self.phone_detector = PhoneDetector(
            model=self.yolo_model,
            confidence=self.PHONE_CONFIDENCE,
            classifier_threshold=self.PHONE_CLASSIFIER_THRESHOLD,
        )
        self.phone_detector.load_classifier()

        self.logger.info("Shared YOLOv8 model loaded.")

        # Open camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise IOError("Webcam cannot be opened.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Start camera thread
        self.capture_running = True

        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )

        self.capture_thread.start()

        # Start YOLO worker
        self.yolo_running = True

        self.yolo_thread = threading.Thread(
            target=self._yolo_worker_loop,
            daemon=True
        )

        self.yolo_thread.start()

        self.logger.info("Camera and YOLO workers started.")

    # =====================================================
    # CAMERA THREAD
    # =====================================================

    def _capture_loop(self):

        while self.capture_running and self.cap is not None:

            success, frame = self.cap.read()

            if success and frame is not None:

                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_frame_id += 1

            else:
                time.sleep(0.005)

    def _read_latest_frame(self):

        with self.frame_lock:

            if self.latest_frame is None:
                return None, self.latest_frame_id

            return self.latest_frame.copy(), self.latest_frame_id

    # =====================================================
    # YOLO THREAD
    # =====================================================

    def _submit_yolo_frame(self, frame):

        with self.yolo_condition:

            # Chi giu frame moi nhat, khong xep hang frame cu
            self.yolo_pending_frame = frame.copy()

            self.yolo_condition.notify()

    def _yolo_worker_loop(self):

        while True:

            with self.yolo_condition:

                while (
                    self.yolo_running
                    and self.yolo_pending_frame is None
                ):
                    self.yolo_condition.wait(timeout=0.2)

                if not self.yolo_running:
                    return

                frame = self.yolo_pending_frame
                self.yolo_pending_frame = None

            try:

                # ONE YOLO inference for both person and phone
                results = self.yolo_model.predict(
                    source=frame,
                    classes=[0, 67],
                    conf=self.YOLO_CONFIDENCE,
                    imgsz=self.YOLO_IMAGE_SIZE,
                    device="cpu",
                    verbose=False
                )

                persons = self.person_detector.parse_results(results)
                phones = self.phone_detector.detect_phone(
                    frame,
                    results=results
                )

                with self.detection_lock:

                    self.persons = persons
                    self.phones = phones
                    self.yolo_error = None

            except Exception as error:

                with self.detection_lock:
                    self.yolo_error = str(error)

                self.logger.exception(
                    "YOLO worker failed: %s",
                    error
                )

    # =====================================================
    # PROCESS FRAME
    # =====================================================

    def process_frame(self, frame):

        if frame is None or frame.size == 0:
            return np.zeros((600, 800, 3), dtype=np.uint8)

        self.frame_count += 1

        h_orig, w_orig = frame.shape[:2]

        resized_view = cv2.resize(
            frame,
            (800, 600),
            interpolation=cv2.INTER_LINEAR
        )

        # Submit frame to YOLO worker every N frames
        if (
            self.frame_count == 1
            or self.frame_count % self.YOLO_EVERY_N_FRAMES == 0
        ):
            self._submit_yolo_frame(frame)

        # MediaPipe runs every 2 frames
        if (
            self.frame_count == 1
            or self.frame_count % self.VISION_EVERY_N_FRAMES == 0
        ):

            faces_detected = self.face_detector.detect_faces(frame)

            self.faces = (
                faces_detected if faces_detected is not None else []
            )

            gaze_res = self.gaze_tracker.detect(frame)

            if isinstance(gaze_res, tuple):
                gaze_str = gaze_res[0]
            else:
                gaze_str = gaze_res

            self.cached_gaze = (
                str(gaze_str).upper()
                if gaze_str else "UNKNOWN"
            )

        else:
            gaze_str = self.cached_gaze

        gaze_str = str(gaze_str).upper() if gaze_str else "UNKNOWN"

        # Get latest YOLO result
        with self.detection_lock:

            self.persons = list(self.persons)
            self.phones = list(self.phones)

        # Head pose
        if len(self.faces) > 0:

            face = self.faces[0]

            try:

                result = self.head_pose_estimator.get_direction(
                    frame,
                    face,
                    self.gaze_tracker.last_landmarks
                )

                if isinstance(result, tuple) and len(result) == 4:

                    (
                        self.pitch,
                        self.yaw,
                        self.roll,
                        self.head_direction
                    ) = result

                else:

                    self.head_direction = str(result)

                    self.pitch = 0.0
                    self.yaw = 0.0
                    self.roll = 0.0

            except Exception:

                self.pitch = 0.0
                self.yaw = 0.0
                self.roll = 0.0
                self.head_direction = "Forward"

        else:

            self.head_direction = "Forward"
            self.pitch = 0.0
            self.yaw = 0.0
            self.roll = 0.0

        # Gaze violation
        if gaze_str != self.current_gaze:

            self.current_gaze = gaze_str
            self.gaze_start_time = time.time()
            self.gaze_logged = False

        else:

            duration = time.time() - self.gaze_start_time

            if (
                gaze_str == "LEFT"
                and duration >= 3.0
                and not self.gaze_logged
            ):

                self.log_violation_event("LOOKING LEFT")
                self.looking_left_count += 1
                self.gaze_logged = True

            elif (
                gaze_str == "RIGHT"
                and duration >= 3.0
                and not self.gaze_logged
            ):

                self.log_violation_event("LOOKING RIGHT")
                self.looking_right_count += 1
                self.gaze_logged = True

            elif (
                gaze_str == "DOWN"
                and duration >= 3.0
                and not self.gaze_logged
            ):

                self.log_violation_event("LOOKING DOWN")
                self.looking_down_count += 1
                self.gaze_logged = True

            elif (
                gaze_str in ["CLOSED", "EYES_CLOSED"]
                and duration >= 2.0
                and not self.gaze_logged
            ):

                self.log_violation_event("EYES CLOSED")
                self.eyes_closed_count += 1
                self.gaze_logged = True

        # Phone violation
        if len(self.phones) > 0:

            if not self.phone_violation_active:

                self.log_violation_event("PHONE DETECTED")
                self.phone_violations += 1
                self.phone_violation_active = True

        else:
            self.phone_violation_active = False

        # Student presence
        now = time.time()

        student_present = (
            len(self.persons) > 0 or len(self.faces) > 0
        )

        if not student_present:

            if self.no_student_start_time is None:
                self.no_student_start_time = now

            elif (
                now - self.no_student_start_time >= 1.0
                and not self.no_student_logged
            ):

                self.log_violation_event("NO STUDENT DETECTED")
                self.no_student_count += 1
                self.no_student_logged = True

        else:

            self.no_student_start_time = None
            self.no_student_logged = False

        # No face
        if len(self.faces) == 0:

            if self.no_face_start_time is None:
                self.no_face_start_time = now

            elif (
                now - self.no_face_start_time >= 1.0
                and not self.no_face_logged
            ):

                self.log_violation_event("NO FACE")
                self.no_face_count += 1
                self.no_face_logged = True

        else:

            self.no_face_start_time = None
            self.no_face_logged = False

        # Feature extraction
        features = self.feature_extractor.extract(
            face_present=len(self.faces) > 0,
            phone_detected=len(self.phones) > 0,
            gaze_direction=gaze_str,
            head_direction=self.head_direction,
            eyes_closed=gaze_str in ["CLOSED", "EYES_CLOSED"],
            yaw=self.yaw
        )

        # Rule-based classification
        self.focus_status = self.focus_analyzer.classify(features)

        if time.time() - self.last_feature_log >= 1.0:

            self.log_feature_vector(features)
            self.last_feature_log = time.time()

        # Save prediction every second
        current_time = time.time()

        if current_time - self.last_save_time >= 1.0:

            with open(
                self.prediction_file,
                "a",
                newline="",
                encoding="utf8"
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    int(current_time - self.start_time),
                    self.focus_status
                ])

            self.last_save_time = current_time

        self.update_focus_score(gaze_str, student_present)
        self.update_statistics()

        self.draw_camera_overlay(
            resized_view,
            w_orig,
            h_orig
        )

        self.draw_warning(resized_view)

        return resized_view

    # =====================================================
    # LOGGING
    # =====================================================

    def log_feature_vector(self, features):

        os.makedirs("evaluation", exist_ok=True)

        file_path = "evaluation/feature_log.csv"

        file_exists = os.path.exists(file_path)

        with open(
            file_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            if not file_exists:

                writer.writerow([
                    "Timestamp",
                    "Yaw",
                    "GazeDirection",
                    "HeadDirection",
                    "EyesClosed",
                    "PhoneDetected",
                    "FacePresent",
                    "Prediction"
                ])

            writer.writerow([
                round(time.time() - self.start_time, 2),
                features["yaw"],
                features["gaze_direction"],
                features["head_direction"],
                features["eyes_closed"],
                features["phone_detected"],
                features["face_present"],
                self.focus_status
            ])

    def log_violation_event(self, violation_type):

        self.violation_count += 1

        timestamp = time.strftime("%H:%M:%S")

        self.logger.warning(
            f"Violation: {violation_type} at {timestamp}. "
            f"Focus: {self.focus_score:.0f}%"
        )

        if hasattr(self.violation_logger, "log_violation"):

            try:

                self.violation_logger.log_violation(
                    timestamp,
                    violation_type,
                    self.focus_score
                )

            except Exception:
                pass

    # =====================================================
    # FOCUS SCORE
    # =====================================================

    def update_focus_score(self, gaze_str, student_present):

        penalty = 0

        if not student_present:

            penalty = -20

        elif len(self.faces) == 0:

            penalty = -15

        else:

            if len(self.phones) > 0:
                penalty -= 20

            if gaze_str == "LEFT":
                penalty -= 5

            elif gaze_str == "RIGHT":
                penalty -= 5

            elif gaze_str == "DOWN":
                penalty -= 8

            elif gaze_str in ["CLOSED", "EYES_CLOSED"]:
                penalty -= 10

        target_score = 100.0 + penalty

        target_score = max(
            0.0,
            min(100.0, target_score)
        )

        self.focus_score = (
            0.94 * self.focus_score
            + 0.06 * target_score
        )

        self.focus_score = max(
            0.0,
            min(100.0, self.focus_score)
        )

    def update_statistics(self):

        self.total_frames += 1
        self.total_fps_sum += self.fps_value
        self.total_focus_sum += self.focus_score

    # =====================================================
    # DRAW ICON
    # =====================================================

    def draw_vector_icon(self, img, icon_name, center, color):

        ix, iy = center

        if icon_name == "student":

            cv2.circle(img, (ix, iy - 4), 4, color, -1, cv2.LINE_AA)

            cv2.ellipse(
                img,
                (ix, iy + 7),
                (8, 5),
                0, 180, 360,
                color,
                -1,
                cv2.LINE_AA
            )

        elif icon_name == "phone":

            cv2.rectangle(
                img,
                (ix - 5, iy - 9),
                (ix + 5, iy + 9),
                color,
                1,
                cv2.LINE_AA
            )

            cv2.rectangle(
                img,
                (ix - 4, iy - 8),
                (ix + 4, iy + 6),
                color,
                -1
            )

            cv2.circle(img, (ix, iy + 7), 1, (255, 255, 255), -1)

        elif icon_name == "eye":

            cv2.ellipse(
                img,
                (ix, iy),
                (9, 5),
                0, 0, 360,
                color,
                1,
                cv2.LINE_AA
            )

            cv2.circle(img, (ix, iy), 3, color, -1, cv2.LINE_AA)

        elif icon_name == "head":

            cv2.circle(img, (ix, iy - 2), 6, color, -1, cv2.LINE_AA)

            cv2.rectangle(
                img,
                (ix - 2, iy + 4),
                (ix + 2, iy + 9),
                color,
                -1
            )

        elif icon_name == "focus":

            cv2.circle(img, (ix, iy), 8, color, 1, cv2.LINE_AA)
            cv2.circle(img, (ix, iy), 4, color, 1, cv2.LINE_AA)
            cv2.circle(img, (ix, iy), 1, color, -1)

        elif icon_name == "warning":

            pts = np.array(
                [
                    [ix, iy - 9],
                    [ix - 9, iy + 8],
                    [ix + 9, iy + 8]
                ],
                np.int32
            )

            cv2.fillPoly(img, [pts], color)

            cv2.line(
                img,
                (ix, iy - 3),
                (ix, iy + 2),
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.circle(img, (ix, iy + 5), 1, (255, 255, 255), -1)

        elif icon_name == "fps":

            cv2.line(
                img,
                (ix - 8, iy + 7),
                (ix + 8, iy + 7),
                color,
                1,
                cv2.LINE_AA
            )

            cv2.line(
                img,
                (ix - 8, iy - 7),
                (ix - 8, iy + 7),
                color,
                1,
                cv2.LINE_AA
            )

            pts = np.array(
                [
                    [ix - 6, iy + 4],
                    [ix - 2, iy - 1],
                    [ix + 2, iy + 3],
                    [ix + 6, iy - 5]
                ],
                np.int32
            )

            cv2.polylines(
                img,
                [pts],
                False,
                color,
                2,
                cv2.LINE_AA
            )

        elif icon_name == "time":

            cv2.circle(img, (ix, iy), 8, color, 1, cv2.LINE_AA)
            cv2.line(img, (ix, iy), (ix, iy - 4), color, 1, cv2.LINE_AA)
            cv2.line(img, (ix, iy), (ix + 3, iy), color, 1, cv2.LINE_AA)

        elif icon_name == "system":

            cv2.rectangle(
                img,
                (ix - 8, iy - 6),
                (ix + 8, iy + 3),
                color,
                1,
                cv2.LINE_AA
            )

            cv2.line(img, (ix - 3, iy + 4), (ix - 3, iy + 7), color, 1, cv2.LINE_AA)
            cv2.line(img, (ix + 3, iy + 4), (ix + 3, iy + 7), color, 1, cv2.LINE_AA)
            cv2.line(img, (ix - 6, iy + 7), (ix + 6, iy + 7), color, 1, cv2.LINE_AA)

    # =====================================================
    # DRAW CARD
    # =====================================================

    def draw_card(
        self,
        canvas,
        x,
        y,
        w,
        h,
        title,
        value,
        status_color,
        icon_name,
        progress=None,
        has_dot=False
    ):

        cv2.rectangle(
            canvas,
            (x + 2, y + 2),
            (x + w + 2, y + h + 2),
            (25, 20, 15),
            -1
        )

        cv2.rectangle(
            canvas,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            -1
        )

        cv2.rectangle(
            canvas,
            (x, y),
            (x + 5, y + h),
            status_color,
            -1
        )

        cv2.rectangle(
            canvas,
            (x, y),
            (x + w, y + h),
            (226, 232, 240),
            1,
            cv2.LINE_AA
        )

        self.draw_vector_icon(
            canvas,
            icon_name,
            (x + 22, y + 32),
            status_color
        )

        cv2.putText(
            canvas,
            title,
            (x + 42, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (100, 116, 139),
            1,
            cv2.LINE_AA
        )

        val_y = y + 44 if progress is not None else y + 48

        if has_dot:

            cv2.circle(
                canvas,
                (x + 46, y + 42),
                4,
                status_color,
                -1,
                cv2.LINE_AA
            )

            cv2.putText(
                canvas,
                value,
                (x + 56, val_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (30, 41, 59),
                2,
                cv2.LINE_AA
            )

        else:

            cv2.putText(
                canvas,
                value,
                (x + 42, val_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (30, 41, 59),
                2,
                cv2.LINE_AA
            )

        if progress is not None:

            bar_x = x + 42
            bar_y = y + 52
            bar_w = w - 60
            bar_h = 4

            cv2.rectangle(
                canvas,
                (bar_x, bar_y),
                (bar_x + bar_w, bar_y + bar_h),
                (241, 245, 249),
                -1
            )

            fill_w = int(
                bar_w * (max(0.0, min(100.0, progress)) / 100.0)
            )

            if fill_w > 0:

                cv2.rectangle(
                    canvas,
                    (bar_x, bar_y),
                    (bar_x + fill_w, bar_y + bar_h),
                    status_color,
                    -1
                )

    # =====================================================
    # CAMERA OVERLAY
    # =====================================================

    def draw_camera_overlay(self, frame, w_orig, h_orig):

        # Person boxes
        for person in self.persons:

            px1, py1, px2, py2, confidence = person

            x1 = int(px1 * 800 / w_orig)
            y1 = int(py1 * 600 / h_orig)
            x2 = int(px2 * 800 / w_orig)
            y2 = int(py2 * 600 / h_orig)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                self.COLOR_PRIMARY,
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Person {confidence:.2f}",
                (x1 + 5, max(20, y1 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.COLOR_PRIMARY,
                1,
                cv2.LINE_AA
            )

        # Face boxes
        for face in self.faces:

            fx, fy, fw, fh = face

            x1 = int(fx * 800 / w_orig)
            y1 = int(fy * 600 / h_orig)

            x2 = int((fx + fw) * 800 / w_orig)
            y2 = int((fy + fh) * 600 / h_orig)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                self.COLOR_SUCCESS,
                2,
                cv2.LINE_AA
            )

            gaze_txt = GAZE_TEXT_MAP.get(
                self.current_gaze,
                "Unknown"
            )

            label = (
                f"Face | Gaze: {gaze_txt} "
                f"| Pose: {self.head_direction}"
            )

            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                1
            )

            badge_y = max(0, y1 - (th + 10))

            cv2.rectangle(
                frame,
                (x1, badge_y),
                (x1 + tw + 14, y1),
                self.COLOR_SUCCESS,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1 + 7, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Phone boxes
        for phone in self.phones:

            px1, py1, px2, py2, confidence = phone

            x1 = int(px1 * 800 / w_orig)
            y1 = int(py1 * 600 / h_orig)
            x2 = int(px2 * 800 / w_orig)
            y2 = int(py2 * 600 / h_orig)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                self.COLOR_DANGER,
                2,
                cv2.LINE_AA
            )

            label = f"PHONE DETECTED | Conf: {confidence:.2f}"

            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                1
            )

            badge_y = max(0, y1 - (th + 10))

            cv2.rectangle(
                frame,
                (x1, badge_y),
                (x1 + tw + 14, y1),
                self.COLOR_DANGER,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1 + 7, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    # =====================================================
    # WARNING
    # =====================================================

    def draw_warning(self, frame):

        active_warning = None
        color = self.COLOR_DANGER

        student_present = (
            len(self.persons) > 0 or len(self.faces) > 0
        )

        if len(self.phones) > 0:

            active_warning = "PHONE DETECTED"
            color = self.COLOR_DANGER

        elif not student_present:

            active_warning = "NO STUDENT"
            color = self.COLOR_DANGER

        elif len(self.faces) == 0:

            active_warning = "NO FACE"
            color = self.COLOR_WARNING

        elif (
            self.current_gaze == "LEFT"
            and time.time() - self.gaze_start_time >= 1.5
        ):

            active_warning = "LOOKING LEFT"
            color = self.COLOR_WARNING

        elif (
            self.current_gaze == "RIGHT"
            and time.time() - self.gaze_start_time >= 1.5
        ):

            active_warning = "LOOKING RIGHT"
            color = self.COLOR_WARNING

        elif (
            self.current_gaze == "DOWN"
            and time.time() - self.gaze_start_time >= 1.5
        ):

            active_warning = "LOOKING DOWN"
            color = self.COLOR_WARNING

        elif (
            self.current_gaze in ["CLOSED", "EYES_CLOSED"]
            and time.time() - self.gaze_start_time >= 1.0
        ):

            active_warning = "EYES CLOSED"
            color = self.COLOR_DANGER

        elif self.focus_score < 50:

            active_warning = "CRITICAL LOW FOCUS"
            color = self.COLOR_DANGER

        elif self.focus_score < 70:

            active_warning = "LOW FOCUS"
            color = self.COLOR_WARNING

        if active_warning is None:
            return

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (20, 20),
            (780, 65),
            color,
            -1
        )

        cv2.addWeighted(
            overlay,
            0.85,
            frame,
            0.15,
            0,
            frame
        )

        cv2.rectangle(
            frame,
            (20, 20),
            (780, 65),
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        pts = np.array(
            [
                [45, 28],
                [36, 46],
                [54, 46]
            ],
            np.int32
        )

        cv2.fillPoly(frame, [pts], (255, 255, 255))

        cv2.line(
            frame,
            (45, 33),
            (45, 41),
            color,
            2
        )

        cv2.circle(
            frame,
            (45, 44),
            1,
            color,
            -1
        )

        cv2.putText(
            frame,
            f"ALERT: {active_warning}",
            (70, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # =====================================================
    # DASHBOARD
    # =====================================================

    def draw_dashboard(self, canvas):

        cv2.putText(
            canvas,
            "Student Focus Monitoring System",
            (40, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        current_time_str = time.strftime(
            "Current Time: %Y-%m-%d %H:%M:%S"
        )

        cv2.putText(
            canvas,
            current_time_str,
            (40, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (148, 163, 184),
            1,
            cv2.LINE_AA
        )

        fps_text = f"FPS: {self.fps_value:.1f}"

        (text_w, text_h), _ = cv2.getTextSize(
            fps_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            2
        )

        badge_w = text_w + 24
        badge_h = 30
        badge_x = 1240 - badge_w
        badge_y = 25

        cv2.rectangle(
            canvas,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            self.COLOR_PRIMARY,
            -1,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            fps_text,
            (badge_x + 12, badge_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        student_present = (
            len(self.persons) > 0 or len(self.faces) > 0
        )

        student_val = "Present" if student_present else "Absent"

        student_color = (
            self.COLOR_SUCCESS
            if student_present
            else self.COLOR_DANGER
        )

        phone_present = len(self.phones) > 0

        phone_val = "Detected" if phone_present else "None"

        phone_color = (
            self.COLOR_DANGER
            if phone_present
            else self.COLOR_SUCCESS
        )

        gaze_txt = GAZE_TEXT_MAP.get(
            self.current_gaze,
            "Unknown"
        )

        if gaze_txt == "Looking Forward":
            gaze_color = self.COLOR_SUCCESS
        elif gaze_txt == "Unknown":
            gaze_color = self.COLOR_DANGER
        else:
            gaze_color = self.COLOR_WARNING

        head_color = (
            self.COLOR_SUCCESS
            if self.head_direction == "Forward"
            else self.COLOR_WARNING
        )

        if self.focus_score >= 80:
            focus_color = self.COLOR_SUCCESS
        elif self.focus_score >= 50:
            focus_color = self.COLOR_WARNING
        else:
            focus_color = self.COLOR_DANGER

        if self.violation_count == 0:
            violation_color = self.COLOR_SUCCESS
        elif self.violation_count < 3:
            violation_color = self.COLOR_WARNING
        else:
            violation_color = self.COLOR_DANGER

        status_color = {
            "Focused": self.COLOR_SUCCESS,
            "Looking Left": self.COLOR_WARNING,
            "Looking Right": self.COLOR_WARNING,
            "Looking Down": self.COLOR_WARNING,
            "Head Left": self.COLOR_WARNING,
            "Head Right": self.COLOR_WARNING,
            "Eyes Closed": self.COLOR_DANGER,
            "Phone Detected": self.COLOR_DANGER,
            "No Face": self.COLOR_DANGER
        }

        sys_color = status_color.get(
            self.focus_status,
            self.COLOR_WARNING
        )

        # Student
        self.draw_card(
            canvas, 880, 90, 170, 68,
            "STUDENT",
            student_val,
            student_color,
            "student",
            has_dot=True
        )

        # Phone
        self.draw_card(
            canvas, 1070, 90, 170, 68,
            "PHONE",
            phone_val,
            phone_color,
            "phone",
            has_dot=True
        )

        # Eye direction
        self.draw_card(
            canvas, 880, 170, 170, 68,
            "EYE DIRECTION",
            gaze_txt,
            gaze_color,
            "eye"
        )

        # Head direction
        self.draw_card(
            canvas, 1070, 170, 170, 68,
            "HEAD DIRECTION",
            self.head_direction,
            head_color,
            "head"
        )

        # Focus score
        self.draw_card(
            canvas, 880, 250, 170, 68,
            "FOCUS SCORE",
            f"{self.focus_score:.0f}%",
            focus_color,
            "focus",
            progress=self.focus_score
        )

        # Violations
        self.draw_card(
            canvas, 1070, 250, 170, 68,
            "VIOLATIONS",
            str(self.violation_count),
            violation_color,
            "warning"
        )

        # FPS
        self.draw_card(
            canvas, 880, 330, 170, 68,
            "FPS",
            f"{self.fps_value:.1f} FPS",
            self.COLOR_PRIMARY,
            "fps"
        )

        # Current time
        self.draw_card(
            canvas, 1070, 330, 170, 68,
            "CURRENT TIME",
            time.strftime("%H:%M:%S"),
            self.COLOR_PRIMARY,
            "time"
        )

        # System status
        self.draw_card(
            canvas, 880, 410, 170, 68,
            "SYSTEM STATUS",
            self.focus_status,
            sys_color,
            "system",
            has_dot=True
        )

        uptime = int(time.time() - self.start_time)

        self.draw_card(
            canvas, 1070, 410, 170, 68,
            "RUNTIME",
            f"{uptime} sec",
            self.COLOR_PRIMARY,
            "time"
        )

        self.draw_session_stats(
            canvas,
            880,
            490,
            360,
            200
        )

    # =====================================================
    # SESSION STATISTICS
    # =====================================================

    def draw_session_stats(self, canvas, x, y, w, h):

        cv2.rectangle(
            canvas,
            (x + 2, y + 2),
            (x + w + 2, y + h + 2),
            (25, 20, 15),
            -1
        )

        cv2.rectangle(
            canvas,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            -1
        )

        cv2.rectangle(
            canvas,
            (x, y),
            (x + w, y + h),
            (226, 232, 240),
            1,
            cv2.LINE_AA
        )

        cv2.rectangle(
            canvas,
            (x, y),
            (x + 5, y + h),
            self.COLOR_PRIMARY,
            -1
        )

        cv2.putText(
            canvas,
            "SESSION STATISTICS",
            (x + 18, y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.COLOR_PRIMARY,
            2,
            cv2.LINE_AA
        )

        divisor = max(1, self.total_frames)

        avg_fps = self.total_fps_sum / divisor
        avg_focus = self.total_focus_sum / divisor

        left_stats = [
            f"Total Frames: {self.total_frames}",
            f"Average FPS: {avg_fps:.1f}",
            f"Average Focus: {avg_focus:.1f}%",
            f"Total Violations: {self.violation_count}"
        ]

        right_stats = [
            f"Phone Viols: {self.phone_violations}",
            f"Left / Right: {self.looking_left_count} / {self.looking_right_count}",
            f"Down / Closed: {self.looking_down_count} / {self.eyes_closed_count}",
            f"No Face / Student: {self.no_face_count} / {self.no_student_count}"
        ]

        text_y = y + 54

        for stat in left_stats:

            cv2.putText(
                canvas,
                stat,
                (x + 18, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (51, 65, 85),
                1,
                cv2.LINE_AA
            )

            text_y += 34

        text_y = y + 54

        for stat in right_stats:

            cv2.putText(
                canvas,
                stat,
                (x + 190, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (51, 65, 85),
                1,
                cv2.LINE_AA
            )

            text_y += 34

    # =====================================================
    # FPS
    # =====================================================

    def calculate_fps(self):

        current = time.time()

        elapsed = current - self.prev_time
        self.prev_time = current

        instant_fps = 1.0 / max(1e-5, elapsed)

        self.fps_value = (
            0.92 * self.fps_value
            + 0.08 * instant_fps
        )

        return self.fps_value

    # =====================================================
    # CLEANUP
    # =====================================================

    def cleanup(self):

        self.logger.info("Cleaning up monitoring system...")

        # Stop camera thread
        self.capture_running = False

        if (
            self.capture_thread is not None
            and self.capture_thread.is_alive()
        ):

            self.capture_thread.join(timeout=1.0)

        # Stop YOLO thread
        with self.yolo_condition:

            self.yolo_running = False
            self.yolo_condition.notify_all()

        if (
            self.yolo_thread is not None
            and self.yolo_thread.is_alive()
        ):

            self.yolo_thread.join(timeout=2.0)

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()

        self.logger.info("Resources released.")

    # =====================================================
    # MAIN
    # =====================================================

    def main(self):

        try:

            self.initialize()

            window_name = "Student Focus Monitoring System"

            cv2.namedWindow(
                window_name,
                cv2.WINDOW_NORMAL
            )

            cv2.resizeWindow(
                window_name,
                1280,
                720
            )

            self.logger.info(
                "Optimized system started. Press ESC to stop."
            )

            last_frame_id = -1

            while True:

                frame, frame_id = self._read_latest_frame()

                if frame is None or frame_id == last_frame_id:

                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                    time.sleep(0.001)
                    continue

                last_frame_id = frame_id

                frame = cv2.flip(frame, 1)

                self.calculate_fps()

                processed_cam = self.process_frame(frame)

                canvas = np.zeros(
                    (720, 1280, 3),
                    dtype=np.uint8
                )

                canvas[:] = (42, 23, 15)

                canvas[90:690, 40:840] = processed_cam

                cv2.rectangle(
                    canvas,
                    (40, 90),
                    (840, 690),
                    (71, 85, 105),
                    1,
                    cv2.LINE_AA
                )

                c_len = 20
                c_color = self.COLOR_PRIMARY

                # Top-left
                cv2.line(canvas, (40, 90), (40 + c_len, 90), c_color, 3, cv2.LINE_AA)
                cv2.line(canvas, (40, 90), (40, 90 + c_len), c_color, 3, cv2.LINE_AA)

                # Top-right
                cv2.line(canvas, (840, 90), (840 - c_len, 90), c_color, 3, cv2.LINE_AA)
                cv2.line(canvas, (840, 90), (840, 90 + c_len), c_color, 3, cv2.LINE_AA)

                # Bottom-left
                cv2.line(canvas, (40, 690), (40 + c_len, 690), c_color, 3, cv2.LINE_AA)
                cv2.line(canvas, (40, 690), (40, 690 - c_len), c_color, 3, cv2.LINE_AA)

                # Bottom-right
                cv2.line(canvas, (840, 690), (840 - c_len, 690), c_color, 3, cv2.LINE_AA)
                cv2.line(canvas, (840, 690), (840, 690 - c_len), c_color, 3, cv2.LINE_AA)

                self.draw_dashboard(canvas)

                cv2.imshow(
                    window_name,
                    canvas
                )

                if cv2.waitKey(1) & 0xFF == 27:

                    self.logger.info("Termination requested.")
                    break

        except KeyboardInterrupt:

            self.logger.info("System stopped by user.")

        except Exception as error:

            self.logger.critical(
                "System loop crashed: %s",
                error,
                exc_info=True
            )

        finally:

            self.cleanup()

            divisor = max(1, self.total_frames)

            avg_fps = self.total_fps_sum / divisor
            avg_focus = self.total_focus_sum / divisor

            runtime = int(time.time() - self.start_time)

            from datetime import datetime

            report_data = {
                "filename": (
                    "reports/report_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S")
                    + ".csv"
                ),
                "Date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "Runtime (s)": runtime,
                "Average FPS": round(avg_fps, 2),
                "Average Focus Score": round(avg_focus, 2),
                "Total Violations": self.violation_count,
                "Phone Violations": self.phone_violations,
                "Looking Left": self.looking_left_count,
                "Looking Right": self.looking_right_count,
                "Looking Down": self.looking_down_count,
                "Eyes Closed": self.eyes_closed_count,
                "No Face": self.no_face_count,
                "No Student": self.no_student_count
            }

            saved_file = self.report.generate(report_data)

            print("===== SESSION REPORT =====")
            print(f"Total Runtime: {runtime} seconds")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Average Focus Score: {avg_focus:.1f}%")
            print(f"Total Violations: {self.violation_count}")
            print(f"Phone Violations: {self.phone_violations}")
            print(f"Left: {self.looking_left_count}")
            print(f"Right: {self.looking_right_count}")
            print(f"Down: {self.looking_down_count}")
            print(f"Eyes Closed: {self.eyes_closed_count}")
            print(f"No Face: {self.no_face_count}")
            print(f"No Student: {self.no_student_count}")
            print(f"CSV Report Saved: {saved_file}")


if __name__ == "__main__":

    app_instance = SystemMonitorApp()
    app_instance.main()
