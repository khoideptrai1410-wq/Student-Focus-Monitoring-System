from PyQt5.QtWidgets import (
QWidget,
QLabel,
QVBoxLayout,
QHBoxLayout
)

from PyQt5.QtCore import Qt

class ExamScreen(QWidget):

    def __init__(
        self,
        name,
        cls,
        sbd,
        duration
    ):

        super().__init__()

        self.setWindowTitle(
            "Student Focus Monitoring"
        )

        self.showFullScreen()

        main_layout = QVBoxLayout()

        main_layout.setContentsMargins(
            20,
            20,
            20,
            20
        )

        # =========================
        # HEADER
        # =========================

        header_layout = QHBoxLayout()

        self.info_label = QLabel(
            f"Họ tên: {name}   |   "
            f"Lớp: {cls}   |   "
            f"SBD: {sbd}"
        )

        self.info_label.setStyleSheet("""
            font-size:18px;
            font-weight:bold;
            color:white;
        """)

        self.timer_label = QLabel(
            f"{duration}:00"
        )

        self.timer_label.setStyleSheet("""
            font-size:22px;
            font-weight:bold;
            color:#ff5555;
        """)

        header_layout.addWidget(
            self.info_label
        )

        header_layout.addStretch()

        header_layout.addWidget(
            self.timer_label
        )

        # =========================
        # CAMERA
        # =========================

        self.camera_label = QLabel()

        self.camera_label.setAlignment(
            Qt.AlignCenter
        )

        self.camera_label.setStyleSheet("""
            background:black;
            border:2px solid #2c3e50;
            border-radius:10px;
        """)

        self.camera_label.setMinimumSize(
            1000,
            700
        )

        # =========================
        # STATUS
        # =========================

        self.status_label = QLabel(
            "STATUS: CENTER"
        )

        self.status_label.setAlignment(
            Qt.AlignCenter
        )

        self.status_label.setStyleSheet("""
            font-size:18px;
            font-weight:bold;
            color:#00ff88;
            padding:10px;
        """)

        # =========================
        # LAYOUT
        # =========================

        main_layout.addLayout(
            header_layout
        )

        main_layout.addSpacing(
            10
        )

        main_layout.addWidget(
            self.camera_label,
            stretch=1
        )

        main_layout.addSpacing(
            10
        )

        main_layout.addWidget(
            self.status_label
        )

        self.setLayout(
            main_layout
        )

        # =========================
        # WINDOW STYLE
        # =========================

        self.setStyleSheet("""
            QWidget{
                background:#0f172a;
            }
        """)
