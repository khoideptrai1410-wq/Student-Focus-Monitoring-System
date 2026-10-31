from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout
)

from PyQt5.QtCore import Qt


class LockScreen(QWidget):

    def __init__(
        self,
        reason
    ):

        super().__init__()

        self.setWindowTitle(
            "SYSTEM LOCKED"
        )

        self.showFullScreen()

        layout = QVBoxLayout()

        title = QLabel(
            "HỆ THỐNG ĐÃ BỊ KHÓA"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        reason_label = QLabel(
            reason
        )

        reason_label.setAlignment(
            Qt.AlignCenter
        )

        teacher_btn = QPushButton(
            "Yêu cầu giáo viên mở khóa"
        )

        layout.addStretch()

        layout.addWidget(title)

        layout.addWidget(
            reason_label
        )

        layout.addWidget(
            teacher_btn
        )

        layout.addStretch()

        self.setLayout(
            layout
        )