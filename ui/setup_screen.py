from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox
)

from PyQt5.QtCore import pyqtSignal


class SetupScreen(QWidget):

    start_exam = pyqtSignal(
        str,  # name
        str,  # class
        str,  # sbd
        int   # duration
    )

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Student Focus Monitoring"
        )

        self.resize(500, 500)

        layout = QVBoxLayout()

        title = QLabel(
            "STUDENT FOCUS MONITORING"
        )

        self.name = QLineEdit()
        self.name.setPlaceholderText(
            "Họ và tên"
        )

        self.cls = QLineEdit()
        self.cls.setPlaceholderText(
            "Lớp"
        )

        self.sbd = QLineEdit()
        self.sbd.setPlaceholderText(
            "Số báo danh"
        )

        self.time_box = QComboBox()

        self.time_box.addItems(
            [
                "15",
                "45",
                "90"
            ]
        )

        self.start_btn = QPushButton(
            "Bắt đầu làm bài"
        )

        self.start_btn.clicked.connect(
            self.start_clicked
        )

        layout.addWidget(title)
        layout.addWidget(self.name)
        layout.addWidget(self.cls)
        layout.addWidget(self.sbd)
        layout.addWidget(self.time_box)
        layout.addWidget(self.start_btn)

        self.setLayout(layout)

    def start_clicked(self):

        if (
            not self.name.text()
            or not self.cls.text()
            or not self.sbd.text()
        ):
            return

        self.start_exam.emit(
            self.name.text(),
            self.cls.text(),
            self.sbd.text(),
            int(
                self.time_box.currentText()
            )
        )