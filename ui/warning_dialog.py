from PyQt5.QtWidgets import QMessageBox


class WarningDialog:

    @staticmethod
    def warning1():

        msg = QMessageBox()

        msg.setWindowTitle(
            "Cảnh báo"
        )

        msg.setText(
            "Vui lòng tập trung vào màn hình"
        )

        msg.exec_()

    @staticmethod
    def warning2():

        msg = QMessageBox()

        msg.setWindowTitle(
            "Cảnh báo nghiêm trọng"
        )

        msg.setText(
            "Bạn đang mất tập trung."
        )

        msg.exec_()