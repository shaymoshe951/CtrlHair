from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QProgressDialog
)
from PyQt5.QtCore import QThread, pyqtSignal
import time


class WorkerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def run(self):
        for i in range(1, 101):
            time.sleep(0.05)  # Simulate long task
            self.progress.emit(i)
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")

        self.button = QPushButton("Start Operation")
        self.button.clicked.connect(self.start_operation)

        layout = QVBoxLayout()
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_operation(self):
        # Disable main window (freeze)
        self.setEnabled(False)

        # Show progress dialog
        self.progress_dialog = QProgressDialog("Processing...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Progress")
        self.progress_dialog.setWindowModality(Qt.ApplicationModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.show()

        # Background thread
        self.worker = WorkerThread()
        self.worker.progress.connect(self.progress_dialog.setValue)
        self.worker.finished.connect(self.operation_done)
        self.worker.start()

    def operation_done(self):
        self.setEnabled(True)
        self.progress_dialog.close()


if __name__ == "__main__":
    import sys
    from PyQt5.QtCore import Qt

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
