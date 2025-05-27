from PyQt5.QtCore import Qt, QThread, pyqtSignal

class ProgressWorker(QThread):
    progress_changed = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, function):
        super().__init__()
        self.setObjectName("ProgressWorker")
        self.function = function

    def run(self):
        while True:
            try:
                progress = self.function()
                print("Progress:", progress)
                self.progress_changed.emit(progress)
                if progress >= 100:
                    self.finished.emit()
                    break
                self.msleep(500)  # sleep 500ms
            except Exception as e:
                print("Error:", e)
                break

class AsyncOperationWorker(QThread):
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, function, progress_func, *args, **kwargs):
        super().__init__()
        self.setObjectName(function.__name__)
        self.function = function
        self.progress_worker = ProgressWorker(progress_func)
        self.progress_changed = self.progress_worker.progress_changed
        self.result = None
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.progress_worker.start()
            self.result = self.function(*self.args, **self.kwargs)
            self.result_ready.emit(self.result)
        except Exception as e:
            self.error_occurred.emit(str(e))
