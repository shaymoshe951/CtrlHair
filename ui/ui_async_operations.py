from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QProgressDialog
)
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
                # print("Progress:", progress)
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

    def __init__(self, function, progress_func, parent, *args, **kwargs):
        super().__init__()
        self.setObjectName(function.__name__)
        self.function = function
        self.progress_worker = ProgressWorker(progress_func)
        self.progress_changed = self.progress_worker.progress_changed
        self.result = None
        self.args = args
        self.kwargs = kwargs
        self.parent = parent

    def run(self):
        try:
            self.progress_worker.start()

            self.result = self.function(*self.args, **self.kwargs)
            self.result_ready.emit(self.result)

            # self.parent.setEnabled(True)
        except Exception as e:
            print(str(e))
            self.error_occurred.emit(str(e))

def run_progress_tmp(parent, worker):
    parent.setEnabled(False)

    progress_dialog = QProgressDialog("Processing...", "Cancel", 0, 100, parent=parent)
    progress_dialog.setWindowTitle("Progress")
    progress_dialog.setWindowModality(Qt.ApplicationModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.show()
    worker.progress_update.connect(progress_dialog.setValue)

    parent.progress_dialog = progress_dialog

class OperationWorker(QThread):
    result_ready = pyqtSignal(object)
    cancel_event = pyqtSignal()
    error_occurred = pyqtSignal(str)

    # OperationWorker(color_modification, (img1, img2, color_name), auto1111_if.get_progress, auto1111_if.cancell_process, evt_auto_proc_result, self)
    def __init__(self, run_function, run_function_params_tuple, progress_check_function, cancel_function, result_ready_func, parent):
        super().__init__()
        self.setObjectName(run_function.__name__)
        self.run_function = run_function
        self.run_params = run_function_params_tuple
        self.progress_check_function = progress_check_function
        self.cancel_function = cancel_function
        self.result_ready_func = result_ready_func
        self.parent = parent
        self.result = None

        self.parent.setEnabled(False)
        self.create_progress_bar()

        # Progress Agent
        self.agent_progress_status = PeriodicProgressCheckAgent(self.progress_check_function)
        self.agent_progress_status.progress_update.connect(self.parent.progress_dialog.setValue)
        self.parent.progress_dialog.setValue(0)

        # Cancel Agent
        self.agent_cancel_check = PeriodicCancelCheckAgent(self._check_cancel_status_function)
        self.agent_cancel_check.cancel_event.connect(self._cancelled)

        self.result_ready.connect(self._finished)
        self.result_ready.connect(self.result_ready_func)

    def execute(self):
        self.agent_progress_status.start()
        self.agent_cancel_check.start()
        self.start()


    def _check_cancel_status_function(self):
        if self.parent.progress_dialog.wasCanceled():
            return True
        return False

    def create_progress_bar(self):
        progress_dialog = QProgressDialog("Processing...", "Cancel", 0, 100, parent=self.parent)
        progress_dialog.setWindowTitle("Progress")
        progress_dialog.setWindowModality(Qt.ApplicationModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()
        self.parent.progress_dialog = progress_dialog

    def _finished(self, result):
        print("In _finished")
        self.parent.setEnabled(True)
        if hasattr(self, 'agent_progress_status'):
            self.agent_progress_status.keep_running = False
            del self.agent_progress_status
        if hasattr(self, 'agent_cancel_check'):
            self.agent_cancel_check.keep_running = False
            del self.agent_cancel_check
        if hasattr(self.parent, 'progress_dialog'):
            self.parent.progress_dialog.close()
        # delete agents
        # self.agent_progress_status.finished.disconnect()
        self.terminate()

    def _cancelled(self):
        print("In _cancelled")
        self.cancel_function()
        self._finished(None)

    def run(self):
        try:
            self.result = self.run_function(*self.run_params)
            self.result_ready.emit(self.result)
            print("Done")

            # self.parent.setEnabled(True)
        except Exception as e:
            print(str(e))
            self.error_occurred.emit(str(e))

class PeriodicProgressCheckAgent(QThread):
    progress_update = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, function):
        super().__init__()
        self.setObjectName("PeriodicProgressCheckAgent")
        self.function = function
        self.keep_running = True

    def run(self):
        while self.keep_running:
            try:
                progress = self.function()
                self.progress_update.emit(progress)
                if progress >= 100:
                    self.finished.emit()
                    break
                self.msleep(500)  # sleep 500ms
            except Exception as e:
                print("Error:", e)
                break

class PeriodicCancelCheckAgent(QThread):
    cancel_event = pyqtSignal()

    def __init__(self, function):
        super().__init__()
        self.setObjectName("PeriodicCancelCheckAgent")
        self.function = function
        self.keep_running = True

    def run(self):
        while self.keep_running:
            try:
                is_cancelled = self.function()
                if is_cancelled:
                    self.cancel_event.emit()
                    break
                self.msleep(500)  # sleep 500ms
            except Exception as e:
                print("Error:", e)
                break