import traceback
from PySide6.QtCore import QObject, Signal, QRunnable, Slot, QThreadPool
import logging

log = logging.getLogger(__name__)

class TaskManager(QObject):
    # Define signals for the rest of the app to listen to
    status_updated = Signal(str)
    progress_updated = Signal(int)
    batch_finished = Signal()
    batch_cancelled = Signal()
    error_occurred = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self.batch_queue = []  # list of (fn, args, kwargs)
        self.cancelled = False
        self._batch_running = False
        self.active_workers = set()
        log.info("Task Manager initialized.")


    def queue_task(self, fn, *args, on_result=None, **kwargs):
        """Add a job at the end of the queue."""
        log.info(f"Queueing task: {fn.__name__}")
        was_idle = not self._batch_running
        self.batch_queue.append((fn, args, kwargs, on_result))
        # if we weren’t already running, go ahead and start
        if was_idle:
            self._batch_running = True
            self.cancelled = False
            #self.btn_cancel.setEnabled(True)
            self._run_next()

    def cancel_batch(self):
        # request cancellation; will stop after current task finishes
        log.warning("Batch cancellation requested.")
        self.cancelled = True
        self.batch_cancelled()
        #self.btn_cancel.setEnabled(False)
        #self.status.setText('Canceling…')

    def _run_next(self):
        # if user hit cancel, or no more jobs
        if self.cancelled or not self.batch_queue:
            # if "error" in self.status.text().lower():
            #     self.status.setText(
            #         f"Cancelled ~ {self.status.text()}" if self.cancelled else f"Ready ~ {self.status.text()}")
            # else:
            #     self.status.setText("Cancelled" if self.cancelled else "Ready")
            #self.progress.setValue(0)
            #self.btn_cancel.setEnabled(False)
            log.info("Batch finished or cancelled.")
            self._batch_running = False
            self.batch_finished.emit()
            return

        fn, args, kwargs, on_result = self.batch_queue.pop(0)
        worker = Worker(fn, *args, **kwargs)

        # connect signals
        #worker.signals.started.connect(lambda: self.status.setText("Starting…"))
        worker.signals.message.connect(self.status_updated)
        worker.signals.progress.connect(self.progress_updated)
        worker.signals.error.connect(self._on_error)

        # connect result to either custom or generic handler
        if on_result:
            worker.signals.result.connect(on_result)
        else:
            worker.signals.result.connect(self._on_result)

        # Connect the finished signal to our new cleanup slot
        worker.signals.finished.connect(lambda w=worker: self._on_worker_finished(w))
        # when this one finishes, run the next in queue
        worker.signals.finished.connect(self._run_next)

        self.active_workers.add(worker)
        log.info(f"Starting worker for {fn.__name__}. Active workers: {len(self.active_workers)}")

        self.pool.start(worker)

    @Slot() # New slot for cleanup
    def _on_worker_finished(self, finished_worker):
        """
        Find which worker sent the 'finished' signal and remove it from our active set.
        """
        # self.sender() returns the QObject that emitted the signal (the WorkerSignals object)
        # Its parent() is the Worker instance itself.
        #finished_worker = self.sender().parent()
        self.active_workers.discard(finished_worker)
        log.info(f"Worker for {finished_worker.fn.__name__} finished. Active workers: {len(self.active_workers)}")


    def _on_error(self, err_tb):
        exc, tb_str = err_tb
        logging.error(f"A worker task failed!\n{tb_str}")
        self.error_occurred.emit((exc, tb_str))
        self.status_updated.emit(f"Error: {exc}")

    def _on_result(self, result):
        # handle generic results if you like
        pass



class Worker(QRunnable):
    """
    QRunnable wrapper that:
      • calls any fn(signals, *args, **kwargs)
      • catches & emits exceptions
      • emits started / finished / result / error
    """
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn      = fn
        self.args    = args
        self.kwargs  = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        self.signals.started.emit()
        try:
            # pass the signals object first,
            # so your function can do:     signals.progress.emit(...)
            result = self.fn(self.signals, *self.args, **self.kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit((e, tb))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

class WorkerSignals(QObject):
    started  = Signal()
    finished = Signal()
    error    = Signal(tuple)           # (exc, traceback_str)
    result   = Signal(object)          # what your function returns
    progress = Signal(int)             # 0–100
    message  = Signal(str)             # any status text
