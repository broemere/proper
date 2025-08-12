import traceback
import uuid
from PySide6.QtCore import QObject, Signal, QRunnable, Slot, QThreadPool
import logging

log = logging.getLogger(__name__)


class WorkerSignals(QObject):
    result = Signal(str, object)
    error = Signal(str, tuple)
    progress = Signal(int)
    message = Signal(str)


class Worker(QRunnable):
    def __init__(self, task_id, fn, *args, **kwargs):
        super().__init__()
        self.task_id = task_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(self.signals, *self.args, **self.kwargs)
            self.signals.result.emit(self.task_id, result)
        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit(self.task_id, (e, tb))


class TaskManager(QObject):
    status_updated = Signal(str)
    progress_updated = Signal(int)
    batch_finished = Signal()
    batch_cancelled = Signal()
    error_occurred = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self.task_queue = []
        self.task_callbacks = {}
        self.cancelled = False
        # --- FINAL FIX: The robust state flag to prevent race conditions ---
        self.is_running = False

        log.info("Task Manager initialized.")

    def queue_task(self, fn, *args, on_result=None, **kwargs):
        log.info(f"Queueing task: {fn.__name__}")
        task = (fn, args, kwargs, on_result)
        self.task_queue.append(task)

        # --- FINAL FIX: Check our own flag, not the unreliable pool count ---
        if not self.is_running:
            # Set the flag immediately to "lock" the queue
            self.is_running = True
            self.cancelled = False
            self._run_next()

    def cancel_batch(self):
        if self.is_running:
            log.warning("Batch cancellation requested. Will stop after current task.")
            self.cancelled = True
            self.batch_cancelled.emit()

    def _run_next(self):
        if self.cancelled:
            log.warning("Batch execution halted due to cancellation.")
            self.task_queue.clear()
            self.is_running = False  # Unlock the queue
            return

        if not self.task_queue:
            log.info("Task queue is empty. Batch finished.")
            # --- FINAL FIX: Unlock the queue only when it's truly empty ---
            self.is_running = False
            self.batch_finished.emit()
            return

        fn, args, kwargs, on_result = self.task_queue.pop(0)

        task_id = str(uuid.uuid4())
        if on_result:
            self.task_callbacks[task_id] = on_result

        worker = Worker(task_id, fn, *args, **kwargs)

        worker.signals.message.connect(self.status_updated)
        worker.signals.progress.connect(self.progress_updated)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self._on_error)

        log.info(f"Starting worker for {fn.__name__} (Task ID: {task_id})")
        self.pool.start(worker)

    @Slot(str, object)
    def _on_result(self, task_id, result):
        log.info(f"Task {task_id} finished successfully.")

        callback = self.task_callbacks.pop(task_id, None)
        if callback:
            try:
                callback(result)
            except Exception as e:
                log.error(f"Error in 'on_result' callback for task {task_id}: {e}", exc_info=True)

        self._run_next()

    @Slot(str, tuple)
    def _on_error(self, task_id, err_tb):
        exc, tb_str = err_tb
        log.error(f"Worker task {task_id} failed!\n{tb_str}")

        self.error_occurred.emit(err_tb)
        self.status_updated.emit(f"Error: {exc}")

        self.task_callbacks.pop(task_id, None)

        self._run_next()