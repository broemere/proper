from PySide6.QtCore import QObject, Signal


# Centralized signal bus
class _ErrorBus(QObject):
    user_error_details = Signal(object, str)  # (Exception, traceback string)

bus = _ErrorBus()


# Short helper for user-level errors
def user_error(message: str, hint: str) -> None:
    e = Exception(message)
    setattr(e, "hint", hint)
    bus.user_error_details.emit(e, "")


# Helper for critical developer errors
def dev_error(exc: Exception) -> None:
    """
    Emit a general, unexpected error (includes traceback).

    Usage:
        try:
            foobar
        except Exception as e:
            dev_error(e)
        return None
    """
    tb_str = traceback.format_exc()
    bus.user_error_details.emit(exc, tb_str)
