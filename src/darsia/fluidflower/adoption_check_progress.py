"""Optional, operation-local observations of the existing geometry Check."""
from contextlib import contextmanager
from contextvars import ContextVar


_OBSERVER = ContextVar("adoption-check-progress", default=None)


@contextmanager
def observe_check_progress(callback, *, strict=False):
    """Restore this operation's observer; strict callbacks may stop their worker."""
    token = _OBSERVER.set((callback, strict))
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def report_check_progress(message):
    """Report an entered stage; only opted-in strict callbacks may interrupt."""
    observer = _OBSERVER.get()
    if observer is None:
        return
    callback, strict = observer
    if callback is not None:
        if strict:
            callback(message)
            return
        try:
            callback(message)
        except Exception:
            # A diagnostic observer must never admit or refuse a candidate.
            pass
