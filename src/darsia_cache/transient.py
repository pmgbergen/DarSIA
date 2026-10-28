"""Shared transient filesystem retry behavior without scientific imports."""
import time

TRANSIENT_RETRY_DELAYS = (0.5, 2.0, 5.0)  # Seconds before each retry of one file step.
# Windows network-path errors that Python may report as a missing file or path.
_NETWORK_WINERRORS = frozenset({51, 53, 54, 55, 58, 59, 64, 67, 121, 1231, 1236})


def transient_os_error(error: BaseException) -> bool:
    """Whether a file error is a short SMB/NAS interruption worth waiting for.

    Protocol answers are never transient: an existing file (a lost exclusive
    create), a missing file or folder (moved, claimed or never written) and a
    wrong file type keep their meaning. A dropped network path is transient
    even when Python reports it as a missing file.
    """
    if not isinstance(error, OSError):
        return False
    if getattr(error, "winerror", None) in _NETWORK_WINERRORS:
        return True
    return not isinstance(error, (FileExistsError, FileNotFoundError,
                                  NotADirectoryError, IsADirectoryError))


def retry_transient(action, *, delays=None, wait=None, on_retry=None, retry_if=None):
    """Run action; wait and retry a transient file error, then raise it.

    A single SMB file or claim error must not crash the work: each transient
    failure waits for the next delay (default TRANSIENT_RETRY_DELAYS) and the
    action runs again. Other errors, and the last transient one, are raised
    unchanged. on_retry(attempt, delay, error) records a retry; wait(seconds)
    may return True to stop early, which raises the error; retry_if(error)
    can refuse a retry that is not safe at that point.
    """
    delays = TRANSIENT_RETRY_DELAYS if delays is None else delays
    wait = time.sleep if wait is None else wait
    for attempt, delay in enumerate((*delays, None), start=1):
        try:
            return action()
        except OSError as error:
            if (delay is None or not transient_os_error(error)
                    or (retry_if is not None and not retry_if(error))):
                raise
            if on_retry is not None:
                on_retry(attempt, delay, error)
            if wait(delay):
                raise
