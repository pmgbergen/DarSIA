"""Read off the GUI thread; answer at once when the reading is quick.

A reading that finishes within a short wait is returned directly, as the same
read on the GUI thread would be. A slower one, such as a check of many files
on a series share, keeps running and answers later through ``answered``, so
the window never waits for it.

A reading may also name itself as a module function with picklable arguments
(``call``). With the helper process on (helper_process.current), it then runs
in that process, so its work never holds the window's interpreter; otherwise,
or when the helper cannot answer, it runs on a thread of the window.
"""
from __future__ import annotations

import threading
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import copy_context
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, Signal, Slot

# The longest the GUI thread waits for a reading before it answers later.
QUICK_ANSWER_S = 0.2


class BackgroundRead(QObject):
    """The latest reading of one owner; a superseded reading's answer is dropped."""

    answered = Signal(object, object, str)
    _late = Signal(int, object, object, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.generation = 0
        # Key of the reading still running, None when no reading runs.
        self.key = None
        self._late.connect(self._deliver, Qt.ConnectionType.QueuedConnection)

    def busy(self) -> bool:
        """True while a reading runs that will answer through ``answered``."""
        return self.key is not None

    def read(self, key: Any, function: Callable[[], Any], *, call: tuple[str, tuple] | None = None,
             wait_s: float | None = None) -> tuple[bool, Any, str]:
        """Start ``function()`` and wait at most ``wait_s`` (default QUICK_ANSWER_S).

        Returns (True, value, error) when it answered within the wait, else
        (False, None, ""): ``answered`` then emits (key, value, error) unless
        another reading started meanwhile. ``error`` names a raised exception.
        ``call`` is the same reading as ``("package.module:function", args)``,
        run in the helper process when it is on.
        """
        self.generation += 1
        generation = self.generation
        self.key = key
        wait = QUICK_ANSWER_S if wait_s is None else wait_s
        if call is not None:
            from . import helper_process
            helper = helper_process.current()
            if helper is not None:
                try:
                    future = helper.submit(call[0], *call[1])
                except helper_process.HelperUnavailable:
                    future = None
                if future is not None:
                    return self._await(future, generation, key, function, wait)
        return self._on_thread(generation, key, function, wait)

    def _on_thread(self, generation, key, function, wait) -> tuple[bool, Any, str]:
        lock = threading.Lock()
        state = dict(done=False, late=False, value=None, error="")
        context = copy_context()

        def run():
            value, error = None, ""
            try:
                value = function()
            except Exception as failure:  # noqa: BLE001 - the owner shows every failure
                error = f"{type(failure).__name__}: {failure}"
            with lock:
                state.update(done=True, value=value, error=error)
                late = state["late"]
            if late:
                self._emit_late(generation, key, value, error)

        thread = threading.Thread(target=lambda: context.run(run), name="darsia-background-read", daemon=True)
        thread.start()
        thread.join(wait)
        with lock:
            if state["done"]:
                if generation == self.generation:
                    self.key = None
                return True, state["value"], state["error"]
            state["late"] = True
        return False, None, ""

    def _await(self, future, generation, key, function, wait) -> tuple[bool, Any, str]:
        """Wait for the helper's answer; it answers later, or a thread reads when the helper ended."""
        from .helper_process import HelperUnavailable

        try:
            value = future.result(timeout=wait)
        except FutureTimeout:
            future.add_done_callback(lambda done: self._helper_answered(done, generation, key, function))
            return False, None, ""
        except HelperUnavailable:
            return self._on_thread(generation, key, function, wait)
        except RuntimeError as error:
            self.key = None
            return True, None, str(error)
        self.key = None
        return True, value, ""

    def _helper_answered(self, done, generation, key, function) -> None:
        from .helper_process import HelperUnavailable

        try:
            value, error = done.result(), ""
        except HelperUnavailable:
            # The helper ended before it answered: read on a thread instead.
            finished, value, error = self._on_thread(generation, key, function, 0.0)
            if not finished:
                return  # The thread answers when it is done.
        except Exception as failure:  # noqa: BLE001 - the function's own failure, as text
            value, error = None, str(failure)
        self._emit_late(generation, key, value, error)

    def _emit_late(self, generation, key, value, error) -> None:
        try:
            self._late.emit(generation, key, value, error)
        except RuntimeError:
            pass  # Its owner has closed; nothing is shown.

    def cancel(self) -> None:
        """Drop the answer of the reading that runs; the reading itself finishes unseen."""
        self.generation += 1
        self.key = None

    @Slot(int, object, object, str)
    def _deliver(self, generation, key, value, error) -> None:
        if generation != self.generation:
            return
        self.key = None
        self.answered.emit(key, value, error)
