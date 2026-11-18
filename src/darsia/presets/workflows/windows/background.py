"""Calculations of an interactive window in a worker thread.

The window's thread only draws; a calculation runs in one worker thread, so the window
stays responsive. A canvas timer polls the calculation and then calls the drawing step in
the window's thread (matplotlib is not thread safe). Rule: the calculation changes only
the data of the window (state), never the figure. One calculation runs at a time; actions
of the user wait (are refused with a message) while it runs.

Without an event loop (e.g. the Agg backend in scripts and checks), calculations run
directly, so the results do not depend on timing.

"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

BUSY_MESSAGE = "Busy - wait for the calculation to finish."
"""Shown when the user acts during a calculation."""


class BackgroundRunner:
    """Run the calculations of a window in a worker thread.

    Parameters
    ----------
    fig : Figure
        The window.
    show_busy : Callable
        Shows (or clears, with "") the message about a running calculation; called as
        `show_busy(message, now)`, with `now` True when it must be drawn right away.
    show_refusal : Callable
        Shows (with `BUSY_MESSAGE`) or clears (with None) the message for actions
        refused during a calculation.
    refusal_shown : Callable
        Whether the refusal message is shown (it is cleared after the calculation).

    """

    def __init__(
        self,
        fig: Figure,
        show_busy: Callable[[str, bool], None],
        show_refusal: Callable[[str | None], None],
        refusal_shown: Callable[[], bool],
    ) -> None:
        self.fig = fig
        self.show_busy = show_busy
        self.show_refusal = show_refusal
        self.refusal_shown = refusal_shown
        self.busy = False
        """Whether a calculation runs in the background."""
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        self._done: Callable[[], None] | None = None
        self._message = ""
        self._progress: tuple[int, int] | None = None
        self._timer = None

    def in_background(self) -> bool:
        """Whether calculations run in a worker thread (windows with an event loop)."""
        return type(self.fig.canvas).required_interactive_framework is not None

    def run(
        self, message: str, work: Callable[..., None], done: Callable[[], None]
    ) -> None:
        """Run a calculation, then update the window.

        Parameters
        ----------
        message : str
            Busy message (without the trailing dots).
        work : Callable
            The calculation; called with a progress callback `progress(number, total)`
            (or None); changes only the state.
        done : Callable
            Updates the window after the calculation (window's thread).

        """
        if not self.in_background():
            self.show_busy(f"{message} ...", True)
            work(None)
            done()
            self.show_busy("", True)
            return
        self.busy = True
        self._message, self._progress, self._done = message, None, done
        self.show_busy(f"{message} ...", False)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="calculation"
            )
        self._future = self._executor.submit(work, self._set_progress)
        self._timer = self.fig.canvas.new_timer(interval=50)
        self._timer.add_callback(self._poll)
        self._timer.start()

    def _set_progress(self, number: int, total: int) -> None:
        """Progress of the calculation (called in the worker thread)."""
        self._progress = (number, total)

    def _poll(self) -> None:
        """Show the progress; finish the calculation when it is done."""
        if not self._future.done():
            progress = self._progress
            if progress is not None:
                number, total = progress
                self.show_busy(f"{self._message}: image {number}/{total} ...", False)
            return
        self._timer.stop()
        self._finish()

    def _finish(self, draw: bool = True) -> None:
        """End of a calculation: report errors, then update the window."""
        future, done = self._future, self._done
        self._future = self._done = self._timer = None
        self.busy = False
        if self.refusal_shown():
            self.show_refusal(None)
        try:
            future.result()
        except Exception:
            logger.exception("The calculation failed.")
            self.show_busy("Calculation failed, see log.", False)
            return
        if draw:
            done()
            self.show_busy("", False)

    def wait(self) -> None:
        """Wait for a running calculation (without updating the window)."""
        if self._future is None:
            return
        self._timer.stop()
        try:
            self._future.result()
        except Exception:
            pass
        self._finish(draw=False)

    def refuse(self) -> bool:
        """Whether an action must wait, because a calculation runs (shows a message)."""
        if self.busy:
            self.show_refusal(BUSY_MESSAGE)
        return self.busy

    def shutdown(self) -> None:
        """Stop the worker thread (after the window is closed)."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
