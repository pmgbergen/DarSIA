"""Utilities for the windows of interactive matplotlib figures."""

import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def maximize_figure_window(fig: plt.Figure) -> None:
    """Open the window of a figure maximized (best effort, depends on the backend).

    Parameters
    ----------
    fig : plt.Figure
        Figure whose window is maximized; call right after creating the figure.

    """
    try:
        window = fig.canvas.manager.window
    except AttributeError:
        return  # non-interactive backend
    try:
        # Qt
        from matplotlib.backends.qt_compat import QtCore

        window.setWindowState(
            window.windowState() | QtCore.Qt.WindowState.WindowMaximized
        )
        return
    except Exception:
        pass
    try:
        window.state("zoomed")  # Tk (Windows)
    except Exception:
        try:
            window.Maximize(True)  # wx
        except Exception:
            logger.debug("Could not maximize the figure window.")
