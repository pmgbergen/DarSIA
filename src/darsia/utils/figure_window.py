"""Utilities for the windows of interactive matplotlib figures."""

import logging

import matplotlib
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


def ask_yes_no(
    title: str,
    question: str,
    default: bool = True,
    yes: str | None = None,
    no: str | None = None,
) -> bool | None:
    """Ask a yes/no question in a dialog (Qt backends only).

    Parameters
    ----------
    title : str
        Title of the dialog.
    question : str
        Text of the question.
    default : bool
        Preselected answer.
    yes, no : str, optional
        Texts of the two buttons instead of "Yes" and "No", e.g. "Save" and "Discard".

    Returns
    -------
    bool | None
        The answer, or None if no dialog could be shown (e.g. non-interactive backend).

    """
    if "qt" not in str(matplotlib.get_backend()).lower():
        # No dialog without a Qt window (e.g. Agg in scripts): it would block or fail.
        return None
    try:
        from matplotlib.backends.qt_compat import QtWidgets

        if QtWidgets.QApplication.instance() is None:
            # Allow dialogs before the first figure: create the application like the
            # matplotlib Qt backend does (same configuration, reused by it later).
            from matplotlib.backends import backend_qt

            backend_qt._create_qApp()
        buttons = QtWidgets.QMessageBox.StandardButton
        box = QtWidgets.QMessageBox()
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        box.setText(question)
        box.setStandardButtons(buttons.Yes | buttons.No)
        box.setDefaultButton(buttons.Yes if default else buttons.No)
        if yes is not None:
            box.button(buttons.Yes).setText(yes)
        if no is not None:
            box.button(buttons.No).setText(no)
        return box.exec() == buttons.Yes
    except Exception:
        logger.debug("Could not show the dialog.")
        return None
