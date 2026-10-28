"""Keep the settings panel from rewriting a run configuration with losses.

DarSIA's settings panel writes the whole configuration it shows back to the
opened TOML file, after every save and before every workflow it starts (Build
run model, Full setup, the protocol steps). That copy is not lossless for every
file: a combo box hands back its text, so an integer comes back as a string, and
settings the panel cannot show (for example facies label tables or the times
and tolerances of an image-selection group) are left out. A file that the panel
cannot reproduce therefore loses entries on the first save, even when nothing
was changed.

:func:`guard_lossless_settings_save` compares, before any write, the file that
would be written with the file on disk:

- If the write keeps every existing entry with its type, the save goes ahead
  as before.
- If the write would remove or retype entries but changes no value, nothing is
  written, and a workflow runs on the unchanged file.
- If it would remove or retype entries and also changes values, nothing is
  written, the reason is shown, and a workflow does not start, so it never runs
  with settings other than the ones shown.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class SettingsChanges:
    """How the entries of a configuration file would change on a save.

    Each field holds dotted entry paths (list items as ``name[index]``):
    ``removed`` would disappear, ``retyped`` would keep their value but get
    another type (``2`` written as ``"2"``), ``changed`` would get another
    value and ``added`` would be new.
    """

    removed: tuple[str, ...] = ()
    retyped: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    added: tuple[str, ...] = ()

    @property
    def lossy(self) -> bool:
        """Whether the save would remove an entry or change an entry's type."""
        return bool(self.removed or self.retyped)

    def summary(self) -> str:
        """A short English description of the losses, with the first entries."""
        parts = []
        if self.removed:
            parts.append(f"drop {len(self.removed)} entr{'y' if len(self.removed) == 1 else 'ies'}")
        if self.retyped:
            parts.append(f"change the type of {len(self.retyped)}")
        first = ", ".join((self.removed + self.retyped)[:3])
        return " and ".join(parts) + (f" (first: {first})" if first else "")


def flatten_settings(value: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of a parsed TOML document by its dotted path.

    Tables are walked by key and lists of tables by index; any other list is
    one leaf. An empty table is a leaf too, so that removing it counts.
    """
    if isinstance(value, dict):
        if not value and prefix:
            return {prefix: {}}
        leaves: dict[str, Any] = {}
        for key, item in value.items():
            leaves.update(flatten_settings(item, f"{prefix}.{key}" if prefix else str(key)))
        return leaves
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(flatten_settings(item, f"{prefix}[{index}]"))
        return leaves
    return {prefix: value}


def _same_value(before: Any, after: Any) -> bool:
    """Whether two leaves of different types still mean the same value.

    A combo box hands back ``"2"`` for ``2``, and a number may come back as
    ``2.0``; both are the same value in another type.
    """
    if isinstance(before, bool) or isinstance(after, bool):
        return str(before).lower() == str(after).lower()
    if isinstance(before, (int, float)) or isinstance(after, (int, float)):
        try:
            return float(before) == float(after)
        except (TypeError, ValueError):
            return False
    return str(before) == str(after)


def compare_settings(before: dict, after: dict) -> SettingsChanges:
    """The changes from the file on disk (``before``) to the file a save writes."""
    old, new = flatten_settings(before), flatten_settings(after)
    removed = tuple(path for path in old if path not in new)
    added = tuple(path for path in new if path not in old)
    retyped, changed = [], []
    for path in old:
        if path not in new:
            continue
        first, second = old[path], new[path]
        if type(first) is type(second):
            if first != second:
                changed.append(path)
        elif _same_value(first, second):
            retyped.append(path)
        else:
            changed.append(path)
            retyped.append(path)
    return SettingsChanges(removed=removed, retyped=tuple(retyped), changed=tuple(changed), added=added)


def _parse_pair(text: str, written: str) -> tuple[dict, dict] | None:
    """Parse the file on disk and the file a save would write with one parser."""
    import tomllib

    try:
        return tomllib.loads(text), tomllib.loads(written)
    except tomllib.TOMLDecodeError:
        pass
    try:
        import toml

        return toml.loads(text), toml.loads(written)
    except Exception:  # noqa: BLE001 - an unreadable pair keeps the old behaviour
        return None


def pending_settings_changes(window) -> SettingsChanges | None:
    """The changes a settings save of ``window`` would make to its opened file.

    Returns None when there is no readable file to compare with, in which case
    the save behaves as before.
    """
    path = getattr(window, "config_file", "")
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    window.settings_factory._sync_settings_inputs_to_config_dict()
    try:
        import toml

        written = toml.dumps(window.config_dict)
    except Exception:  # noqa: BLE001 - the original save reports its own error
        return None
    parsed = _parse_pair(text, written)
    if parsed is None:
        return None
    return compare_settings(*parsed)


def _warn(window, text: str) -> None:
    """Show a refusal in a message box as well as in the log."""
    from PySide6.QtWidgets import QMessageBox

    QMessageBox.warning(window, "Settings not saved", text)


#: The widget kinds through which a person changes a setting in the panel.
_INPUT_CLASS_NAMES = ("QLineEdit", "QComboBox", "QAbstractSpinBox", "QAbstractButton",
                      "QAbstractItemView", "QTextEdit", "QPlainTextEdit")


def is_settings_input(widget, container) -> bool:
    """Whether ``widget`` is, or lies inside, an input of the settings panel ``container``.

    Inputs are line edits, combo boxes, spin boxes, buttons (check boxes and the
    add/remove row buttons), item views and text edits. Labels, tab bars, scroll
    bars and empty space are not inputs.
    """
    from PySide6 import QtWidgets

    if widget is None or container is None or not container.isAncestorOf(widget):
        return False
    inputs = tuple(getattr(QtWidgets, name) for name in _INPUT_CLASS_NAMES)
    current = widget
    while current is not None and current is not container:
        if isinstance(current, QtWidgets.QScrollBar):
            return False
        if isinstance(current, inputs):
            return True
        current = current.parentWidget()
    return False


def _watch_user_input(window, state) -> None:
    """Mark the panel as edited on any key, click or wheel on one of its inputs.

    One event filter sits on every input inside ``window.settings_container``.
    It is attached again after each render of the panel
    (``settings_factory._render_settings_tabs``), so only the panel's own
    inputs are watched. Switching tabs, scrolling the page and the workflow
    buttons outside the panel never mark it. A click without a change counts
    too: the guard then refuses rather than guesses. Values the panel fills in
    by itself send no input events.
    """
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QWidget

    factory = window.settings_factory
    container = getattr(window, "settings_container", None)
    render = getattr(factory, "_render_settings_tabs", None)
    if container is None:
        return
    kinds = {QEvent.Type.KeyPress, QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick,
             QEvent.Type.Wheel}

    class _UserInput(QObject):
        def eventFilter(self, obj, event):  # noqa: N802 - Qt's name
            if event.type() in kinds:
                state["edited"] = True
            return False

    watcher = _UserInput(container)

    def attach() -> None:
        for widget in container.findChildren(QWidget):
            if is_settings_input(widget, container):
                widget.installEventFilter(watcher)

    if callable(render):
        def _render_settings_tabs(*args, **kwargs):
            """Render as before, then watch the new inputs."""
            result = render(*args, **kwargs)
            attach()
            return result

        _render_settings_tabs.__wrapped__ = render
        factory._render_settings_tabs = _render_settings_tabs
    attach()
    state["watcher"] = watcher
    state["attach"] = attach


def guard_lossless_settings_save(window, *, warn: Callable[[Any, str], None] | None = None) -> None:
    """Refuse every settings save that would remove or retype file entries.

    Wraps ``window.settings_factory.save_settings``,
    ``window.process_runner.start_workflow_process`` and, when present,
    ``window.config_controller.load_config``. The panel counts as edited after
    any input by the user since the file was loaded (see _watch_user_input),
    or when a value would differ from the file. A workflow first checks the
    pending save. If it is lossy and the panel is not edited, the save made
    inside the workflow start is skipped and the workflow runs on the unchanged
    file. If it is lossy and the panel is edited, the workflow does not start,
    so a setting added or changed in the panel is never silently left out.
    Saves that keep every entry and its type behave as before.
    """
    factory = window.settings_factory
    runner = window.process_runner
    save = factory.save_settings
    start = runner.start_workflow_process
    state = {"skip_next_save": False, "edited": False}
    window._settings_save_guard = state
    notify = warn or _warn

    def _name() -> str:
        return Path(window.config_file).name

    def _edited(changes: SettingsChanges) -> bool:
        return state["edited"] or bool(changes.changed)

    def _refusal(changes: SettingsChanges) -> str:
        return (f"Writing {_name()} from this panel would also {changes.summary()}. "
                f"Settings shown here were changed or added ({len(changes.changed)} value(s) differ "
                f"from the file), and they are not saved. Edit {_name()} directly, then reload it. "
                "If you changed nothing, select the run again to reload its file.")

    def save_settings(*args, **kwargs):
        """Save as before, unless the write would remove or retype entries."""
        if state["skip_next_save"]:
            state["skip_next_save"] = False
            return None
        changes = pending_settings_changes(window)
        if changes is None or not changes.lossy:
            result = save(*args, **kwargs)
            state["edited"] = False
            return result
        if not _edited(changes):
            window.print_log(f"{_name()}: no setting changed, so the file is not rewritten. "
                             f"Rewriting it here would {changes.summary()}.")
            return None
        text = "Not saved: " + _refusal(changes)
        window.print_log(text)
        notify(window, text)
        return None

    def start_workflow_process(argv, *args, **kwargs):
        """Start as before, without a lossy save first; refuse when edits would be lost."""
        changes = pending_settings_changes(window)
        if changes is not None and changes.lossy:
            if _edited(changes):
                text = "Run cancelled: " + _refusal(changes)
                window.print_log(text)
                notify(window, text)
                return None
            state["skip_next_save"] = True
            window.print_log(f"{_name()}: no setting changed, so it is not rewritten before this run. "
                             f"Rewriting it here would {changes.summary()}.")
        try:
            return start(argv, *args, **kwargs)
        finally:
            state["skip_next_save"] = False

    controller = getattr(window, "config_controller", None)
    load = getattr(controller, "load_config", None)
    if callable(load):
        def load_config(*args, **kwargs):
            """Load as before; a freshly loaded file has no edits."""
            result = load(*args, **kwargs)
            state["edited"] = False
            return result

        load_config.__wrapped__ = load
        controller.load_config = load_config
    _watch_user_input(window, state)
    save_settings.__wrapped__ = save
    start_workflow_process.__wrapped__ = start
    factory.save_settings = save_settings
    runner.start_workflow_process = start_workflow_process


__all__ = ["SettingsChanges", "flatten_settings", "compare_settings", "pending_settings_changes",
           "is_settings_input", "guard_lossless_settings_save"]
