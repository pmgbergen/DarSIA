"""The settings panel never rewrites a configuration file with losses.

Synthetic files only: a fake window with the panel's save and workflow start
reproduces the order of the real ones (the runner saves first, then starts),
and the real native window is checked to install the guard.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import settings_save_guard as guard
from .test_native_sidebar import native_window  # noqa: F401 - the shared real-window fixture

#: A run configuration with entries the panel cannot reproduce: an integer
#: combo value, facies label tables and the times of image-selection groups.
ORIGINAL = """\
[rig]
dim = 2
[facies.3]
labels = [3, 4]
[[data_time]]
folder = "a"
times = ["1h", "2h"]
tol = "5min"
[[data_time]]
folder = "b"
times = ["3h"]
tol = "5min"
"""


def _lossy(config, edits=()):
    """The panel's copy of ORIGINAL: dim as text, no facies labels, no times, defaults added."""
    copy = dict(rig=dict(dim="2"), data_time=[dict(folder="a"), dict(folder="b")],
                depth=dict(constant_depth=0.012))
    for key, value in edits:
        copy["rig"][key] = value
    return copy


class FakeFactory:
    """The two settings-panel methods the guard wraps, as the real panel has them."""

    def __init__(self, window, panel):
        self.window, self.panel, self.writes = window, panel, 0

    def _sync_settings_inputs_to_config_dict(self):
        self.window.config_dict = self.panel()

    def save_settings(self):
        import toml

        self._sync_settings_inputs_to_config_dict()
        Path(self.window.config_file).write_text(toml.dumps(self.window.config_dict), encoding="utf-8")
        self.writes += 1


class FakeRunner:
    """The workflow start: it saves the settings, then starts the process."""

    def __init__(self, window):
        self.window, self.started = window, 0

    def start_workflow_process(self, argv):
        self.window.settings_factory.save_settings()
        self.started += 1
        return "process"


def _window(tmp_path, panel, text=ORIGINAL):
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")
    window = SimpleNamespace(config_file=str(path), config_dict={}, logs=[], warnings=[])
    window.print_log = window.logs.append
    window.settings_factory = FakeFactory(window, panel)
    window.process_runner = FakeRunner(window)
    guard.guard_lossless_settings_save(window, warn=lambda _window, text: window.warnings.append(text))
    return window, path


def test_flatten_walks_tables_and_lists_of_tables_and_keeps_empty_tables():
    leaves = guard.flatten_settings(dict(a=dict(b=1, c=[1, 2]), d=[dict(e=1), dict(f=2)], g={}))
    assert leaves == {"a.b": 1, "a.c": [1, 2], "d[0].e": 1, "d[1].f": 2, "g": {}}


def test_same_value_across_types():
    assert guard._same_value(2, "2") and guard._same_value(2, 2.0) and guard._same_value(True, "true")
    assert not guard._same_value(2, "3") and not guard._same_value(2, "two")


def test_the_panels_copy_of_a_rich_file_is_lossy_without_changes():
    import tomllib

    changes = guard.compare_settings(tomllib.loads(ORIGINAL), _lossy(None))
    assert changes.lossy and not changes.changed
    assert set(changes.removed) == {"facies.3.labels", "data_time[0].times", "data_time[0].tol",
                                    "data_time[1].times", "data_time[1].tol"}
    assert changes.retyped == ("rig.dim",) and changes.added == ("depth.constant_depth",)
    assert "drop 5 entries" in changes.summary() and "change the type of 1" in changes.summary()


def test_an_edit_counts_as_changed_and_a_lossless_save_is_not_lossy():
    before = dict(rig=dict(dim=2, width=0.9))
    edited = guard.compare_settings(before, dict(rig=dict(dim="3", width=0.9)))
    assert edited.changed == ("rig.dim",) and edited.retyped == ("rig.dim",) and edited.lossy
    plain = guard.compare_settings(before, dict(rig=dict(dim=2, width=1.0), extra=dict(x=1)))
    assert not plain.lossy and plain.changed == ("rig.width",) and plain.added == ("extra.x",)


def test_a_lossless_save_writes_as_before(tmp_path):
    import tomllib

    window, path = _window(tmp_path, lambda: dict(rig=dict(dim=3)), text="[rig]\ndim = 2\n")
    window.settings_factory.save_settings()
    assert window.settings_factory.writes == 1 and tomllib.loads(path.read_text())["rig"]["dim"] == 3


def test_a_lossy_save_without_changes_leaves_the_file_unchanged(tmp_path):
    window, path = _window(tmp_path, lambda: _lossy(None))
    assert window.settings_factory.save_settings() is None
    assert path.read_text(encoding="utf-8") == ORIGINAL and window.settings_factory.writes == 0
    assert "not rewritten" in window.logs[-1] and window.warnings == []


def test_a_lossy_save_with_changes_is_refused_and_shown(tmp_path):
    window, path = _window(tmp_path, lambda: _lossy(None, edits=[("dim", "3")]))
    assert window.settings_factory.save_settings() is None
    assert path.read_text(encoding="utf-8") == ORIGINAL and window.settings_factory.writes == 0
    assert window.logs[-1].startswith("Not saved:") and window.warnings == [window.logs[-1]]


def test_a_workflow_runs_on_the_unchanged_file_when_nothing_changed(tmp_path):
    window, path = _window(tmp_path, lambda: _lossy(None))
    assert window.process_runner.start_workflow_process(["setup"]) == "process"
    assert window.process_runner.started == 1 and window.settings_factory.writes == 0
    assert path.read_text(encoding="utf-8") == ORIGINAL
    # The skip covers only the save inside that start.
    window.settings_factory.save_settings()
    assert window.settings_factory.writes == 0 and "not rewritten" in window.logs[-1]


def test_a_workflow_does_not_start_when_changes_would_be_lost(tmp_path):
    window, path = _window(tmp_path, lambda: _lossy(None, edits=[("dim", "3")]))
    assert window.process_runner.start_workflow_process(["setup"]) is None
    assert window.process_runner.started == 0 and window.settings_factory.writes == 0
    assert window.logs[-1].startswith("Run cancelled:") and window.warnings == [window.logs[-1]]
    assert path.read_text(encoding="utf-8") == ORIGINAL


@pytest.mark.parametrize("text", [None, "not = [valid toml"])
def test_without_a_readable_file_the_save_behaves_as_before(tmp_path, text):
    window, path = _window(tmp_path, lambda: dict(rig=dict(dim=3)), text=text or ORIGINAL)
    if text is None:
        path.unlink()
    window.settings_factory.save_settings()
    assert window.settings_factory.writes == 1


def test_the_pending_changes_follow_the_panels_own_sync(tmp_path):
    window, _ = _window(tmp_path, lambda: _lossy(None))
    changes = guard.pending_settings_changes(window)
    assert changes.lossy and window.config_dict == _lossy(None)
    window.config_file = ""
    assert guard.pending_settings_changes(window) is None


def test_a_setting_added_in_a_lossy_panel_is_never_silently_left_out(tmp_path):
    # The panel would retype rig.dim and the person added a value the file lacks.
    added = lambda: dict(_lossy(None), image_porosity=dict(active=True))  # noqa: E731
    window, path = _window(tmp_path, added)
    changes = guard.pending_settings_changes(window)
    assert changes.lossy and not changes.changed and "image_porosity.active" in changes.added
    window._settings_save_guard["edited"] = True  # what the input watcher records
    assert window.process_runner.start_workflow_process(["setup"]) is None
    assert window.process_runner.started == 0 and window.settings_factory.writes == 0
    assert window.logs[-1].startswith("Run cancelled:") and "changed or added" in window.logs[-1]
    assert path.read_text(encoding="utf-8") == ORIGINAL


def test_loading_a_file_clears_the_edit_mark(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(ORIGINAL, encoding="utf-8")
    loads = []
    window = SimpleNamespace(config_file=str(path), config_dict={}, logs=[],
                             config_controller=SimpleNamespace(load_config=lambda: loads.append(1)))
    window.print_log = window.logs.append
    window.settings_factory = FakeFactory(window, lambda: _lossy(None))
    window.process_runner = FakeRunner(window)
    guard.guard_lossless_settings_save(window, warn=lambda *_: None)
    window._settings_save_guard["edited"] = True
    window.config_controller.load_config()
    assert loads == [1] and window._settings_save_guard["edited"] is False
    assert window.process_runner.start_workflow_process(["setup"]) == "process"


def test_a_written_save_clears_the_edit_mark(tmp_path):
    window, _ = _window(tmp_path, lambda: dict(rig=dict(dim=3)), text="[rig]\ndim = 2\n")
    window._settings_save_guard["edited"] = True
    window.settings_factory.save_settings()
    assert window.settings_factory.writes == 1 and window._settings_save_guard["edited"] is False


def test_only_inputs_inside_the_panel_count_as_user_edits(qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

    container, outside = QWidget(), QPushButton("Run")
    layout = QVBoxLayout(container)
    qtbot.addWidget(container)
    qtbot.addWidget(outside)
    built = {}

    def render(_sections=None):
        """The panel's render: it replaces its widgets with new ones."""
        while layout.count():
            layout.takeAt(0).widget().deleteLater()
        built.update(line=QLineEdit(), box=QCheckBox("active"), label=QLabel("A label"))
        for widget in built.values():
            layout.addWidget(widget)

    render()
    assert guard.is_settings_input(built["line"], container) and guard.is_settings_input(built["box"], container)
    assert not guard.is_settings_input(built["label"], container)
    assert not guard.is_settings_input(outside, container)
    state = {"edited": False}
    factory = SimpleNamespace(_render_settings_tabs=render)
    guard._watch_user_input(SimpleNamespace(settings_container=container, settings_factory=factory), state)
    container.show()
    qtbot.mouseClick(built["label"], Qt.MouseButton.LeftButton)
    qtbot.mouseClick(outside, Qt.MouseButton.LeftButton)
    assert state["edited"] is False
    qtbot.keyClick(built["line"], "a")
    assert state["edited"] is True
    # A new render gets new inputs, and they are watched too.
    state["edited"] = False
    factory._render_settings_tabs()
    qtbot.mouseClick(built["box"], Qt.MouseButton.LeftButton)
    assert state["edited"] is True


def test_the_native_window_installs_the_guard(native_window):  # noqa: F811
    # The fixture replaces the workflow start with a failing stub, so only the
    # save chain is visible here; both wrappers are installed by the same call.
    names, save = [], native_window.settings_factory.save_settings
    while save is not None:
        names.append(save.__qualname__)
        save = getattr(save, "__wrapped__", None)
    assert any("guard_lossless_settings_save" in name for name in names)
    mine = next(i for i, name in enumerate(names) if "guard_lossless_settings_save" in name)
    assert any("guard_config_file" in name for name in names[mine:])
