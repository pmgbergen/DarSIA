"""Rig identity/UI boundaries, local TOML and fake panels; no geometry work."""

import copy
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .test_native_sidebar import native_window, run_window, _select_run


def test_start_physical_choice_is_persisted_as_identity_only(qtbot, tmp_path):
    from darsia.fluidflower.series_wizard import NewSeriesFolderWizard
    from darsia.fluidflower.series_creation import _series_toml

    wizard = NewSeriesFolderWizard(experiment_folder=tmp_path)
    qtbot.addWidget(wizard)
    page = wizard.series_page
    page.series_id.setText("test-series")
    page.name.setText("Test series")
    page.rig_class.setCurrentIndex(page.rig_class.findData("medium"))
    page.physical_rig.setCurrentIndex(page.physical_rig.findData("FF Bilbo"))
    draft = wizard.draft()
    assert draft.rig_profile == "FF Bilbo"
    manifest = tomllib.loads(_series_toml(draft, "2026-10-16T00:00:00Z"))
    assert manifest["rig_profile"] == "FF Bilbo"
    assert "rig" not in manifest
    assert "width" not in manifest and "height" not in manifest
    # A previous hidden Medium choice must not leak into another rig class.
    page.rig_class.setCurrentIndex(page.rig_class.findData("large"))
    assert wizard.draft().rig_profile == ""
    assert "rig_profile" not in tomllib.loads(
        _series_toml(wizard.draft(), "2026-10-16T00:00:00Z")
    )


def test_header_follows_current_series_and_is_hidden_on_start_without_writes(
    run_window,
):
    context = run_window
    window = context.window
    manifest = context.roots["series-a"] / "series.toml"
    before = manifest.read_bytes()
    manifest.write_bytes(b'rig_profile = "FF AC"\n' + before)
    original = {path: path.read_bytes() for path in context.paths}
    _select_run(window, "series-a", "first")
    window._select_row(("preprocessing", "protocols"))
    # The rig line is a detail of the one header line (N141).
    assert not window.rig_label.isHidden()
    assert "Albus" in window.rig_label.text()
    assert "Albus" in window.status_header.line.toolTip()
    assert "chosen at Start" in window.rig_label.text()
    assert window.geometry_workspace.rig_class.isHidden()
    assert not window.geometry_workspace.rig_class.isEnabled()
    window._select_series("series-b")
    assert "Albus" not in window.rig_label.text()
    assert "physical rig not recorded" in window.rig_label.text()
    window._select_row(("start", "series"))
    assert window.rig_label.isHidden()
    for path, content in original.items():
        assert path.read_bytes() == content


def test_setup_hides_rig_fields_but_full_configuration_editor_stays_intact(run_window):
    window = run_window.window
    _select_run(window, "series-a", "first")
    window._select_row(("setup", "all"))
    assert not any(key.startswith("rig.") for key in window.settings_inputs)
    rig = copy.deepcopy(window.config_dict["rig"])
    window._on_open_full_config()
    assert any(key.startswith("rig.") for key in window.settings_inputs)
    assert window.config_dict["rig"] == rig


def test_open_foreign_config_never_supplies_the_selected_series_rig_header(
    run_window, tmp_path
):
    context = run_window
    window = context.window
    _select_run(window, "series-a", "first")
    window._select_row(("preprocessing", "protocols"))
    registered_path = window.run_combo.currentData()
    foreign = tmp_path / "unregistered-foreign.toml"
    foreign.write_text("[rig]\nwidth=9.87\nheight=6.54\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in context.paths}
    assert window._load_run_config(str(foreign))
    # The original load callback records the currently selected series even for
    # File > Open. Matching that field alone is therefore insufficient proof.
    assert window._config_series == "series-a"
    assert window.run_combo.currentData() == registered_path
    assert window.config_file == str(foreign)
    assert window.rig_context.display.width != 9.87
    assert window.rig_context.display.height != 6.54
    assert "common configuration" in window.rig_context.display.note
    for path, content in before.items():
        assert path.read_bytes() == content


def test_native_wizard_hides_repeated_choices_without_mutating_existing_rig(
    run_window, monkeypatch
):
    from darsia.gui.ui.preprocessing_wizard import PreprocessingWizardDialog, STEPS

    window = run_window.window
    _select_run(window, "series-a", "first")
    window._select_row(("preprocessing", "protocols"))
    rig = copy.deepcopy(window.config_dict["rig"])
    seen = []

    def inspect(dialog):
        seen.append(dialog)
        for step in ("series", "rig"):
            index = next(i for i, entry in enumerate(STEPS) if entry[0] == step)
            dialog._goto(index)
            scroll, _layout = dialog._page_containers[index]
            assert not scroll.findChildren(QComboBox)
            assert not any(key.startswith("rig.") for key in window.settings_inputs)
        for step in ("rig", "series", "rig"):
            index = next(i for i, entry in enumerate(STEPS) if entry[0] == step)
            dialog._goto(index)
            assert not any(key.startswith("rig.") for key in window.settings_inputs)
        assert window.config_dict["rig"] == rig
        depth = next(i for i, entry in enumerate(STEPS) if entry[0] == "depth")
        dialog._goto(depth)
        assert "depth.measurements" in window.settings_inputs
        dialog.close()
        return 0

    monkeypatch.setattr(PreprocessingWizardDialog, "exec", inspect)
    window.open_wizard_for("preprocessing")
    assert len(seen) == 1 and seen[0]._native_rig_readonly
    assert window.config_dict["rig"] == rig
    # The upstream ordinary wizard class and another instance remain unchanged.
    ordinary = PreprocessingWizardDialog(window)
    index = next(i for i, entry in enumerate(STEPS) if entry[0] == "rig")
    ordinary._goto(index)
    assert "rig.width" in window.settings_inputs
    assert not getattr(ordinary, "_native_rig_readonly", False)
    ordinary.close()


class FakePanel(QWidget):
    def __init__(self, *, width=None, height=None):
        super().__init__()
        self._loading = False
        self.frame = None
        self.reference_data = SimpleNamespace(width_m=width, height_m=height)
        self.frame_status = QLabel("No frame. Enter the rig's inner width and height.")
        layout = QVBoxLayout(self)
        self.size_row = QHBoxLayout()
        self.width_m, self.height_m = QDoubleSpinBox(), QDoubleSpinBox()
        for field in (self.width_m, self.height_m):
            field.setDecimals(5)
        self.size_row.addWidget(QLabel("Inner width"))
        self.size_row.addWidget(self.width_m)
        self.size_row.addWidget(QLabel("Inner height"))
        self.size_row.addWidget(self.height_m)
        layout.addLayout(self.size_row)
        self.class_row = QWidget()
        self.use_class_size = QPushButton("Use class size", self.class_row)
        layout.addWidget(self.class_row)
        layout.addWidget(self.frame_status)
        self.hint_calls = 0
        self.frame_calls = 0

    def reference(self):
        return self.reference_data

    def _update_class_hint(self):
        self.hint_calls += 1
        self.class_row.show()
        self.use_class_size.setEnabled(True)

    def _show_frame(self):
        self.frame_calls += 1
        self.width_m.setValue(getattr(self.frame, "width_m", None) or 0)
        self.height_m.setValue(getattr(self.frame, "height_m", None) or 0)
        self._update_class_hint()


class FakeWorkspace(QWidget):
    def __init__(self):
        super().__init__()
        self.rig_class = QComboBox()
        self.label = QLabel("Rig class")
        layout = QHBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.rig_class)
        self.simple_view = None
        self.rig_panel = None
        self.page = None

    def _load_rig_class(self):
        self.rig_class.show()

    def _show_page(self):
        self.simple_view = SimpleNamespace(rig_panel=FakePanel(width=1.25, height=0.65))

    def _install_rig_frame_panel(self, page):
        self.rig_panel = FakePanel(width=1.25, height=0.65)

    def _create_advanced_page(self):
        self.page = QWidget()
        layout = QVBoxLayout(self.page)
        self.page.width_m = QDoubleSpinBox()
        self.page.height_m = QDoubleSpinBox()
        self.page.width_m.setValue(1.15)
        self.page.height_m.setValue(0.75)
        self.page.grouping_mode = QComboBox()
        self.page.grouping_mode.addItem("Existing material grouping")
        for field in (
            self.page.width_m,
            self.page.height_m,
            self.page.grouping_mode,
        ):
            layout.addWidget(field)


def test_lazy_panels_keep_existing_sizes_and_cannot_reopen_rig_questions(qtbot):
    from darsia.fluidflower.rig_display import NativeRigContext

    workspace = FakeWorkspace()
    qtbot.addWidget(workspace)
    factory = SimpleNamespace(_render_settings_tabs=lambda values: None)
    window = SimpleNamespace(geometry_workspace=workspace, settings_factory=factory)
    context = NativeRigContext(window)
    workspace._show_page()
    panel = workspace.simple_view.rig_panel
    qtbot.addWidget(panel)
    # Missing frame size comes only from the same reference's existing values.
    assert panel.width_m.value() == 1.25 and panel.height_m.value() == 0.65
    assert panel.frame is None, "Presentation must not save or create a frame"
    assert panel.width_m.isHidden() and not panel.width_m.isEnabled()
    assert panel.class_row.isHidden() and not panel.use_class_size.isEnabled()
    panel._show_frame()
    assert panel.frame_calls == 1
    assert panel.width_m.value() == 1.25 and panel.height_m.value() == 0.65
    assert panel.class_row.isHidden()
    panel.frame = SimpleNamespace(width_m=1.10, height_m=0.60)
    panel._show_frame()
    assert panel.width_m.value() == 1.10 and panel.height_m.value() == 0.60
    workspace._install_rig_frame_panel(None)
    qtbot.addWidget(workspace.rig_panel)
    workspace.rig_panel._update_class_hint()
    assert workspace.rig_panel.class_row.isHidden()
    workspace._load_rig_class()
    assert workspace.rig_class.isHidden()
    assert workspace.label.isHidden()


def test_lazy_advanced_dimensions_are_readonly_without_changes_to_other_inputs(qtbot):
    from darsia.fluidflower.rig_display import NativeRigContext

    workspace = FakeWorkspace()
    qtbot.addWidget(workspace)
    window = SimpleNamespace(
        geometry_workspace=workspace,
        settings_factory=SimpleNamespace(_render_settings_tabs=lambda values: None),
    )
    NativeRigContext(window)
    assert workspace.page is None
    workspace._create_advanced_page()
    page = workspace.page
    qtbot.addWidget(page)
    assert page.width_m.value() == 1.15 and page.height_m.value() == 0.75
    assert not page.width_m.isEnabled() and not page.height_m.isEnabled()
    assert page.grouping_mode.isEnabled()
    # The original reload/prefill can still present existing saved values;
    # locking the display must not block that or write any model/configuration.
    page.width_m.setValue(1.10)
    page.height_m.setValue(0.70)
    assert page.width_m.value() == 1.10 and page.height_m.value() == 0.70
    assert not page.width_m.isEnabled() and not page.height_m.isEnabled()
    assert page.grouping_mode.isEnabled()
