"""Mass analysis > Image selection shows the saved production grid and opens its editor."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QComboBox, QGroupBox, QStackedWidget

from darsia.fluidflower import workflow_row_status as status
from darsia.fluidflower.image_selection_ui import NativeImageSelectionWorkflow
from darsia.fluidflower.production_image_grid import FORMAT
from darsia.fluidflower.production_image_selection_ui import ProductionImageSelectionPage

SERIES = 'schema_version = 1\nseries_id = "demo"\n[runs]\nids = ["r1", "r2"]\n'


def _save_grid(root, *, counts=None):
    grid = dict(format=FORMAT, created_utc="2026-10-27T18:30:00+00:00", created_by="reviewer",
                parameters=dict(main_count=80, first_h=0.25, through_h=48.0, late_count=10),
                runs={run: dict(counts=dict(baselines=1, main=80, late=late, selected=81 + late, missing=0))
                      for run, late in (counts or dict(r1=10, r2=4)).items()})
    raw = json.dumps(grid).encode()
    path = root / "image_selection" / "production" / "grid.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    (root / "series.toml").write_text(
        SERIES + '[production_image_selection.selection_manifest]\npath = "image_selection/production/grid.json"\n'
        f'sha256 = "{hashlib.sha256(raw).hexdigest()}"\nsize_bytes = {len(raw)}\n', encoding="utf-8")
    return path


@pytest.fixture
def window(tmp_path, qtbot):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text(SERIES, encoding="utf-8")
    stack = QStackedWidget()
    qtbot.addWidget(stack)
    opened = []
    return SimpleNamespace(root=root, _method_series_root=lambda: root, content_stack=stack, opened=opened,
                           image_selection_workflow=SimpleNamespace(open=lambda **kw: opened.append(kw)))


def test_without_a_saved_grid_the_page_says_mass_analysis_needs_one(window):
    page = ProductionImageSelectionPage(window)
    window.content_stack.addWidget(page.panel)
    page.show()
    assert window.content_stack.currentWidget() is page.panel
    assert page.status.text() == ("No production image selection is saved yet. "
                                  "Mass analysis needs one; use Choose production images.")
    assert page.table.rowCount() == 0 and page.edit_button.isEnabled()


def test_a_saved_grid_shows_its_counts_per_run_in_run_order(window):
    path = _save_grid(window.root, counts=dict(r2=4, r1=10))
    page = ProductionImageSelectionPage(window)
    page.refresh()
    text = page.status.text()
    assert text.startswith("Saved production grid: 80 images from 0.25 h through 48 h after injection, "
                           "up to 10 later images, plus each run's baseline (saved 2026-10-27 18:30 by reviewer UTC).")
    assert text.endswith(str(path))
    rows = [[page.table.item(row, column).text() for column in range(page.table.columnCount())]
            for row in range(page.table.rowCount())]
    assert rows == [["R1", "1", "80", "10", "91", "0"], ["R2", "1", "80", "4", "85", "0"]]


def test_a_grid_changed_after_saving_is_reported_not_shown(window):
    path = _save_grid(window.root)
    path.write_text("{}", encoding="utf-8")
    page = ProductionImageSelectionPage(window)
    page.refresh()
    assert page.status.text().startswith("The saved production grid could not be read: The saved production grid changed")
    assert page.table.rowCount() == 0


def test_choose_opens_the_existing_editor_at_the_production_grid(window, qtbot):
    page = ProductionImageSelectionPage(window)
    page.edit_button.click()
    assert window.opened == [dict(production=True)]
    group = QGroupBox()
    group.setCheckable(True)
    group.setChecked(False)
    qtbot.addWidget(group)
    shown = []
    dialog = SimpleNamespace(production_controls=SimpleNamespace(group=group), run_combo=QComboBox(),
                             saved=SimpleNamespace(connect=lambda _slot: None),
                             exec=lambda: shown.append(group.isChecked()))
    qtbot.addWidget(dialog.run_combo)
    host = SimpleNamespace(_method_series_root=lambda: window.root, run_combo=QComboBox())
    qtbot.addWidget(host.run_combo)
    editor = NativeImageSelectionWorkflow(host, dialog_factory=lambda *_a, **_k: dialog,
                                          stored_plan_provider=lambda _metadata: None)
    editor.open(False)
    editor.open(production=True)
    assert shown == [False, True]


def test_the_row_is_saved_only_with_a_bound_grid(window):
    context = status.WorkflowStatusContext(window.root, manifest=dict(schema_version=1, series_id="demo"))
    missing = status.row_status(context, "image_selection", "production")
    assert missing.state == "missing" and missing.summary == "No production image selection is saved."
    _save_grid(window.root)
    import tomllib
    manifest = tomllib.loads((window.root / "series.toml").read_text(encoding="utf-8"))
    saved = status.row_status(status.WorkflowStatusContext(window.root, manifest=manifest), "image_selection", "production")
    assert saved.state != "missing" and saved.summary == "A production image selection is saved."
