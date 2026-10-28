"""Setup > Overflow data: every run's overflow source, file and status, read only.

Synthetic series and Studio projects in tmp_path, written by the project's own
writers. Reading and showing the page writes nothing and opens no network path.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from PySide6.QtCore import QCoreApplication, QSettings, Qt
from PySide6.QtWidgets import QAbstractItemView, QScrollArea

from darsia.fluidflower import overflow_data_page as module
from darsia.fluidflower.overflow_data_page import (
    OVERFLOW_DATA_ROW,
    UNKNOWN,
    OverflowCell,
    OverflowData,
    OverflowDataPage,
    OverflowRun,
    read_overflow_data,
    studio_project_root,
)


@pytest.fixture(autouse=True)
def no_storage_aliases(monkeypatch, tmp_path):
    # A missing registry file means no storage aliases on any machine.
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "no-registry.toml"))


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _snapshot(*roots):
    return {
        str(path): path.read_bytes()
        for root in roots
        for path in Path(root).rglob("*")
        if path.is_file()
    }


def _series(root, runs, *, pointer=None, extra=""):
    ids = ", ".join(f'"{run}"' for run in runs)
    legacy = f'[legacy]\nstudio_project = "{pointer}"\n' if pointer else ""
    put(
        root / "series.toml",
        'schema_version = 1\nseries_id = "fixture"\nname = "Fixture"\n'
        f"[runs]\nids = [{ids}]\n{legacy}{extra}",
    )
    return root


def _studio(tmp_path, *, series_id="fixture", save=True):
    """A Studio project: r1 reads a table, r2 scale images, r3 and r9 none."""
    from darsia.fluidflower.overflow import (
        OverflowRevisionStore,
        apply_registered_overflow_mode,
        normalize_overflow_config,
        scan_overflow_run,
    )
    from darsia.fluidflower.project import FluidFlowerProject

    sources = tmp_path / "sources"
    table = put(sources / "r1" / "r1_overflow.csv", "elapsed_h,reading\n0.0,1.0\n0.5,2.0\n")
    images = sources / "r2" / "scale"
    images.mkdir(parents=True)
    for minute in (0, 1):
        Image.new("RGB", (4, 4)).save(images / f"scale_2024-01-02_03-0{minute}-00.png")
    project = FluidFlowerProject.create(
        tmp_path / "studio",
        name="Fixture",
        series_id=series_id,
        template_id="custom",
        rig_type="custom",
    )
    project.set_runs(
        [
            {"run_id": "r1", "source_mapping": {"run_id": "r1", "overflow_table": str(table),
                                                "overflow_tables": [str(table)]}},
            {"run_id": "r2", "source_mapping": {"run_id": "r2",
                                                "overflow_image_folders": [str(images)]}},
            {"run_id": "r3", "source_mapping": {"run_id": "r3"}},
            {"run_id": "r9", "source_mapping": {"run_id": "r9"}},
        ]
    )
    project.set_module_status("data_integrity", "approved")
    store = OverflowRevisionStore(project)
    config = normalize_overflow_config(project, store.current_config())
    config, _ = apply_registered_overflow_mode(config, mode="table", run_ids=["r1"])
    config, _ = apply_registered_overflow_mode(config, mode="scale_images", run_ids=["r2"])
    for row in config["runs"]:
        row["scan"] = scan_overflow_run(row)
    if save:
        store.save_revision(config, actor="", reason="Synthetic sources")
    else:
        store.save_draft(config)
    review = project.root / "results/overflow/scale_review"
    records = [
        {"status": "Verified", "time_from_injection_h": 0.5},
        {"status": "Corrected", "time_from_injection_h": 1.0},
        {"status": "Partial", "time_from_injection_h": 1.5},
        {"status": "Pending"},
    ]
    put(
        review / "r2/r2_overflow_scale_review.json",
        json.dumps({"metadata": {"updated_at": "2024-01-03T10:00:00"}, "records": records}),
    )
    put(review / "r3/r3_overflow_scale_review.json", "{not json")
    return project


def test_reader_follows_the_series_pointer_to_the_saved_revision(tmp_path):
    project = _studio(tmp_path)
    root = _series(tmp_path / "series", ["r1", "r2", "r3", "r4"], pointer="../studio/series.json")
    before = _snapshot(tmp_path)
    data = read_overflow_data(root)
    assert _snapshot(tmp_path) == before
    rows = {run.run: run for run in data.runs}
    assert list(rows) == ["r1", "r2", "r3", "r4"]

    table = rows["r1"]
    assert (table.source.text, table.file.text) == ("Table", "r1_overflow.csv")
    assert table.file.tip == "Recorded: " + str((tmp_path / "sources/r1/r1_overflow.csv").resolve())
    assert table.status.text == "Ready · 2 rows · 1 warning · step review required"
    assert "Warning: No standard weight column was recognized" in table.status.tip
    assert "Assignment: Global policy" in table.source.tip

    images = rows["r2"]
    assert (images.source.text, images.file.text) == ("Scale images", "scale")
    assert images.status.text == (
        "Ready · 2 images · 3 scale images reviewed · step review required"
    )
    for line in ("Exact readings (verified or corrected): 2", "Uncertain readings: 1",
                 "Unreadable images: 0", "Reviewed times: 0.50 to 1.50 h"):
        assert line in images.status.tip

    none = rows["r3"]
    assert (none.source.text, none.file.text) == ("None", UNKNOWN)
    assert none.status.text == "Not used · scale review unreadable · step review required"
    assert "Scale-reader review could not be read" in none.status.tip

    missing = rows["r4"]
    assert {missing.source.text, missing.file.text, missing.status.text} == {UNKNOWN}
    assert "has no decision for this run" in missing.status.tip

    revision = json.loads((project.root / "overrides/overflow/current.json").read_text("utf-8"))
    assert data.summary.startswith(
        f"From the Studio project's saved overflow revision {revision['revision_id'][:8]} ("
    )
    assert data.summary.endswith(
        "1 run with a table, 1 run with scale images, 1 run with none, 1 run with no decision."
    )
    assert "Overflow mode chosen at setup: per run" in data.detail
    assert "Overflow measurements step: in progress" in data.detail
    assert "does not list: r9." in data.detail
    assert "writes nothing" in data.detail


def test_series_without_a_studio_pointer_shows_every_run_unknown(tmp_path):
    root = _series(tmp_path / "series", ["r1", "r2"])
    data = read_overflow_data(root)
    assert [run.run for run in data.runs] == ["r1", "r2"]
    for run in data.runs:
        for cell in (run.source, run.file, run.status):
            assert cell.text == UNKNOWN
            assert "[legacy] studio_project" in cell.tip and "has no overflow entry" in cell.tip
    assert data.summary == data.runs[0].status.tip
    _series(root, ["r1"], extra='[overflow]\nmode = "table"\n')
    assert "in a form this page does not read" in read_overflow_data(root).summary


def test_unreadable_or_foreign_studio_project_gives_the_reason(tmp_path):
    root = _series(tmp_path / "series", ["r1"], pointer="../missing/series.json")
    data = read_overflow_data(root)
    assert data.runs[0].source.text == UNKNOWN
    assert data.summary.startswith("The Studio project that series.toml names could not be read")
    _studio(tmp_path, series_id="other")
    _series(root, ["r1"], pointer="../studio/series.json")
    data = read_overflow_data(root)
    assert "belongs to another series" in data.runs[0].status.tip


def test_studio_project_without_a_saved_revision_selects_no_source(tmp_path):
    _studio(tmp_path, save=False)
    root = _series(tmp_path / "series", ["r1", "r2"], pointer="../studio")
    data = read_overflow_data(root)
    assert {cell.text for run in data.runs for cell in (run.source, run.file, run.status)} == {
        UNKNOWN
    }
    assert "no saved overflow revision" in data.summary
    assert "unsaved working draft" in data.detail and "Record:" not in data.detail


def test_no_series_or_an_unreadable_manifest_shows_no_rows(tmp_path):
    assert read_overflow_data(None).summary.startswith("Choose a series on Start")
    data = read_overflow_data(tmp_path / "absent")
    assert data.runs == () and data.summary.startswith("The series manifest could not be read")


def test_studio_pointer_names_the_project_file_or_folder_only(tmp_path):
    root = tmp_path / "series"
    assert studio_project_root(root, {}) is None
    assert studio_project_root(root, {"legacy": "not a table"}) is None
    assert studio_project_root(root, {"legacy": {"studio_project": "  "}}) is None
    expected = (tmp_path / "studio").resolve()
    for pointer in ("../studio/series.json", "../studio", str(tmp_path / "studio/series.json")):
        assert studio_project_root(root, {"legacy": {"studio_project": pointer}}) == expected


def test_cells_read_times_paths_counts_and_scans_as_recorded(tmp_path):
    assert module._utc_text("2026-09-04T06:36:49.412415+00:00") == "2026-09-04 06:36 UTC"
    assert module._utc_text("2026-09-04T08:36:49+02:00") == "2026-09-04 06:36 UTC"
    assert module._utc_text("2026-09-04T06:36:49", date_only=True) == "2026-09-04"
    assert module._utc_text("not a time") == "not a time"
    assert module._file_name("\\\\share\\data\\run\\a_overflow.xlsx") == "a_overflow.xlsx"
    assert module._file_name("/data/run/scale/") == "scale"
    assert module._plural(1, "run") == "1 run" and module._plural(1749, "row") == "1,749 rows"
    assert module._int(True) is None and module._int(3) == 3 and module._int("3") is None
    assert module._strings("abc") == [] and module._strings(["a", " ", 2]) == ["a", "2"]
    assert module._run_key("Run 1") == "run-1"
    moved = module._file_cell(
        "table", {"table_path": "//old/share/r1.csv"}, lambda path: path.replace("//old/share", "D:/data")
    )
    assert moved.text == "r1.csv" and moved.tip.endswith("\nOn this machine: D:/data/r1.csv")
    folders = module._file_cell("scale_images", {"image_folders": ["/a/one", "/a/two"]}, str)
    assert folders.text == "one (+1 more)" and folders.tip == "Recorded folders:\n/a/one\n/a/two"
    assert module._file_cell("table", {}, str).text == UNKNOWN
    unscanned = module._status_cell("table", {"scan": {}}, {}, {}, "overflow revision x")
    assert unscanned.text == "Not scanned"
    failed = module._status_cell(
        "table", {"scan": {"status": "error", "errors": ["Table is unavailable"]}}, {}, {}, "rev"
    )
    assert failed.text == "Error" and "Error: Table is unavailable" in failed.tip
    assert failed.tip.endswith("Recorded in rev.")
    assert module._review_parts({"available": False}) == ([], [])
    unknown = module._unknown_run("r1", "Why")
    assert unknown.source == unknown.file == unknown.status == OverflowCell(UNKNOWN, "Why")
    project = SimpleNamespace(manifest={}, root=tmp_path)
    saved = {"revision_id": "abcdef0123", "configuration": {"runs": [{"run_id": "r1", "mode": "odd"}]}}
    rows, counts, extra = module._revision_rows(["r1", "r2"], project, saved)
    assert "unsupported source 'odd'" in rows[0].status.tip
    assert "overflow revision abcdef01 has no decision" in rows[1].status.tip
    assert counts["missing"] == 2 and extra == []


def test_page_shows_rows_read_only_in_one_table_without_scroll_areas(qtbot):
    page = OverflowDataPage()
    qtbot.addWidget(page)
    assert page.table.rowCount() == 0 and not page.reload_button.isEnabled()
    assert page.summary.text().startswith("Choose a series on Start")
    row = OverflowRun(
        "r1", OverflowCell("Table", "From a table."), OverflowCell("a.csv", "Recorded: /x/a.csv"),
        OverflowCell("Ready", "Scanned."),
    )
    page.set_data(OverflowData((row,), "Summary.", "Detail."))
    headers = [page.table.horizontalHeaderItem(column).text() for column in range(4)]
    assert headers == ["Run", "Source", "File", "Status"]
    assert [page.table.item(0, column).text() for column in range(4)] == ["r1", "Table", "a.csv", "Ready"]
    assert page.table.item(0, 2).toolTip() == "Recorded: /x/a.csv"
    assert not page.table.item(0, 3).flags() & Qt.ItemFlag.ItemIsEditable
    assert page.table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert (page.summary.text(), page.detail.text()) == ("Summary.", "Detail.")
    assert not page.findChildren(QScrollArea)
    assert page.table.maximumHeight() == 16777215
    assert "single series" in page.note.text()


def test_show_series_and_reload_read_the_records_again(qtbot, tmp_path):
    root = _series(tmp_path / "series", ["r1"])
    page = OverflowDataPage()
    qtbot.addWidget(page)
    data = page.show_series(root)
    assert page.series_root == root and page.reload_button.isEnabled()
    assert [run.run for run in data.runs] == ["r1"]
    assert page.table.item(0, 1).text() == UNKNOWN
    _series(root, ["r1", "r2"])
    qtbot.mouseClick(page.reload_button, Qt.MouseButton.LeftButton)
    assert page.table.rowCount() == 2 and page.data.runs[1].run == "r2"
    assert page.reload().runs == page.data.runs


@pytest.fixture
def overflow_window(qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower import native_darsia_app as native

    old = (
        QSettings.defaultFormat(),
        QCoreApplication.organizationName(),
        QCoreApplication.applicationName(),
    )
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    QCoreApplication.setOrganizationName("DarSIA-overflow-data-tests")
    QCoreApplication.setApplicationName(tmp_path.name)
    _studio(tmp_path)
    root = _series(tmp_path / "series", ["r1", "r2", "r3", "r4"], pointer="../studio/series.json")
    monkeypatch.setattr(native, "configured_series", lambda: (("fixture", "Fixture", root),))
    monkeypatch.setattr(native, "ProjectRegistry", lambda: SimpleNamespace(entries=lambda: []))
    monkeypatch.setattr(native.PhaseModelWorkspace, "ensure_studio", lambda self: self.studio_placeholder)
    monkeypatch.setattr(native.SeriesGeometryWorkspace, "ensure_page", lambda self: self)
    monkeypatch.setattr(
        native.NativeFluidFlowerMainWindow,
        "ensure_mass_analysis",
        lambda self: self.mass_analysis_placeholder,
    )
    window = native.NativeFluidFlowerMainWindow(initial_series="fixture")
    qtbot.addWidget(window)
    monkeypatch.setattr(
        window.process_runner,
        "start_workflow_process",
        lambda *_a, **_k: pytest.fail("Opening Overflow data must not start a workflow"),
    )
    window.show()
    qtbot.waitUntil(window.isVisible)
    window._open_series_from_start(str(root))
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    try:
        yield SimpleNamespace(window=window, data=[root, tmp_path / "studio", tmp_path / "sources"])
    finally:
        window.close()
        QCoreApplication.setOrganizationName(old[1])
        QCoreApplication.setApplicationName(old[2])
        QSettings.setDefaultFormat(old[0])


def test_setup_row_follows_protocols_and_shows_all_runs_read_only(overflow_window, qtbot):
    window = overflow_window.window
    before = _snapshot(*overflow_window.data)
    section = window.sidebar._sections["setup"]
    assert [row.label for row in section._rows.values()][:3] == [
        "Protocols",
        "Overflow data",
        "Image selection",
    ]
    row = section.get_row("overflow_data:runs")
    assert (row.action, row.checkbox_id) == OVERFLOW_DATA_ROW
    window._select_row(OVERFLOW_DATA_ROW)
    page = window.overflow_data_page
    assert window.sidebar.get_selected() == OVERFLOW_DATA_ROW
    assert window.content_stack.currentWidget() is page
    texts = [[page.table.item(index, column).text() for column in range(4)]
             for index in range(page.table.rowCount())]
    assert [values[:2] for values in texts] == [
        ["r1", "Table"], ["r2", "Scale images"], ["r3", "None"], ["r4", UNKNOWN]
    ]
    # A whole-series page: the header names the series and hides the run selector.
    assert window.status_header.line.text().endswith(" · Whole series (4 runs)")
    assert "Applies to: whole series" in row.help_text
    assert not window.run_combo.isVisible() and not window.run_combo.isEnabled()
    # No invented saved-output status, nothing to play; the user confirms the step.
    assert OVERFLOW_DATA_ROW not in window.workflow_status.rows
    assert window.workflow_status.panel.isVisible()
    assert window.workflow_status.label.text() == "Not confirmed"
    assert window.workflow_status.confirm_button.text() == "Confirm step"
    assert not window.workflow_status.open_button.isVisible()
    assert not window.toolbar_builder.play_action.isEnabled()
    qtbot.mouseClick(page.reload_button, Qt.MouseButton.LeftButton)
    assert page.table.rowCount() == 4
    assert _snapshot(*overflow_window.data) == before
    # Run pages keep the run selector.
    window._select_row(("setup", "depth"))
    assert window.run_combo.isVisible()
