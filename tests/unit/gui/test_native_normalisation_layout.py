from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QTabWidget

from darsia.fluidflower.normalisation_step import NormalisationStep

from .test_native_normalisation_step import evidence, series


def test_vertical_panels_preserve_loaded_evidence_and_reference(qtbot, evidence, series):
    source, _runs, provider = evidence
    root, original = series
    step = NormalisationStep(provider=provider)
    qtbot.addWidget(step)
    step.resize(1440, 1000)
    step.show()
    step.set_context(root, "SERIES_X")
    step.set_source(source)
    qtbot.waitUntil(lambda: step.can_close() and not step.metadata_busy())

    assert step.sections["method"].toggle.text().endswith("Image correction")
    assert step.preparation_note.isVisible()
    assert not step.history_panel.isVisible()
    assert not step.sections["evidence"].toggle.isChecked()
    assert not step.sections["reference"].toggle.isChecked()
    assert not step.sections["history"].toggle.isChecked()
    qtbot.mouseClick(step.sections["history"].toggle, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(step.sections["evidence"].toggle, Qt.MouseButton.LeftButton)
    assert step.history_panel.isVisible()
    assert not step.preparation_note.isVisible()
    assert not step.findChildren(QTabWidget)
    assert list(step.sections) == ["crop", "method", "history", "evidence", "reproduction", "reference"]
    for section in step.sections.values():
        assert step.workflow.controls_widget.isAncestorOf(section)
    assert step.workflow.workspace.isAncestorOf(step.table)
    assert step.workflow.workspace.isAncestorOf(step.imported_fields)
    assert step.table.mapTo(step, QPoint()).x() > step.sections["method"].mapTo(step, QPoint()).x()
    assert step.workflow.controls_scroll.horizontalScrollBar().maximum() == 0
    assert step.table.rowCount() == 4
    assert step.reference.currentData() == "zz1"
    for label in (step.source_label, step.ranking_note, step.provenance):
        qtbot.waitUntil(lambda: label.height() >= label.heightForWidth(label.width()))
    qtbot.waitUntil(lambda: step.table.y() >= step.ranking_note.geometry().bottom())
    qtbot.waitUntil(lambda: step.choose_source.y() >= step.source_label.geometry().bottom())

    section = step.sections["reference"]
    section.set_expanded(False)
    assert not step.reference.isVisible()
    section.set_expanded(True)
    assert step.reference.currentData() == "zz1"
    qtbot.mouseClick(step.save_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: step.can_close() and not step.metadata_busy())
    assert step.ready
    assert (root / "series.toml").read_bytes() != original
    assert step.table.rowCount() == 4


def test_missing_history_does_not_hide_or_block_current_choice(qtbot, tmp_path):
    from darsia.fluidflower.image_normalisation import OFF

    root = tmp_path / "series"
    root.mkdir()
    manifest = root / "series.toml"
    original = b'series_id="example"\n[runs]\nids=["one","two"]\n'
    manifest.write_bytes(original)
    step = NormalisationStep()
    qtbot.addWidget(step)
    step.resize(1200, 800)
    step.show()
    step.set_context(root, "example")
    qtbot.waitUntil(lambda: step.can_close() and not step.metadata_busy())
    assert not step.ready  # Historical admission remains separate.
    assert step.method_choice.method.isEnabled()
    step.method_choice.method.setCurrentIndex(step.method_choice.method.findData(OFF))
    assert step.method_choice.save_button.isEnabled()
    assert step.preparation_note.isVisible()
    assert not step.status.isVisible()
    assert not step.table.isVisible()
    assert manifest.read_bytes() == original


def test_reproduction_buttons_keep_original_callbacks_and_values(qtbot, tmp_path):
    step = NormalisationStep()
    qtbot.addWidget(step)
    step.resize(1440, 1000)
    step.show()
    replay = step.reproduction
    step.sections["history"].set_expanded(True)
    step.sections["reproduction"].set_expanded(True)
    calls = []
    request = {"rows": [{"run": "ac28"}]}

    def prepare(root, run, frame):
        calls.append(("prepare", root, run, frame))
        return request

    def execute(actual_request, output, *, progress, check_cancelled):
        check_cancelled()
        calls.append(("execute", actual_request, output))
        return tmp_path / "result"

    replay.preparer, replay.executor = prepare, execute
    replay.set_context(tmp_path, "FF_AC")
    from .test_normalisation_replay_step import saved_raw_document
    replay.install_snapshot(["ac28"], document=saved_raw_document())
    replay.frame.setValue(3)
    replay.output.setText(str(tmp_path / "separate output"))
    qtbot.mouseClick(replay.prepare, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: step.can_close() and not step.metadata_busy())
    assert replay.start.isEnabled()
    step.sections["reproduction"].set_expanded(False)
    step.sections["reproduction"].set_expanded(True)
    assert replay.frame.value() == 3
    qtbot.mouseClick(replay.start, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: step.can_close() and not step.metadata_busy())
    assert calls == [
        ("prepare", tmp_path, "ac28", 3),
        ("execute", request, str(tmp_path / "separate output")),
    ]
    assert "Exact saved lighting and output reproduced" in replay.status.text()


def test_prepared_crop_uses_baseline_grid_and_clears_when_series_changes(qtbot):
    from PySide6.QtCore import QObject, Signal
    from darsia.fluidflower.preparation_crop_summary import PreparationCropSummary

    class Choice(QObject):
        context_updated = Signal()
        series_root = "/example/one"
        series_id = "one"

    class Preview(QObject):
        preview_loaded = Signal(object)

    choice, preview = Choice(), Preview()
    summary = PreparationCropSummary(choice, preview)
    qtbot.addWidget(summary)
    preview.preview_loaded.emit({"run": "r2", "grid": {
        "dimensions": [.4, .8], "origin": [.1, .6], "num_voxels": [400, 800],
    }})
    assert "R2 prepared crop: 800 × 400 mm" in summary.summary.text()
    assert "x: 100–900 mm; y: 200–600 mm" in summary.summary.text()
    choice.series_root, choice.series_id = "/example/two", "two"
    choice.context_updated.emit()
    assert "R2" not in summary.summary.text()
    assert "will appear" in summary.summary.text()


def test_crop_review_routes_to_existing_geometry_editor_without_writing(qtbot):
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QWidget, QVBoxLayout
    from darsia.fluidflower.preparation_crop_summary import PreparationCropSummary

    class Choice(QObject):
        context_updated = Signal()
        series_root = "/example/one"
        series_id = "one"

    class Preview(QObject):
        preview_loaded = Signal(object)

    owner = QWidget()
    qtbot.addWidget(owner)
    calls = []
    owner._select_row = calls.append
    layout = QVBoxLayout(owner)
    choice, preview = Choice(), Preview()
    summary = PreparationCropSummary(choice, preview)
    layout.addWidget(summary)
    summary.review.click()
    assert calls == [("geometry_facies", "rig_frame")]
