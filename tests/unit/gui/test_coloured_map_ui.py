"""Synthetic local UI-boundary tests; no real series or NAS writes."""

from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json

import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QPushButton,
    QStackedWidget,
)

from darsia.fluidflower import coloured_map_ui as ui
from darsia.fluidflower import coloured_map_import as maps
from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.series_geometry import SeriesRunInputs


class Emissions(QObject):
    changed = Signal()
    exported = Signal(object)


def _fixture(tmp_path):
    root = tmp_path / "series"
    root.mkdir()
    baseline = root / "raw.png"
    Image.new("RGB", (20, 16), (30, 40, 50)).save(baseline)
    config = root / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="raw.png"\n', encoding="utf-8")
    (root / "series.toml").write_text("schema=1\n", encoding="utf-8")
    reference = SeriesRunInputs(
        "ac52", config, baseline_image=baseline, width_m=0.9, height_m=0.49
    )
    session, proposal = maps.imported_layer_state(
        np.ones((16, 20), np.uint8), reference, raw_size=(20, 16), frame=None
    )
    work = tmp_path / "own-import"
    work.mkdir()
    receipt = work / "import.json"
    receipt.write_text(
        json.dumps(
            {"extra_trees_fit_run": False, "source_map_sha256": "source-binding"}
        ),
        encoding="utf-8",
    )
    imported = maps.ImportedColouredMap(
        session, proposal, receipt, work, work / "labels.npz"
    )
    project = NS(root=root / "project")
    old_session = lp.PaintingSession.default("ac52", baseline, (20, 16))
    old_proposal = lp.LayerProposal(
        np.full((16, 20), 3, np.uint8),
        np.full((16, 20), 255, np.uint8),
        (20, 16),
        "old-digest",
    )
    lp.save_painting_session(project, old_session)
    old_proposal.save(lp.proposal_path(project, "ac52"))
    emissions = Emissions()
    view = NS(
        project=project,
        session=old_session,
        proposal=old_proposal,
        _undo_stack=["prior"],
        _redo_stack=[],
        last_export="old-export",
        _dirty=False,
        _loading=False,
        _save_timer=NS(stop=lambda: None),
        changed=emissions.changed,
        show_proposal=QCheckBox(),
        propose_status=QLabel(),
        exported=emissions.exported,
        _emissions=emissions,
    )
    for name in (
        "_sync_settings",
        "_refresh_layers",
        "_refresh_facies_table",
        "_render_marks",
        "_render_proposal",
        "_refresh_export_info",
        "_refresh_status",
    ):
        setattr(view, name, lambda: None)
    return root, reference, view, imported


def test_explicit_import_uses_original_draft_saves_and_preserves_previous_bytes(
    qapp, tmp_path
):
    root, ref, view, imported = _fixture(tmp_path)
    before = {
        p: p.read_bytes()
        for p in (
            lp.session_path(view.project, "ac52"),
            lp.proposal_path(view.project, "ac52"),
        )
    }
    receipt = ui.install_imported_draft(view, imported)
    saved = json.loads(receipt.read_bytes())
    assert saved["state"] == "complete"
    assert view.session is imported.session and view.proposal is imported.proposal
    assert view._ui_import_later_fit is False
    for record in saved["before"]:
        if "backup" in record:
            assert Path(record["backup"]).read_bytes() == before[Path(record["path"])]
        else:
            assert record["previously_absent"] is True
    assert (
        lp.LayerProposal.load(lp.proposal_path(view.project, "ac52")).digest
        == imported.proposal.digest
    )
    assert not (view.project.root / "overrides").exists()
    assert (
        ref.config_path.read_text(encoding="utf-8")
        == '[data]\nfolder="."\nbaseline="raw.png"\n'
    )


def test_original_save_failure_preserves_old_editor_and_backs_up_disk(
    qapp, tmp_path, monkeypatch
):
    _, _, view, imported = _fixture(tmp_path)
    old_session, old_proposal = view.session, view.proposal
    before = lp.session_path(view.project, "ac52").read_bytes()

    def fail(*a, **k):
        raise OSError("literal original serializer failure")

    monkeypatch.setattr(lp, "save_painting_session", fail)
    with pytest.raises(OSError, match="literal original serializer failure"):
        ui.install_imported_draft(view, imported)
    assert view.session is old_session and view.proposal is old_proposal
    assert lp.session_path(view.project, "ac52").read_bytes() == before
    failed_path = next(
        (lp.painting_folder(view.project) / "ui_import_cache").glob(
            "*/installation.failed.json"
        )
    )
    failed = json.loads(failed_path.read_bytes())
    assert "literal original serializer failure" in failed["traceback"]


@pytest.mark.parametrize(
    "failure_step",
    [
        "proposal-serialize",
        "proposal-install",
        "receipt",
        "origin-prepare",
        "origin-after-write",
    ],
)
def test_later_import_failure_restores_canonical_pair_before_actual_original_reload(
    qapp, tmp_path, monkeypatch, failure_step
):
    _, ref, view, imported = _fixture(tmp_path)
    old_session, old_proposal = view.session, view.proposal
    canonical_session = lp.session_path(view.project, ref.run_id)
    canonical_proposal = lp.proposal_path(view.project, ref.run_id)
    canonical_origin = ui._origin_path(view.project, ref.run_id)
    before = {
        p: p.read_bytes() if p.is_file() else None
        for p in (canonical_session, canonical_proposal, canonical_origin)
    }
    expected_loaded_session = lp.load_painting_session(view.project, ref).to_dict()
    if failure_step == "proposal-serialize":
        original = lp.LayerProposal.save

        def save(proposal, path):
            if proposal is imported.proposal:
                raise OSError("literal second serializer failure")
            return original(proposal, path)

        monkeypatch.setattr(lp.LayerProposal, "save", save)
    elif failure_step in {"proposal-install", "origin-after-write"}:
        original = ui._replace_expected
        failed = []

        def replace(path, raw, expected):
            target = (
                canonical_proposal
                if failure_step == "proposal-install"
                else canonical_origin
            )
            if path == target and not failed:
                failed.append(True)
                if failure_step == "origin-after-write":
                    original(path, raw, expected)
                raise OSError("literal guarded replacement failure")
            return original(path, raw, expected)

        monkeypatch.setattr(ui, "_replace_expected", replace)
    elif failure_step == "receipt":
        original = ui._write_new

        def write(path, raw):
            if path.name == "installation.json":
                raise OSError("literal installation receipt failure")
            return original(path, raw)

        monkeypatch.setattr(ui, "_write_new", write)
    else:
        monkeypatch.setattr(
            ui,
            "_origin_bytes",
            lambda *a: (_ for _ in ()).throw(
                OSError("literal origin preparation failure")
            ),
        )
    with pytest.raises(OSError, match="literal"):
        ui.install_imported_draft(view, imported)
    assert view.session is old_session and view.proposal is old_proposal
    assert all(
        (p.read_bytes() if p.is_file() else None) == raw for p, raw in before.items()
    )
    loaded_session = lp.load_painting_session(view.project, ref)
    loaded_proposal = lp.LayerProposal.load(canonical_proposal)
    assert loaded_session.to_dict() == expected_loaded_session
    assert loaded_proposal.digest == old_proposal.digest
    assert np.array_equal(loaded_proposal.classes, old_proposal.classes)
    failed_receipt = next(
        (lp.painting_folder(view.project) / "ui_import_cache").glob(
            "*/installation.failed.json"
        )
    )
    failure = json.loads(failed_receipt.read_bytes())
    assert failure["state"] == "failed-and-originals-restored"
    assert "literal" in failure["traceback"]
    for record in failure["restoration"]:
        assert record["state"] in {"restored", "already-original"}
        if record["state"] == "restored":
            assert Path(record["displaced"]).is_file()


def test_install_failure_never_overwrites_external_change_and_reports_partial_paths(
    qapp, tmp_path, monkeypatch
):
    _, ref, view, imported = _fixture(tmp_path)
    canonical = lp.session_path(view.project, ref.run_id)
    external = b'{"external":"another writer"}'
    original = ui._write_new

    def write(path, raw):
        if path.name == "installation.json":
            canonical.write_bytes(external)
            raise OSError("literal receipt failure after external draft change")
        return original(path, raw)

    monkeypatch.setattr(ui, "_write_new", write)
    with pytest.raises(RuntimeError, match="partial write") as error:
        ui.install_imported_draft(view, imported)
    assert str(canonical) in str(error.value)
    assert "literal receipt failure after external draft change" in str(error.value)
    assert canonical.read_bytes() == external
    failed_receipt = next(
        (lp.painting_folder(view.project) / "ui_import_cache").glob(
            "*/installation.failed.json"
        )
    )
    failure = json.loads(failed_receipt.read_bytes())
    assert failure["state"] == "partial-write-failure"
    assert any(
        row["path"] == str(canonical) and row["state"] == "external-change-preserved"
        for row in failure["restoration"]
    )


def test_failed_reimport_restores_existing_origin_and_original_reload_linkage(
    qapp, tmp_path, monkeypatch
):
    _, ref, view, first = _fixture(tmp_path)
    ui.install_imported_draft(view, first)
    canonical = [
        lp.session_path(view.project, ref.run_id),
        lp.proposal_path(view.project, ref.run_id),
        ui._origin_path(view.project, ref.run_id),
    ]
    before = {path: path.read_bytes() for path in canonical}
    previous_cache = view._ui_import.work_directory
    labels = np.ones((16, 20), np.uint8)
    labels[:, 10:] = 2
    session, proposal = maps.imported_layer_state(
        labels, ref, raw_size=(20, 16), frame=None
    )
    work = tmp_path / "another-explicit-import"
    work.mkdir()
    receipt = work / "import.json"
    receipt.write_bytes(first.receipt.read_bytes())
    second = maps.ImportedColouredMap(
        session, proposal, receipt, work, work / "labels.npz"
    )
    original = ui._replace_expected
    failed = []

    def replace(path, raw, expected):
        original(path, raw, expected)
        if path == canonical[2] and not failed:
            failed.append(True)
            raise OSError("literal failure after replacing previous import origin")

    monkeypatch.setattr(ui, "_replace_expected", replace)
    with pytest.raises(
        OSError, match="literal failure after replacing previous import origin"
    ):
        ui.install_imported_draft(view, second)
    assert all(path.read_bytes() == raw for path, raw in before.items())
    loaded_session = lp.load_painting_session(view.project, ref)
    loaded_proposal = lp.LayerProposal.load(canonical[1])
    restored = ui.read_import_origin(view.project, loaded_session, loaded_proposal)
    assert restored[0].work_directory == previous_cache
    assert restored[0].proposal.digest == first.proposal.digest


def test_install_controls_is_readonly_and_preserves_original_propose_callback(
    qapp, tmp_path
):
    _, _, view, imported = _fixture(tmp_path)
    window = QWidget()
    window.print_log = lambda text: None
    controller = ui.NativeColouredMapWorkflow(window, work_root=tmp_path / "imports")
    box = QWidget()
    layout = QVBoxLayout(box)
    original_button = QPushButton("Propose boundaries")
    layout.addWidget(original_button)
    view.step_propose = NS(content=box)
    calls = []

    def original_ready(value):
        calls.append(value)
        view.proposal = value

    view._proposal_ready = original_ready
    before = {p: p.read_bytes() for p in view.project.root.rglob("*") if p.is_file()}
    controller.install(view)
    controller.install(view)
    assert layout.count() == 3
    assert original_button.text() == "Propose boundaries"
    assert "raw reference" in view.import_coloured_map_button.toolTip()
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert not controller.work_root.exists()
    ui.install_imported_draft(view, imported)
    view._ui_import_later_fit = False
    view._proposal_ready(imported.proposal)
    assert calls == [imported.proposal]
    assert view._ui_import_later_fit is True


def test_reference_config_reads_actual_reference_only_and_records_races(qapp, tmp_path):
    root, ref, _, _ = _fixture(tmp_path)
    config, bindings = ui.reference_config(root, ref)
    assert config["data"]["baseline"] == "raw.png"
    ref.config_path.write_text(
        '[data]\nfolder="."\nbaseline="other.png"\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="changed"):
        bindings[-1].require_unchanged()
    with pytest.raises(ValueError, match="another raw baseline"):
        ui.reference_config(root, ref)


def _export(view, imported):
    current = view.project.root / "overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    current.write_text(
        json.dumps(
            {
                "revision_id": "saved-r1",
                "configuration": {
                    "facies_generation": {"proposal_digest": imported.proposal.digest}
                },
            }
        ),
        encoding="utf-8",
    )
    destination = view.project.root / "exports/saved-r1"
    destination.mkdir(parents=True)
    (destination / "export.json").write_text(
        '{"revision_id":"saved-r1"}', encoding="utf-8"
    )
    setup = destination / "native_setup.toml"
    setup.write_text('method="legacy-saved-format"\n', encoding="utf-8")
    view._save_worker = NS(
        session=lp.PaintingSession.from_dict(view.session.to_dict()),
        proposal=view.proposal,
    )
    return (
        NS(destination=destination, revision_id="saved-r1", setup_toml=setup),
        current,
    )


def test_import_provenance_survives_original_serializers_in_saved_export(
    qapp, tmp_path
):
    _, _, view, imported = _fixture(tmp_path)
    ui.install_imported_draft(view, imported)
    export, current = _export(view, imported)
    original_paths = [
        current,
        export.destination / "export.json",
        export.setup_toml,
        lp.session_path(view.project, "ac52"),
        lp.proposal_path(view.project, "ac52"),
    ]
    before = {p: p.read_bytes() for p in original_paths}
    sidecar = ui.bind_import_to_export(view, export)
    saved = json.loads(sidecar.read_bytes())
    assert saved["revision_id"] == "saved-r1"
    assert saved["initial_import_extra_trees_fit_run"] is False
    assert saved["subsequent_propose_boundaries_fit_run"] is False
    assert saved["legacy_kernel_method_tag"] == lp.PROPOSAL_METHOD
    assert saved["legacy_method_tag_is_retained_serializer_metadata"] is True
    assert saved["exported_proposal_digest"] == imported.proposal.digest
    assert (
        saved["revision_json_sha256"]
        == hashlib.sha256(current.read_bytes()).hexdigest()
    )
    assert any(row["source_relative"] == "import.json" for row in saved["files"])
    for row in saved["files"]:
        assert (
            hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
        )
    assert all(p.read_bytes() == raw for p, raw in before.items())
    view._ui_import_later_fit = True
    ui.persist_import_origin(view)
    second = ui.bind_import_to_export(view, export)
    assert second != sidecar and sidecar.is_file()
    assert (
        json.loads(second.read_bytes())["subsequent_propose_boundaries_fit_run"] is True
    )


def test_sidecar_refuses_changed_revision_or_receipt(qapp, tmp_path):
    _, _, view, imported = _fixture(tmp_path)
    ui.install_imported_draft(view, imported)
    export, current = _export(view, imported)
    current.write_text('{"revision_id":"foreign"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="saved revision changed"):
        ui.bind_import_to_export(view, export)
    view._ui_import.receipt.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt changed"):
        ui.bind_import_to_export(view, export)
    assert not (export.destination / "ui_import_provenance").exists()


def test_async_original_failure_keeps_existing_draft_and_reports_literal_error(
    qtbot, tmp_path, monkeypatch
):
    root, ref, view, imported = _fixture(tmp_path)
    window = QWidget()
    logs = []
    window.print_log = logs.append
    window.geometry_workspace = NS(busy=lambda: False)
    window.process_runner = NS(active=False)
    window.crop_workflow = NS(running=False)
    controller = ui.NativeColouredMapWorkflow(window, work_root=tmp_path / "imports")
    view.busy = lambda: False
    view.save_session = lambda: None
    view.import_coloured_map_note = QLabel()
    context = ui.ImportContext("series", root, view.project.root, ref, None, {}, ())
    monkeypatch.setattr(controller, "_context_for", lambda v: context)
    monkeypatch.setattr(
        ui.QFileDialog, "getOpenFileName", lambda *a, **k: (str(ref.baseline_image), "")
    )
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *a: None)
    calls = []

    def fail(*a, **k):
        calls.append(1)
        raise ValueError("literal native labeling error")

    monkeypatch.setattr(ui, "import_coloured_map", fail)
    old_session, old_proposal = view.session, view.proposal
    before = {p: p.read_bytes() for p in view.project.root.rglob("*") if p.is_file()}
    with qtbot.waitSignal(controller.finished, timeout=5000) as signal:
        assert controller.start(view)
    assert signal.args[0] is False
    assert "literal native labeling error" in signal.args[1] and "UTC" in signal.args[1]
    assert calls == [1]
    assert view.session is old_session and view.proposal is old_proposal
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert window.isEnabled() and not controller.running


def test_context_change_after_import_discards_result_before_any_install(
    qtbot, tmp_path, monkeypatch
):
    root, ref, view, imported = _fixture(tmp_path)
    window = QWidget()
    window.print_log = lambda text: None
    controller = ui.NativeColouredMapWorkflow(window)
    context = ui.ImportContext("series", root, view.project.root, ref, None, {}, ())
    controller._context, controller._view = context, view
    controller._outcome = (imported, "")
    controller._previous_enabled = True
    view.import_coloured_map_note = QLabel()
    monkeypatch.setattr(
        controller,
        "_context_for",
        lambda v: ui.ImportContext(
            "foreign", root, view.project.root, ref, None, {}, ()
        ),
    )
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *a: None)
    monkeypatch.setattr(
        ui, "install_imported_draft", lambda *a: pytest.fail("must not install")
    )
    with qtbot.waitSignal(controller.finished, timeout=1000) as signal:
        controller._finish()
    assert signal.args[0] is False and "series or reference changed" in signal.args[1]
    assert not (imported.work_directory / "before-editor-draft").exists()


def test_actual_export_ready_signal_attaches_sidecar_while_save_snapshot_exists(
    qapp, tmp_path
):
    from darsia.fluidflower.series_layer_view import LayerPainterView

    _, _, view, imported = _fixture(tmp_path)
    ui.install_imported_draft(view, imported)
    export, _ = _export(view, imported)
    view._export_state_key = lambda: "current-state"
    view._export_key = "current-state"
    observed = []
    view.exported.connect(
        lambda result: observed.append(ui.bind_import_to_export(view, result))
    )
    LayerPainterView._export_ready(view, export)
    assert view.last_export is export and view._export_stale is False
    assert len(observed) == 1 and observed[0].is_file()
    assert (
        json.loads(observed[0].read_bytes())["save_snapshot_proposal_classes_sha256"]
        == hashlib.sha256(imported.proposal.classes.tobytes()).hexdigest()
    )


def test_newly_created_common_or_merged_input_during_import_discards_result(
    qtbot, tmp_path, monkeypatch
):
    root, ref, view, imported = _fixture(tmp_path)
    window = QWidget()
    window.print_log = lambda text: None
    controller = ui.NativeColouredMapWorkflow(window)
    context = ui.ImportContext("series", root, view.project.root, ref, None, {}, ())
    controller._context, controller._view = context, view
    controller._outcome = (imported, "")
    controller._previous_enabled = True
    view.import_coloured_map_note = QLabel()
    newly_read = ui.ReadBinding(root / "common.toml", b"[labeling]\nrtol=0.1\n")
    monkeypatch.setattr(
        controller,
        "_context_for",
        lambda v: ui.ImportContext(
            "series",
            root,
            view.project.root,
            ref,
            None,
            {"labeling": {"rtol": 0.1}},
            (newly_read,),
        ),
    )
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *a: None)
    monkeypatch.setattr(
        ui, "install_imported_draft", lambda *a: pytest.fail("must not install")
    )
    with qtbot.waitSignal(controller.finished, timeout=1000) as signal:
        controller._finish()
    assert (
        signal.args[0] is False
        and "reference configuration inputs changed" in signal.args[1]
    )
    assert not (imported.work_directory / "before-editor-draft").exists()


def test_close_reopen_restores_explicit_origin_and_binds_future_export_after_temp_moves(
    qapp, tmp_path
):
    _, ref, first, imported = _fixture(tmp_path)
    ui.install_imported_draft(first, imported)
    durable = first._ui_import.work_directory
    assert durable.is_relative_to(lp.painting_folder(first.project))
    original_cache = imported.work_directory
    original_cache.rename(tmp_path / "temporary-cache-no-longer-at-original-path")
    # A new view uses only original serializers and the one exact per-run pointer.
    session = lp.load_painting_session(first.project, ref)
    proposal = lp.LayerProposal.load(lp.proposal_path(first.project, ref.run_id))
    reopened = NS(project=first.project, session=session, proposal=proposal)
    restored = ui.restore_import_origin(reopened)
    assert restored is not None and reopened._ui_import.work_directory == durable
    assert reopened._ui_import_later_fit is False
    export, _ = _export(reopened, reopened._ui_import)
    path = ui.bind_import_to_export(reopened, export)
    document = json.loads(path.read_bytes())
    assert document["initial_import_extra_trees_fit_run"] is False
    assert document["subsequent_propose_boundaries_fit_run"] is False
    assert (
        document["draft_origin_sha256"]
        == hashlib.sha256((path.parent / "draft-origin.json").read_bytes()).hexdigest()
    )
    assert document["exported_proposal_digest"] == proposal.digest


def test_origin_restore_does_not_guess_when_proposal_binding_differs(qapp, tmp_path):
    _, ref, view, imported = _fixture(tmp_path)
    ui.install_imported_draft(view, imported)
    loaded = lp.LayerProposal.load(lp.proposal_path(view.project, ref.run_id))
    loaded.classes[0, 0] = 9
    reopened = NS(
        project=view.project,
        session=lp.load_painting_session(view.project, ref),
        proposal=loaded,
    )
    with pytest.raises(RuntimeError, match="no longer matches"):
        ui.restore_import_origin(reopened)
    assert reopened._ui_import is None


def test_controller_reopen_and_reload_retain_only_the_explicit_run_origin(
    qapp, tmp_path
):
    _, ref, view, imported = _fixture(tmp_path)
    ui.install_imported_draft(view, imported)
    # Drop all view-only import attributes, as a new LayerPainterView does.
    for name in ("_ui_import", "_ui_import_installation", "_ui_import_later_fit"):
        delattr(view, name)
    view.session = lp.load_painting_session(view.project, ref)
    view.proposal = lp.LayerProposal.load(lp.proposal_path(view.project, ref.run_id))
    box = QWidget()
    QVBoxLayout(box)
    view.step_propose = NS(content=box)

    def ready(proposal):
        view.proposal = proposal

    view._proposal_ready = ready

    def reload_other_reference():
        view.session = lp.PaintingSession.default("ac19", ref.baseline_image, (20, 16))
        view.proposal = None

    view.reload = reload_other_reference
    window = QWidget()
    logs = []
    window.print_log = logs.append
    controller = ui.NativeColouredMapWorkflow(window)
    controller.install(view)
    assert (
        view._ui_import is not None
        and "Restored" in view.import_coloured_map_note.text()
    )
    fitted = lp.LayerProposal(
        np.full((16, 20), 2, np.uint8),
        np.full((16, 20), 255, np.uint8),
        (20, 16),
        "actual-later-fit",
    )
    fitted.save(lp.proposal_path(view.project, ref.run_id))
    view._proposal_ready(fitted)
    origin = ui.read_import_origin(view.project, view.session, fitted)
    assert origin[2] is True
    view.reload()
    assert view.session.run_id == "ac19" and view._ui_import is None


def test_restore_rejects_modified_source_manifest_instead_of_trusting_new_hashes(
    qapp, tmp_path
):
    _, _, view, imported = _fixture(tmp_path)
    ui.install_imported_draft(view, imported)
    (view._ui_import.work_directory / "cache-files.json").write_bytes(b"[]")
    with pytest.raises(RuntimeError, match="source manifest changed"):
        ui.read_import_origin(view.project, view.session, view.proposal)


def test_advanced_import_routes_explicitly_to_same_simple_editor_action(
    qapp, tmp_path, monkeypatch
):
    window = QWidget()
    window.print_log = lambda text: None
    controller = ui.NativeColouredMapWorkflow(window)
    workspace = QWidget()
    outer = QVBoxLayout(workspace)
    controls = QHBoxLayout()
    outer.addLayout(controls)
    advanced = QPushButton("Advanced view")
    advanced.setCheckable(True)
    controls.addWidget(advanced)
    controls.addStretch()
    workspace.advanced_button = advanced
    workspace.simple_view = object()
    workspace.busy = lambda: False
    calls = []
    workspace.show_advanced = lambda value: calls.append(("show_advanced", value))
    monkeypatch.setattr(
        controller, "install", lambda view: calls.append(("install", view))
    )
    monkeypatch.setattr(controller, "start", lambda view: calls.append(("start", view)))
    controller.install_workspace(workspace)
    button = workspace.advanced_import_coloured_map_button
    assert button.isHidden()
    advanced.setChecked(True)
    assert not button.isHidden()
    button.click()
    assert calls == [
        ("show_advanced", False),
        ("install", workspace.simple_view),
        ("start", workspace.simple_view),
    ]
    assert button.isHidden()
    assert not controller.work_root.exists()


def test_series_discard_hides_advanced_import_when_original_blocks_toggle_signals(qapp):
    from darsia.fluidflower.series_geometry_workspace import SeriesGeometryWorkspace

    window = QWidget()
    window.print_log = lambda text: None
    controller = ui.NativeColouredMapWorkflow(window)
    workspace = QWidget()
    outer = QVBoxLayout(workspace)
    controls = QHBoxLayout()
    outer.addLayout(controls)
    workspace.advanced_button = QPushButton("Advanced view")
    workspace.advanced_button.setCheckable(True)
    controls.addWidget(workspace.advanced_button)
    controls.addStretch()
    workspace.view_stack = QStackedWidget()
    old_series_view = QWidget()
    old_series_view.wait_for_workers = lambda: None
    workspace.view_stack.addWidget(old_series_view)
    workspace.simple_view = old_series_view
    workspace.export_button = QPushButton("Export")
    workspace._discard_advanced_page = lambda: None
    workspace._discard_page = lambda: SeriesGeometryWorkspace._discard_page(workspace)
    workspace.show_advanced = lambda state: None
    workspace.busy = lambda: False
    controller.install_workspace(workspace)
    workspace.advanced_button.setChecked(True)
    assert not workspace.advanced_import_coloured_map_button.isHidden()
    toggles = []
    workspace.advanced_button.toggled.connect(toggles.append)
    # This is the original method used while replacing the loaded series page.
    workspace._discard_page()
    assert workspace.simple_view is None
    assert not workspace.advanced_button.isChecked()
    assert toggles == []  # The original implementation blocked signals.
    assert workspace.advanced_import_coloured_map_button.isHidden()


def test_final_setup_sidebar_rows_and_navigation_have_no_import_computation(
    qtbot, tmp_path, monkeypatch
):
    from PySide6.QtCore import QSettings, QCoreApplication
    from darsia.fluidflower import native_darsia_app as native

    old = (
        QSettings.defaultFormat(),
        QCoreApplication.organizationName(),
        QCoreApplication.applicationName(),
    )
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )
    QCoreApplication.setOrganizationName("DarSIA-import-ui-tests")
    QCoreApplication.setApplicationName(tmp_path.name)
    monkeypatch.setattr(native, "configured_series", lambda: ())
    monkeypatch.setattr(native, "ProjectRegistry", lambda: NS(entries=lambda: []))
    monkeypatch.setattr(
        native.PhaseModelWorkspace,
        "ensure_studio",
        lambda self: self.studio_placeholder,
    )
    monkeypatch.setattr(
        native.SeriesGeometryWorkspace, "ensure_page", lambda self: self
    )
    monkeypatch.setattr(
        ui,
        "import_coloured_map",
        lambda *a, **k: pytest.fail("navigation must not compute"),
    )
    window = native.NativeFluidFlowerMainWindow()
    qtbot.addWidget(window)
    try:
        rows = window.sidebar._sections["setup"]._rows
        assert list(rows) == [
            "preprocessing:protocols",
            "overflow_data:runs",
            "image_selection:plan",
            "geometry_facies:rig_frame",
            "setup:depth",
            "geometry_facies:layers",
            "geometry_facies:export",
            "setup:rig",
            "setup:all",
        ]
        assert rows["geometry_facies:layers"].label == "Layers and facies"
        assert rows["setup:rig"].label == "Build run model"
        assert rows["setup:all"].label == "Full setup"
        assert not window.coloured_map_workflow.running
        window._select_row(("geometry_facies", "layers"))
        assert not window.coloured_map_workflow.running
    finally:
        window.close()
        QCoreApplication.setOrganizationName(old[1])
        QCoreApplication.setApplicationName(old[2])
        QSettings.setDefaultFormat(old[0])
