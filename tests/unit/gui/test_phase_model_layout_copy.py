"""Copy-only storage with actual private ledger, library and session readers."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace as NS

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent

from darsia.fluidflower import phase_model_layout as layout
from darsia.fluidflower import phase_model_library as library
from darsia.fluidflower import phase_review_retractions as withdrawals
from darsia.fluidflower import review_storage, series_paths
from darsia.fluidflower.phase_model_layout_dialog import PhaseModelLayoutDialog
from darsia.fluidflower.phase_workflow import discover_phase_review_sessions
from darsia.fluidflower.series_method import read_series_method


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())
    return path


@pytest.fixture
def originals(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if ((name.startswith("DARSIA_") and name.endswith("_PROJECT_ROOT"))
                or name in {"DARSIA_FLUIDFLOWER_DATA_ROOT", "DARSIA_FLUIDFLOWER_REVIEW_ROOT", "DARSIA_REVIEW_ALIAS_FILES"}):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version = 1\nseries_id = "FF_AC"\nname = "Private series"\n'
        '\n[paths]\nreviews = "reviews"\nmodels = "models"\n'
        '\n# This selected model and calibration must remain verbatim.\n[method]\nactive = "phase_model"\n'
        '[method.phase_model]\nmodel_version = "models/phase-model-example"\nmass_calibration = "saved-calibration"\n',
        encoding="utf-8")
    series_paths.register_series(root)
    model = root / "models/phase-model-example"
    write(model / "summary.json", dict(model_id=model.name, created_utc="2026-01-01T00:00:00+00:00",
        inventory=dict(datasets=["ffac"]), provisional=True, validation_only=True))
    from darsia.fluidflower.phase_features import OpticalFeatureContract
    contract = OpticalFeatureContract()
    features = list(contract.feature_names)
    for head in ("co2_presence", "gas_within_co2"):
        write(model / "models/generic" / (head + "_signal.json"), dict(
            feature_contract=contract.to_dict(), feature_names=features,
            feature_mean=[0.0] * len(features), feature_scale=[1.0] * len(features),
            coefficients=[0.0] * len(features), intercept=0.0))
    (model / "weights.bin").write_bytes(b"immutable optical parameters")
    pointer = write(root / "models/phase/example.json", dict(format=library.FORMAT, kind="phase",
        source_identity=library.source_identity(model), target=dict(path="models/phase-model-example", relative_to="series_root"),
        metadata=library._metadata(model, series="ffac")))
    session = root / "reviews/reviewer/campaign/workspace/manual_pairwise_review"
    write(session / "review_session.json", dict(format="fluidflower-local-review-session-v1",
        reviewer_id="reviewer", campaign_id="campaign", workspace="workspace", target_model_id=model.name,
        decision_dir=str(session), local_review_root=str(root / "reviews"), ffac_runs=["experiment"], ffum_runs=[],
        include_in_training=False, editing_locked=True))
    write(session / "pairwise_review_decisions.json", dict(decisions={"experiment": dict(annotations=[], frame_choices=[])}))
    event = write(root / "reviews/evidence_actions/events/example.json", dict(format=withdrawals.EVENT_FORMAT,
        event_id="example", action="retract", actor="reviewer", created_utc="2026-01-01T00:00:00+00:00",
        target=dict(series="ffac", run="experiment", kind="manual", identifier="region")))
    write(root / "phase_review_collections/example/binding.json", dict(frozen=True, admission=False))
    review_storage.review_aliases(refresh=True)
    return NS(root=root, model=model, pointer=pointer, session=session, event=event,
        before={str(p):p.read_bytes() for p in root.rglob("*") if p.is_file()})


def test_preview_has_no_writes_and_lists_exact_bytes_and_locations(originals):
    data = originals
    result = layout.preview_layout(data.root)
    assert result["files"] and result["bytes_to_copy"] > 0
    assert not (data.root / "ExpertKnowledge/PhaseModels").exists()
    assert {str(p):p.read_bytes() for p in data.root.rglob("*") if p.is_file()} == data.before
    assert all(row["source"]["sha256"] == layout._pin(row["source"]["path"])["sha256"] for row in result["files"])
    assert result["paths"]["evidence_actions"] == "ExpertKnowledge/PhaseModels/TrainingInput/evidence_actions"


def test_copy_preserves_all_originals_model_identity_method_sessions_and_withdrawals(originals):
    data = originals
    chosen, snapshot = read_series_method(data.root), withdrawals.load_retractions(data.root, series="ffac")
    plan = layout.preview_layout(data.root)
    result = layout.apply_layout(plan, writers_quiescent=True)
    assert result["complete"] is True and result["adoption"] is False and result["deletion"] is False
    assert read_series_method(data.root) == chosen
    for path, raw in data.before.items():
        if Path(path).name != "series.toml":
            assert Path(path).read_bytes() == raw
    assert Path(result["original_series_manifest"]["path"]).read_bytes() == data.before[str(data.root / "series.toml")]
    assert withdrawals.load_retractions(data.root, series="ffac") == snapshot
    assert withdrawals.ledger_folder(data.root, before_phase_model=True) == withdrawals.ledger_folder(data.root)
    assert (data.root / "reviews/evidence_actions/events/example.json").is_file()
    copied = review_storage.resolve_review_path(data.model)
    assert copied.is_relative_to(data.root / "ExpertKnowledge/PhaseModels/Models") and copied.name == data.model.name
    assert library.source_identity(copied) == plan["model_checks"][0]["identity"]
    records = library.library_records(data.root)
    assert len(records) == 1 and records[0]["available"] and Path(records[0]["root"]) == copied
    sessions = discover_phase_review_sessions([data.root / "ExpertKnowledge/PhaseModels/TrainingInput"])
    assert len(sessions) == 1 and sessions[0]["include_in_training"] is False and sessions[0]["editing_locked"] is True
    assert Path(sessions[0]["decision_dir"]).is_relative_to(data.root / "ExpertKnowledge/PhaseModels/TrainingInput")
    assert layout.apply_layout(plan, writers_quiescent=True) == result


def test_copy_keeps_classification_and_validation_roles_together_with_only_two_main_folders(originals):
    data = originals
    session = data.root / "reviews" / "training" / "manual_pairwise_review"
    payload = json.loads((data.session / "review_session.json").read_bytes())
    payload.update(decision_dir=str(session), include_in_training=True, editing_locked=False)
    write(session / "review_session.json", payload)
    write(session / "pairwise_review_decisions.json", dict(decisions={}))
    before = (session / "review_session.json").read_bytes()
    result = layout.apply_layout(layout.preview_layout(data.root), writers_quiescent=True)
    assert result["complete"]
    assert {path.name for path in (data.root / "ExpertKnowledge/PhaseModels").iterdir()} == {"TrainingInput", "Models"}
    copied = discover_phase_review_sessions([data.root / "ExpertKnowledge/PhaseModels" / "TrainingInput"])
    assert sorted((row["include_in_training"], row["editing_locked"]) for row in copied) == [(False, True), (True, False)]
    assert review_storage.resolve_review_path(session / "review_session.json").read_bytes() == before


@pytest.mark.parametrize("previous_layout", [layout._LEGACY_LAYOUT, layout._PREVIOUS_LAYOUT])
def test_legacy_reviewed_copy_plan_remains_applicable_with_its_original_paths(originals, monkeypatch, previous_layout):
    data = originals
    with monkeypatch.context() as legacy:
        legacy.setattr(layout, "_LAYOUT", dict(previous_layout))
        plan = layout.preview_layout(data.root)
    result = layout.apply_layout(plan, writers_quiescent=True)
    assert result["complete"] and plan["paths"] == previous_layout
    assert review_storage.resolve_review_path(data.model).is_relative_to(data.root / previous_layout["phase_models"])
    assert not (data.root / "ExpertKnowledge/PhaseModels").exists()


def test_organise_does_not_copy_mass_calibration_as_a_phase_model(originals, monkeypatch):
    data = originals
    calibration = data.root / "models" / "mass_calibration"
    write(calibration / "summary.json", dict(format="mass-calibration-summary-v1", fitted=True))
    write(calibration / "parameters.json", dict(coefficient=1.25))
    plan = layout.preview_layout(data.root)
    assert len(plan["model_checks"]) == 1
    assert all(not Path(tree["source"]).is_relative_to(calibration) for tree in plan["trees"])


def test_copy_requires_explicit_quiescence_and_refuses_a_foreign_ledger_lock(originals):
    data = originals
    plan = layout.preview_layout(data.root)
    with pytest.raises(ValueError, match="explicitly confirm"):
        layout.apply_layout(plan)
    lock = withdrawals.ledger_folder(data.root) / ".write.lock"
    lock.write_text("another writer", encoding="utf-8")
    with pytest.raises(ValueError, match="Another evidence change"):
        layout.preview_layout(data.root)
    with pytest.raises(ValueError, match="Another evidence change"):
        layout.apply_layout(plan, writers_quiescent=True)
    assert lock.read_text(encoding="utf-8") == "another writer"
    assert not (data.root / "ExpertKnowledge/PhaseModels").exists()
    assert (data.root / "series.toml").read_bytes() == data.before[str(data.root / "series.toml")]


def test_canonical_lock_covers_cutover_and_future_writes_use_the_new_ledger(originals, monkeypatch):
    data = originals
    plan = layout.preview_layout(data.root)
    old_folder = withdrawals.ledger_folder(data.root)
    replace = layout.os.replace
    seen = []
    def cutover(source, destination):
        assert (old_folder / ".write.lock").is_file()
        with pytest.raises(ValueError, match="Another evidence change"):
            with withdrawals._ledger_write(data.root):
                pytest.fail("A concurrent old-ledger writer acquired the migration lock")
        seen.append(True)
        return replace(source, destination)
    monkeypatch.setattr(layout.os, "replace", cutover)
    result = layout.apply_layout(plan, writers_quiescent=True)
    assert seen == [True] and result["writers_quiescent_confirmed"] is True
    assert "Delayed stale clients" in result["concurrency_limit"]
    new_folder = withdrawals.ledger_folder(data.root)
    assert new_folder != old_folder and not (old_folder / ".write.lock").exists()
    assert not (new_folder / ".write.lock").exists()
    event = json.loads(data.event.read_bytes())
    event["event_id"] = "later"
    event["target"]["identifier"] = "later-region"
    with withdrawals._ledger_write(data.root):
        withdrawals._append_event(data.root, event)
    assert (new_folder / "later.json").is_file() and not (old_folder / "later.json").exists()
    assert len(withdrawals.load_retractions(data.root, series="ffac")["events"]) == 2


def test_incremental_registry_change_refuses_during_preview_without_copying(originals, tmp_path):
    data = originals
    layout.apply_layout(layout.preview_layout(data.root), writers_quiescent=True)
    external = tmp_path / "later_reviews/session"
    write(external / "review_session.json", dict(format="fluidflower-local-review-session-v1",
        ffac_runs=["experiment"], ffum_runs=[], local_review_root=str(external.parent)))
    before = {str(p):p.read_bytes() for p in data.root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="first-migration tool cannot extend"):
        layout.preview_layout(data.root, external_sessions=[external], model_roots=[])
    assert {str(p):p.read_bytes() for p in data.root.rglob("*") if p.is_file()} == before


def test_existing_context_specific_aliases_follow_the_copy_without_rewriting_sessions(originals, tmp_path):
    data = originals
    old_save = tmp_path / "legacy_save"
    old_session = old_save / "saved_session"
    registry = data.root / "reviews/storage_aliases.json"
    write(registry, dict(format=review_storage.FORMAT, aliases=[dict(source=str(old_session),
        destination=str(data.session), original_save_root=str(old_save), save_root=str(data.root / "reviews"))]))
    session_file = data.session / "review_session.json"
    payload = json.loads(session_file.read_bytes())
    payload["local_review_root"] = str(old_save)
    session_file.write_bytes(json.dumps(payload).encode())
    original_session, original_registry = session_file.read_bytes(), registry.read_bytes()
    review_storage.review_aliases(refresh=True)
    plan = layout.preview_layout(data.root)
    layout.apply_layout(plan, writers_quiescent=True)
    copied_session = review_storage.resolve_review_path(old_session)
    assert copied_session.is_relative_to(data.root / "ExpertKnowledge/PhaseModels/TrainingInput")
    assert review_storage.resolve_review_path(old_save, context=copied_session) == data.root / "ExpertKnowledge/PhaseModels/TrainingInput"
    assert (copied_session / "review_session.json").read_bytes() == original_session
    assert registry.read_bytes() == original_registry
    historical_registry = review_storage.resolve_review_path(registry)
    assert historical_registry.is_relative_to(data.root / "ExpertKnowledge/PhaseModels/TrainingInput/storage_aliases_history")
    assert historical_registry.read_bytes() == original_registry


@pytest.mark.parametrize("changed", ["session", "model", "event"])
def test_changed_original_after_preview_refuses_before_copy_or_path_update(originals, changed):
    data = originals
    plan = layout.preview_layout(data.root)
    target = {"session": data.session / "review_session.json", "model": data.model / "weights.bin", "event": data.event}[changed]
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 1
    target.write_bytes(raw)
    with pytest.raises(ValueError, match="source tree changed"):
        layout.apply_layout(plan, writers_quiescent=True)
    assert not (data.root / "ExpertKnowledge/PhaseModels").exists()
    assert (data.root / "series.toml").read_bytes() == data.before[str(data.root / "series.toml")]


def test_destination_collision_preserves_existing_file_and_old_storage_settings(originals):
    data = originals
    plan = layout.preview_layout(data.root)
    target = Path(plan["files"][0]["destination"])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing different file")
    with pytest.raises(ValueError, match="planned source or copy changed"):
        layout.apply_layout(plan, writers_quiescent=True)
    assert target.read_bytes() == b"existing different file"
    assert (data.root / "series.toml").read_bytes() == data.before[str(data.root / "series.toml")]
    assert list((data.root / "ExpertKnowledge/PhaseModels/TrainingInput/.layout").glob("*/error.json"))


def test_interrupted_copy_retains_partial_and_never_retries_its_attempt(originals, monkeypatch):
    data = originals
    plan = layout.preview_layout(data.root)
    def fail(_incoming, outgoing, **_kwargs):
        outgoing.write(b"partial copy")
        raise OSError("Destination unavailable")
    monkeypatch.setattr(layout.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="Destination unavailable"):
        layout.apply_layout(plan, writers_quiescent=True)
    target = Path(plan["files"][0]["destination"])
    assert target.read_bytes() == b"partial copy"
    with pytest.raises(ValueError, match="partial files are preserved"):
        layout.apply_layout(plan, writers_quiescent=True)
    assert (data.root / "series.toml").read_bytes() == data.before[str(data.root / "series.toml")]


def test_external_mixed_series_session_and_changed_plan_destination_refuse(originals, tmp_path):
    data = originals
    external = tmp_path / "external"
    write(external / "review_session.json", dict(format="fluidflower-local-review-session-v1", ffac_runs=["experiment"], ffum_runs=["other"]))
    with pytest.raises(ValueError, match="only the selected series' run scope"):
        layout.preview_layout(data.root, external_sessions=[external])
    plan = layout.preview_layout(data.root)
    plan["files"][0]["destination"] = str(tmp_path / "outside.bin")
    material = deepcopy(plan)
    material.pop("plan_sha256")
    plan["plan_sha256"] = layout._digest(material)
    with pytest.raises(ValueError, match="leaves the selected phase-model"):
        layout.apply_layout(plan, writers_quiescent=True)
    assert not (tmp_path / "outside.bin").exists() and not (data.root / "ExpertKnowledge/PhaseModels").exists()


@pytest.mark.parametrize("identity", ['series_id = "Campaign_A"', 'dataset = "Campaign_A"', '[series]\nid = "Campaign_A"'])
def test_selected_manifest_series_controls_copy_ownership_sessions_and_withdrawals(originals, tmp_path, identity):
    data = originals
    root = tmp_path / "generic_series"
    root.mkdir()
    manifest = root / "series.toml"
    manifest.write_text('schema_version = 1\nseries_id = "Campaign_A"\n', encoding="utf-8")
    series_paths.register_series(root)
    registry_id = '' if identity.startswith('series_id') else 'series_id = "Campaign_A"\n'
    manifest.write_text('schema_version = 1\n' + registry_id + identity
        + '\n[paths]\nreviews = "reviews"\nmodels = "models"\n', encoding="utf-8")
    model = root / "models/model-copy"
    shutil.copytree(data.model, model)
    summary = json.loads((model / "summary.json").read_bytes())
    summary["inventory"]["datasets"] = ["Campaign-A"]
    write(model / "summary.json", summary)
    library.index_model(root, model)
    event = json.loads(data.event.read_bytes())
    event["target"]["series"] = "campaigna"
    write(root / "reviews/evidence_actions/events/example.json", event)
    external = tmp_path / "generic_external/session"
    write(external / "review_session.json", dict(format="fluidflower-local-review-session-v1",
        campaigna_runs=["run-one"], unused_runs=[], local_review_root=str(external.parent)))
    original = {path: path.read_bytes() for folder in (root, external) for path in folder.rglob("*") if path.is_file()}
    before = withdrawals.load_retractions(root, series="campaigna")
    plan = layout.preview_layout(root, external_sessions=[external])
    assert plan["series"] == "campaigna" and plan["withdrawal_snapshot"] == before
    assert [Path(row["source"]) for row in plan["model_checks"]] == [model]
    assert any(row["kind"] == "external_session" for row in plan["trees"])
    result = layout.apply_layout(plan, writers_quiescent=True)
    assert result["complete"] and withdrawals.load_retractions(root, series="campaigna") == before
    assert all(path.read_bytes() == raw for path, raw in original.items() if path.name != "series.toml")
    assert layout.apply_layout(plan, writers_quiescent=True) == result


@pytest.mark.parametrize("settings", [dict(series=dict(id="Campaign_A")), dict(dataset="Campaign_A"), dict(series_id="Campaign_A")])
def test_manifest_series_aliases_normalise_without_a_fixed_campaign(settings):
    assert layout._manifest_series(settings) == "campaigna"


@pytest.mark.parametrize("scope", [["ffac", "another"], ["another"], []])
def test_model_ownership_never_guesses_or_imports_another_series(originals, scope):
    data = originals
    summary = json.loads((data.model / "summary.json").read_bytes())
    summary["inventory"]["datasets"] = scope
    write(data.model / "summary.json", summary)
    if not scope:
        with pytest.raises(ValueError, match="Model ownership is not recorded"):
            layout.preview_layout(data.root, model_roots=[data.model])
    else:
        plan = layout.preview_layout(data.root, model_roots=[data.model])
        assert not plan["model_checks"] and plan["notices"]
        assert all(row["kind"] != "model" for row in plan["trees"])
    assert not (data.root / "ExpertKnowledge/PhaseModels").exists()


def test_rebound_plan_cannot_name_another_series(originals):
    plan = layout.preview_layout(originals.root)
    plan["series"] = "another"
    material = dict(plan)
    material.pop("plan_sha256")
    plan["plan_sha256"] = layout._digest(material)
    with pytest.raises(ValueError, match="another selected series"):
        layout.apply_layout(plan, writers_quiescent=True)
    assert not (originals.root / "ExpertKnowledge/PhaseModels").exists()


def test_control_receipts_cannot_escape_through_an_existing_layout_junction(originals, tmp_path):
    data = originals
    plan = layout.preview_layout(data.root)
    external = tmp_path / "external_control"
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_bytes(b"preserve existing external files")
    phase_root = data.root / "ExpertKnowledge/PhaseModels" / "TrainingInput"
    phase_root.mkdir(parents=True)
    link = phase_root / ".layout"
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip("Directory symlink unavailable: " + str(error))
        command = subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(link), str(external)],
            capture_output=True, text=True, check=False)
        if command.returncode:
            pytest.skip("Private directory junction unavailable: " + command.stderr)
    with pytest.raises(ValueError, match="link or junction"):
        layout.apply_layout(plan, writers_quiescent=True)
    assert list(external.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"preserve existing external files"
    assert (data.root / "series.toml").read_bytes() == data.before[str(data.root / "series.toml")]


def test_ancestor_alias_cannot_place_the_copy_inside_its_physical_source(originals, tmp_path):
    data = originals
    alias = tmp_path / "ancestor_alias"
    try:
        alias.symlink_to(tmp_path, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip("Directory symlink unavailable: " + str(error))
        command = subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(alias), str(tmp_path)],
            capture_output=True, text=True, check=False)
        if command.returncode:
            pytest.skip("Private directory junction unavailable: " + command.stderr)
    manifest = data.root / "series.toml"
    raw = manifest.read_text(encoding="utf-8")
    raw = raw.replace('reviews = "reviews"', 'reviews = ' + json.dumps(str(data.root)))
    manifest.write_text(raw, encoding="utf-8")
    before = {str(p):p.read_bytes() for p in data.root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="physically inside its original source"):
        layout.preview_layout(alias / data.root.name, model_roots=[])
    assert not (data.root / "ExpertKnowledge/PhaseModels").exists()
    assert {str(p):p.read_bytes() for p in data.root.rglob("*") if p.is_file()} == before


def test_layout_refresh_keeps_the_exact_older_selection_and_ignores_another_series(originals, qtbot):
    from PySide6.QtWidgets import QComboBox
    from darsia.fluidflower.native_phase_steps import NativePhaseSteps
    from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage as Page
    data = originals
    plan = layout.preview_layout(data.root)
    layout.apply_layout(plan, writers_quiescent=True)
    copied = review_storage.resolve_review_path(data.model)
    older = dict(root=str(data.model), model_id=data.model.name, version_sha256="older-version",
        datasets=["ffac"], created_timestamp=1)
    newer = dict(root=str(data.root / "models/newer"), model_id="newer", version_sha256="newer-version",
        datasets=["ffac"], created_timestamp=2)
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItem("older", older["root"])
    combo.addItem("newer", newer["root"])
    combo.setCurrentIndex(0)
    requests = []
    studio = NS(validation_model=combo, model_history=[newer, older], _active_model_series="ffac",
        _saved_model_preference_key=None, _saved_model_preference_root="",
        _saved_model_preference_pending=False, _model_catalog_generation=0,
        _validation_model_label=lambda row: row["model_id"], _validation_model_changed=lambda: None,
        _sync_default_model_libraries=lambda: None)
    studio._selected_validation_model = lambda: Page._selected_validation_model(studio)
    studio._populate_validation_models = lambda **kwargs: Page._populate_validation_models(studio, **kwargs)
    def refresh():
        requests.append(True)
        studio.model_history = [newer, dict(older, root=str(copied))]
        studio._populate_validation_models()
    studio._request_async_model_refresh = refresh
    studio.set_saved_model_preference = lambda root, **kwargs: Page.set_saved_model_preference(studio, root, **kwargs)
    phases = NS(series_root=data.root, studio=studio, refresh=lambda: None)
    NativePhaseSteps._phase_locations_changed(phases, data.root)
    selected = studio._selected_validation_model()
    assert selected["root"] == str(copied) and selected["version_sha256"] == "older-version"
    assert requests == [True]
    phases.series_root = data.root.parent / "another_series"
    NativePhaseSteps._phase_locations_changed(phases, data.root)
    assert requests == [True] and studio._selected_validation_model() == selected


def test_gui_requires_preview_then_explicit_copy_and_keeps_work_off_gui_thread(qtbot, tmp_path, record_property):
    from PySide6.QtWidgets import QStyle, QStyleOptionButton
    calls, main_thread = [], threading.get_ident()
    def planner(root, **_kwargs):
        calls.append(("preview", threading.get_ident(), root))
        return dict(files=[], bytes_to_copy=0, paths=dict(layout._LAYOUT), notices=[])
    def copier(plan, *, writers_quiescent):
        assert writers_quiescent is True
        calls.append(("copy", threading.get_ident(), plan))
        return dict(complete=True)
    widget = PhaseModelLayoutDialog(tmp_path, planner=planner, copier=copier)
    qtbot.addWidget(widget, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    widget.show()
    assert widget.isModal() and widget.windowModality() == Qt.WindowModality.ApplicationModal
    assert not widget.copy.isEnabled() and not calls
    qtbot.mouseClick(widget.preview, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: widget.can_close() and widget.plan is not None, timeout=10000)
    assert [row[0] for row in calls] == ["preview"]
    assert not widget.copy.isEnabled()
    assert widget.quiescent.isEnabled() and not widget.quiescent.isChecked()
    option = QStyleOptionButton()
    widget.quiescent.initStyleOption(option)
    style = widget.quiescent.style()
    indicator = style.subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, widget.quiescent)
    click_rect = style.subElementRect(QStyle.SubElement.SE_CheckBoxClickRect, option, widget.quiescent)
    point = indicator.center()
    geometry = dict(widget=widget.quiescent.rect().getRect(), indicator=indicator.getRect(),
        click_rect=click_rect.getRect(), point=(point.x(), point.y()),
        default_centre_in_click_rect=click_rect.contains(widget.quiescent.rect().center()))
    record_property("quiescence_click_geometry", json.dumps(geometry, sort_keys=True))
    assert not indicator.isEmpty() and click_rect.contains(point), geometry
    with qtbot.waitSignal(widget.quiescent.toggled, timeout=1000) as toggled:
        qtbot.mouseClick(widget.quiescent, Qt.MouseButton.LeftButton, pos=point)
    assert toggled.args == [True] and widget.quiescent.isChecked(), geometry
    assert widget.copy.isEnabled(), geometry
    qtbot.mouseClick(widget.copy, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: widget.can_close() and len(calls) == 2, timeout=10000)
    assert all(row[1] != main_thread for row in calls)
    assert not widget.copy.isEnabled() and "Original files remain" in widget.summary.text()


def test_gui_close_and_main_window_close_wait_for_valid_copy(qtbot, tmp_path):
    from darsia.fluidflower.native_darsia_app import NativeFluidFlowerMainWindow
    entered, release = threading.Event(), threading.Event()
    def planner(_root, **_kwargs):
        entered.set()
        assert release.wait(5)
        return dict(files=[], bytes_to_copy=0, paths=dict(layout._LAYOUT), notices=[])
    widget = PhaseModelLayoutDialog(tmp_path, planner=planner)
    def finish(value):
        release.set()
        qtbot.waitUntil(value.can_close, timeout=10000)
    qtbot.addWidget(widget, before_close_func=finish)
    widget.show()
    widget.preview_files()
    qtbot.waitUntil(entered.is_set, timeout=5000)
    try:
        widget.close()
        assert not widget.can_close() and widget.isVisible()
        logs = []
        event = QCloseEvent()
        parent = NS(phase_model_workspace=NS(studio_page=NS(native_steps=widget)), print_log=logs.append)
        NativeFluidFlowerMainWindow.closeEvent(parent, event)
        assert not event.isAccepted() and logs
    finally:
        release.set()
    qtbot.waitUntil(widget.can_close, timeout=10000)
    assert widget._thread is None and widget._worker is None


def test_real_catalogue_native_action_and_copy_keep_the_selected_older_model(originals, qtbot, monkeypatch, tmp_path, record_property):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QStyle, QStyleOptionButton
    from darsia.fluidflower import native_phase_steps
    from darsia.fluidflower.ui import model_training

    data = originals
    newer = data.model.with_name("phase-model-newer")
    shutil.copytree(data.model, newer)
    summary = json.loads((newer / "summary.json").read_bytes())
    summary.update(model_id=newer.name, created_utc="2026-02-01T00:00:00+00:00")
    write(newer / "summary.json", summary)
    library.index_model(data.root, newer)
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_PHASE_MODEL_DEFAULTS", str(tmp_path / "no-defaults.json"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    for name in ("DARSIA_FLUIDFLOWER_PHASE_MODEL_LIBRARY", "DARSIA_NATIVE_CONFIGURATION",
                 "DARSIA_NATIVE_CONFIGURATION_SHA256"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(model_training, "QSettings", lambda *_args: QSettings(
        str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(model_training, "default_phase_review_roots", lambda: (data.root / "reviews",))
    studio = model_training.SharedPhaseModelTrainingPage()

    def readers_done():
        return (not studio._model_catalog_inflight and not studio._model_catalog_pending
                and not studio._review_sessions_loading and not studio._retraction_inflight
                and studio._pending_evidence_inflight is None)

    def finish(_widget):
        for timer in studio.findChildren(model_training.QTimer):
            timer.stop()
        qtbot.waitUntil(readers_done, timeout=10000)

    qtbot.addWidget(studio, before_close_func=finish)
    steps = native_phase_steps.NativePhaseSteps(studio, actor_provider=lambda: "reviewer")
    qtbot.addWidget(steps, before_close_func=finish)
    steps.set_context(data.root, "ffac")
    for timer in studio.findChildren(model_training.QTimer):
        timer.stop()
    steps.show_section("phase_model")
    steps.resize(1080, 700)
    steps.show()
    qtbot.waitUntil(readers_done, timeout=10000)
    assert studio.select_validation_model(data.model)
    selected_before = dict(studio._selected_validation_model())
    assert not selected_before["errors"]
    assert selected_before["model_id"] == data.model.name
    assert studio.validation_model.count() == 2
    identity_before = library.source_identity(data.model)
    method_before = read_series_method(data.root)
    withdrawals_before = withdrawals.load_retractions(data.root, series="ffac")
    steps.refresh()
    steps.phase_tabs.setCurrentWidget(steps.groups[0])
    qtbot.wait(20)
    assert steps.organise_button.isVisible() and steps.organise_button.isEnabled()
    assert steps.phase_workflow.archive_button.isVisible()
    assert steps.grab().save(str(tmp_path / "phase_model_basis.png"))
    # The original copy workflow opens inside the native working area.
    steps.organise_button.click()
    dialog = steps._layout_dialog
    qtbot.addWidget(dialog, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    assert dialog.isVisible() and dialog.plan is None
    assert not dialog.isWindow() and not dialog.isModal()
    assert steps.workspace_stack.currentWidget() is dialog
    qtbot.mouseClick(dialog.preview, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog.can_close() and dialog.plan is not None, timeout=10000)
    plan = dialog.plan
    original_files = {row["source"]["path"]: Path(row["source"]["path"]).read_bytes()
                      for row in plan["files"]}
    assert len(plan["model_checks"]) == 2 and dialog.table.model().rowCount() == len(plan["files"])
    source_index = dialog.table.model().index(0, 0)
    assert dialog.table.model().data(source_index) == Path(plan["files"][0]["source"]["path"]).relative_to(data.root).as_posix()
    assert dialog.table.model().data(source_index, Qt.ItemDataRole.ToolTipRole) == plan["files"][0]["source"]["path"]
    assert not (data.root / "ExpertKnowledge/PhaseModels").exists() and not dialog.copy.isEnabled()
    assert dialog.grab().save(str(tmp_path / "phase_model_layout_preview.png"))
    option = QStyleOptionButton()
    dialog.quiescent.initStyleOption(option)
    indicator = dialog.quiescent.style().subElementRect(
        QStyle.SubElement.SE_CheckBoxIndicator, option, dialog.quiescent)
    qtbot.mouseClick(dialog.quiescent, Qt.MouseButton.LeftButton, pos=indicator.center())
    assert dialog.quiescent.isChecked() and dialog.copy.isEnabled()
    qtbot.mouseClick(dialog.copy, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog.can_close() and dialog.plan is None, timeout=10000)
    qtbot.waitUntil(readers_done, timeout=10000)
    copied = review_storage.resolve_review_path(data.model)
    selected_after = studio._selected_validation_model()
    assert selected_after["root"] == str(copied)
    assert selected_after["model_id"] == selected_before["model_id"]
    assert not selected_after["errors"]
    assert library.source_identity(copied) == identity_before
    assert all(Path(path).read_bytes() == raw for path, raw in original_files.items())
    assert read_series_method(data.root) == method_before
    assert withdrawals.load_retractions(data.root, series="ffac") == withdrawals_before
    assert studio.validation_model.count() == 2
    assert dialog.grab().save(str(tmp_path / "phase_model_layout_completed.png"))
    record_property("private_gui_screenshots", str(tmp_path))


def test_actual_review_and_training_controls_fit_without_starting_jobs(originals, qtbot, monkeypatch, tmp_path, record_property):
    """Walk the actual compact pages using an isolated catalogue without running jobs."""
    from PySide6.QtCore import QPoint, QSettings
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QLabel, QScrollArea
    from darsia.fluidflower import native_phase_steps, phase_workflow
    from darsia.fluidflower.native_continuation_training import NativeContinuationTraining
    from darsia.fluidflower.pages.phase_interpretation.subpages.train_model import PhaseModelStudioSubpage
    from darsia.fluidflower.ui import model_training
    from darsia.fluidflower.ui.theme import apply_application_theme

    data = originals
    series_id = json.loads((data.model / "summary.json").read_bytes())["inventory"]["datasets"][0]
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_PHASE_MODEL_DEFAULTS", str(tmp_path / "no-defaults.json"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    for name in ("DARSIA_FLUIDFLOWER_PHASE_MODEL_LIBRARY", "DARSIA_NATIVE_CONFIGURATION",
                 "DARSIA_NATIVE_CONFIGURATION_SHA256"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(model_training, "QSettings", lambda *_args: QSettings(
        str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(model_training, "default_phase_review_roots", lambda: (data.root / "reviews",))
    monkeypatch.setattr(phase_workflow, "default_phase_review_roots", lambda: (data.root / "reviews",))
    studio = model_training.SharedPhaseModelTrainingPage()

    def readers_done():
        """Wait for the actual local catalogue and evidence readers to finish."""
        return (not studio._model_catalog_inflight and not studio._model_catalog_pending
                and not studio._review_sessions_loading and not studio._retraction_inflight
                and studio._pending_evidence_inflight is None)

    def finish(_widget):
        """Drain local readers before their Qt receivers are destroyed."""
        for timer in studio.findChildren(model_training.QTimer):
            timer.stop()
        qtbot.waitUntil(readers_done, timeout=10000)

    qtbot.addWidget(studio, before_close_func=finish)
    steps = native_phase_steps.NativePhaseSteps(studio, actor_provider=lambda: "reviewer")
    NativeContinuationTraining(steps)
    steps.setFont(QFont("Segoe UI", 10))
    apply_application_theme(steps)
    qtbot.addWidget(steps, before_close_func=finish)
    steps.set_context(data.root, series_id)
    for timer in studio.findChildren(model_training.QTimer):
        timer.stop()
    steps.show_section("phase_model")
    steps.resize(1280, 800)
    steps.show()
    qtbot.waitUntil(readers_done, timeout=10000)
    assert studio.select_validation_model(data.model)
    qtbot.waitUntil(readers_done, timeout=10000)
    assert not studio._selected_validation_model()["errors"]
    assert studio.review_sessions
    assert all(Path(row["session_path"]).is_relative_to(data.root) for row in studio.review_sessions)
    assert studio.saved_sessions.count() == 0
    assert not studio.add_selected_session_button.isEnabled()
    assert "Select a saved review session" in studio.add_selected_session_button.toolTip()
    assert studio.start_button.text() == "Train model version"
    assert studio.stop_button.text() == "Stop training now"
    steps.model_name.setText("Candidate version")
    assert not studio.start_button.isEnabled()
    assert not studio.stop_button.isEnabled()
    assert [scroll for scroll in steps.findChildren(QScrollArea) if scroll.isVisible()] == [steps.scroll]
    review_hint = next(label for label in steps.groups[3].findChildren(QLabel)
                       if label.text().startswith("Saved model maps and drawing tools"))
    assert review_hint.wordWrap()

    for number, controls in (
        (2, (studio.open_model_correction, steps.phase_workflow.archive_button)),
        (3, (studio.open_blind_labeling, review_hint)),
        (4, (steps.model_name, studio.start_button, studio.stop_button)),
    ):
        steps.phase_tabs.setCurrentWidget(steps.groups[number])
        qtbot.wait(20)
        assert steps.size().width() == 1280 and steps.size().height() == 800
        assert steps.scroll.horizontalScrollBar().maximum() == 0
        for control in controls:
            position = control.mapTo(steps.scroll.widget(), QPoint(0, 0))
            assert 0 <= position.x() <= steps.scroll.widget().width() - control.width()
            steps.scroll.ensureWidgetVisible(control)
            qtbot.wait(10)
            assert control.visibleRegion().boundingRect().height() > 0
        steps.scroll.verticalScrollBar().setValue(0)
        assert steps.grab().save(str(tmp_path / ("phase_model_step_" + str(number) + ".png")))

    steps.advanced.setChecked(True)
    qtbot.wait(20)
    assert steps.advanced.isHidden()
    assert not studio.add_selected_session_button.isVisible()
    assert not studio.add_selected_session_button.isEnabled()
    assert steps.scroll.horizontalScrollBar().maximum() == 0
    steps.scroll.ensureWidgetVisible(steps.phase_workflow.selected_evidence)
    qtbot.wait(20)
    assert steps.phase_workflow.selected_evidence.visibleRegion().boundingRect().height() > 0
    assert steps.grab().save(str(tmp_path / "phase_model_training_inputs.png"))
    assert studio.training_process is None and studio.validation_process is None
    assert PhaseModelStudioSubpage._active_job_notice(studio) is None
    assert all(Path(path).read_bytes() == raw for path, raw in data.before.items())
    record_property("private_gui_screenshots", str(tmp_path))
