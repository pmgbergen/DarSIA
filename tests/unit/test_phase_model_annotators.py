"""Choose whose annotations train a phase model ([settings] reviewers).

A small synthetic bundle holds two people's boxes and blind answers, plus each
person's withdrawals and reclassifications in the series ledger. Without a
reviewer setting every input must stay exactly as before; with one, only the
chosen people's evidence (and evidence actions) may reach the trainer.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from darsia.fluidflower import phase_review_retractions as withdrawals
from darsia.fluidflower import series_paths
from darsia.fluidflower.phase_training import scope, trainer
from darsia.fluidflower.phase_workflow import build_phase_training_command, prepare_phase_training_request

SERIES, KEY = "FF_TEST", "fftest"
LEGACY_INVENTORY_HEADER = ("region_index,region_id,dataset,run,frame_index,elapsed_h,phase_class,"
                           "confidence,source,reviewer,supervision_weight")


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _box(identifier, reviewer=None, reviewer_id=None, frame_index=0):
    # Distinct geometry per ID, so no two boxes are merged as one region.
    offset = int(hashlib.sha256(identifier.encode()).hexdigest()[:4], 16) / 0x10000 * 0.5
    item = {"id": identifier, "candidate": "A", "frame_index": frame_index, "elapsed_h": float(frame_index),
            "category": "confirmed_free_gas", "confidence": "high", "shape": "rectangle",
            "x": round(offset, 6), "y": 0.1, "width": 0.2, "height": 0.2,
            "updated_utc": "2026-01-01T00:00:00+00:00"}
    if reviewer is not None:
        item["reviewer"] = reviewer
    if reviewer_id is not None:
        item["reviewer_id"] = reviewer_id
    return item


def _answer(proposal, **fields):
    return {"proposal_id": proposal, "candidate": "A", "frame_index": 1, "elapsed_h": 1.0,
            "human_choice": "only_free_gas", "phase_class": "gas", "confidence": "high",
            "x": 0.5, "y": 0.5 if proposal == "q1" else 0.7, "width": 0.1, "height": 0.1, **fields}


def _event(event_id, actor, run, identifier, action="retract", replacement=None):
    event = {"format": withdrawals.EVENT_FORMAT, "event_id": event_id, "action": action,
             "created_utc": "2026-01-01T00:00:00+00:00", "actor": actor,
             "target": {"series": KEY, "run": run, "kind": "manual", "identifier": identifier}}
    if replacement is not None:
        event["replacement"] = replacement
    return event


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    """A registered series, a two-person review bundle and a two-person ledger."""
    for name in tuple(os.environ):
        if (name.startswith("DARSIA_") and name.endswith("_PROJECT_ROOT")) or name in {
                "DARSIA_FLUIDFLOWER_DATA_ROOT", "DARSIA_FLUIDFLOWER_REVIEW_ROOT", "DARSIA_REVIEW_ALIAS_FILES"}:
            monkeypatch.delenv(name)
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text(
        f'schema_version = 1\nseries_id = "{SERIES}"\nname = "Test"\n\n[paths]\nphase_model = "PhaseModel"\n',
        encoding="utf-8")
    series_paths.register_series(root)
    prepared = tmp_path / "prepared"
    natives = {}
    for run in ("r1", "r2"):
        natives[run] = _write(prepared / "native_frames" / run / "manifest.json", {
            "run": run, "dataset": KEY, "frame_count": 2, "canvas_wh": [16, 9],
            "content_box_xywh": [0, 0, 16, 9], "analysis_image_shape_hw": [9, 16],
            "frame_sources": [{"frame_index": index, "elapsed_h": float(index),
                               "source_name": f"{run}_{index}.JPG", "source_sha256": f"{index:064x}"}
                              for index in range(2)]})
    jobs = [{"run": run, "dataset": KEY, "candidate_id": "c1", "review_output_dir": str(prepared)}
            for run in ("r1", "r2")]
    manifest = _write(tmp_path / "inputs" / "manifest.json", {"jobs": jobs})
    decisions = _write(tmp_path / "inputs" / "decisions.json", {"decisions": {
        # Ada's record, with one box Bob added to the same document.
        "r1": {"reviewer": "Ada", "reviewer_id": "ada", "annotations": [
            _box("a1", "Ada", "ada"), _box("a2", "Ada", "ada"), _box("a3", "Ada", "ada"),
            _box("b1", "Bob", "bob", frame_index=1)],
            "phase_quiz_answers": [_answer("q1")]},
        # Bob's record from before reviewer IDs were saved: names only.
        "r2": {"reviewer": "Bob", "annotations": [_box("b2", "Bob")],
               "phase_quiz_answers": [_answer("q2")]},
    }})

    def replacement(identifier, actor, run, frame_index):
        native = natives[run]
        timeline = json.loads(native.read_text(encoding="utf-8"))
        return {"annotation": dict(_box(identifier, actor, frame_index=frame_index),
                                   category="confirmed_only_aqueous_co2"),
                "series": KEY, "run": run, "job": jobs[0],
                "native_manifest": {"path": str(native), "sha256": _sha(native)},
                "native_manifest_data": timeline,
                "combined_manifest": {"path": str(manifest), "sha256": _sha(manifest)},
                "combined_manifest_data": {"jobs": [job for job in jobs if job["run"] == run]},
                "source_photo": dict(timeline["frame_sources"][frame_index])}

    events = [
        _event("e-bob-withdraws-a1", "Bob", "r1", "a1"),
        _event("e-ada-withdraws-a3", "Ada", "r1", "a3"),
        _event("e-bob-reclassifies-a2", "Bob", "r1", "a2", "reclassify", replacement("bob-new", "Bob", "r1", 0)),
        _event("e-ada-reclassifies-b1", "Ada", "r1", "b1", "reclassify", replacement("ada-new", "Ada", "r1", 1)),
    ]
    ledger = root / "PhaseModel" / "Annotations" / "evidence_actions" / "events"
    for event in events:
        _write(ledger / f"{event['event_id']}.json", event)
    return NS(root=root.resolve(), tmp=tmp_path, decisions=decisions, manifest=manifest,
              bundles=[(decisions, manifest)], events=events)


def _prepare(evidence, settings, name="models"):
    return prepare_phase_training_request(
        output_parent=evidence.tmp / name, cache_parent=evidence.tmp / "cache" / name,
        bundles=evidence.bundles, actor="Ada", settings=settings)


def _snapshot_decisions(request):
    return json.loads(Path(request["bundles"][0]["decisions"]["snapshot_path"]).read_text(encoding="utf-8"))


def _ids(document):
    return {run: sorted(str(item.get("id") or item.get("proposal_id")) for field in
                        ("annotations", "phase_quiz_answers") for item in record.get(field) or [])
            for run, record in (document.get("decisions") or {}).items()}


def test_requested_reviewers_and_the_reviewer_of_each_item():
    assert scope.requested_reviewers({}) is None
    assert scope.requested_reviewers({"reviewers": []}) is None
    assert scope.requested_reviewers({"reviewers": ["Bob", "Ada Example", "bob"]}) == ("ada-example", "bob")
    assert scope.requested_reviewers({"reviewers": "Ada"}) == ("ada",)
    record = {"reviewer": "Ada", "reviewer_id": "ada"}
    assert scope.evidence_reviewer({"reviewer_id": "carl"}, record) == "carl"
    assert scope.evidence_reviewer({}, record) == "ada"
    assert scope.evidence_reviewer({"reviewer": "Dora D"}, {"reviewer": "Ada"}) == "dora-d"
    assert scope.evidence_reviewer({}, {"reviewer": "Ada"}) == "ada"
    assert scope.evidence_reviewer({}, {}) == "anonymous"
    from darsia.fluidflower.reviewer_preferences import reviewer_identifier
    assert reviewer_identifier(" Åsa  Bø ") == "sa-b" and reviewer_identifier("") == "anonymous"


def test_a_subset_keeps_only_that_persons_boxes_and_answers(evidence):
    source = json.loads(evidence.decisions.read_text(encoding="utf-8"))
    original = json.loads(json.dumps(source))
    assert scope.partition_reviewers(source, None) is source
    ada = scope.partition_reviewers(source, ("ada",))
    assert _ids(ada) == {"r1": ["a1", "a2", "a3", "q1"]}
    bob = scope.partition_reviewers(source, ("bob",))
    assert _ids(bob) == {"r1": ["b1"], "r2": ["b2", "q2"]}
    assert source == original
    # Historical {frame: verdict} maps follow the record's reviewer.
    verdicts = {"decisions": {"r1": {"reviewer": "Ada", "frame_choices": {"0": "acceptable"}}}}
    assert scope.partition_reviewers(verdicts, ("ada",)) == verdicts
    assert scope.partition_reviewers(verdicts, ("bob",)) == {"decisions": {}}


def test_ledger_events_and_replacements_by_other_actors_are_excluded(evidence):
    snapshot = withdrawals.load_retractions(series=SERIES)
    assert len(snapshot["events"]) == 4
    assert scope.partition_review_events(snapshot, None) is snapshot
    ada = scope.partition_review_events(snapshot, ("ada",))
    assert [event["event_id"] for event in ada["events"]] == ["e-ada-reclassifies-b1", "e-ada-withdraws-a3"]
    assert [event["replacement"]["annotation"]["id"] for event in withdrawals.active_replacements(ada)] == ["ada-new"]
    assert withdrawals.validate_snapshot(ada) == ada

    request = _prepare(evidence, {"reviewers": ["Ada"], "series": [SERIES]})
    output = Path(request["output_root"])
    saved = json.loads((output / "training_request.json").read_text(encoding="utf-8"))
    assert saved["settings"]["reviewers"] == ["ada"]
    assert [event["event_id"] for event in withdrawals.model_retractions(output)["events"]] == [
        "e-ada-reclassifies-b1", "e-ada-withdraws-a3"]
    assert sorted(path.name for path in (output / "input_snapshot" / "evidence_actions").iterdir()) == [
        "e-ada-reclassifies-b1"]
    # Bob's withdrawal and reclassification of Ada's boxes do not apply; Ada's own withdrawal does.
    assert _ids(_snapshot_decisions(request)) == {"r1": ["a1", "a2", "q1"]}
    assert [len(request["bundles"]), request["bundles"][0]["decisions"]["derived_reviewer_subset"]] == [2, ["ada"]]
    assert _ids(json.loads(Path(request["bundles"][1]["decisions"]["snapshot_path"]).read_text())) == {
        "r1": ["ada-new"]}

    pairs = [trainer.ReviewBundle(Path(row["decisions"]["snapshot_path"]), Path(row["combined_manifest"]["snapshot_path"]))
             for row in request["bundles"]]
    _sources, regions, _ = trainer._load_sources_and_regions(
        pairs, withdrawal_snapshot=withdrawals.model_retractions(output), reviewers=("ada",))
    assert sorted(region.region_id for region in regions) == [
        "r1:manual:a1", "r1:manual:a2", "r1:manual:ada-new", "r1:quiz:q1"]
    assert {region.reviewer_id for region in regions} == {"ada"}
    # A direct trainer call on the original files gives the same evidence.
    _sources, direct, _ = trainer._load_sources_and_regions(
        [trainer.ReviewBundle(evidence.decisions, evidence.manifest)],
        withdrawal_snapshot=withdrawals.load_retractions(series=SERIES), reviewers=("ada",))
    assert sorted(region.region_id for region in direct) == sorted(region.region_id for region in regions)
    with pytest.raises(ValueError, match="frozen evidence-action policy"):
        trainer._load_sources_and_regions([trainer.ReviewBundle(evidence.decisions, evidence.manifest)],
                                          reviewers=("ada",))


def test_reviewer_id_is_recorded_in_inventory_and_provenance(evidence, tmp_path):
    _sources, regions, _ = trainer._load_sources_and_regions(
        [trainer.ReviewBundle(evidence.decisions, evidence.manifest)],
        withdrawal_snapshot=withdrawals.empty_snapshot(KEY))
    by_id = {region.region_id: region for region in regions}
    assert by_id["r1:manual:a1"].reviewer_id == "ada" and by_id["r2:manual:b2"].reviewer_id == "bob"
    assert by_id["r2:quiz:q2"].reviewer == "Bob" and by_id["r2:quiz:q2"].reviewer_id == "bob"
    path = tmp_path / "scoped.csv"
    trainer._write_region_inventory(path, regions, reviewer_ids=True)
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    assert list(rows[0]) == [*LEGACY_INVENTORY_HEADER.split(",")[:10], "reviewer_id", "supervision_weight"]
    assert {row["region_id"]: row["reviewer_id"] for row in rows}["r1:manual:b1"] == "bob"
    assert trainer._reviewer_inventory(regions, ("ada", "bob")) == {
        "reviewer_scope": ["ada", "bob"], "reviewer_ids": {"ada": 4, "bob": 3}}
    assert trainer._reviewer_provenance(regions, ("ada", "bob")) == {
        "reviewer_scope": ["ada", "bob"], "reviewer_ids": ["ada", "bob"]}
    assert trainer._reviewer_inventory(regions, None) == trainer._reviewer_provenance(regions, None) == {}


def test_without_a_reviewer_setting_inputs_and_inventory_are_unchanged(evidence, tmp_path):
    # Without a ledger, today's request copies the reviewed files byte for byte.
    for path in (evidence.root / "PhaseModel" / "Annotations" / "evidence_actions" / "events").glob("*.json"):
        path.unlink()
    plain = _prepare(evidence, {})
    assert Path(plain["bundles"][0]["decisions"]["snapshot_path"]).read_bytes() == evidence.decisions.read_bytes()
    assert Path(plain["bundles"][0]["combined_manifest"]["snapshot_path"]).read_bytes() == evidence.manifest.read_bytes()
    assert "reviewers" not in plain["settings"]
    assert not any("derived_reviewer_subset" in row[role] for row in plain["bundles"]
                   for role in ("decisions", "combined_manifest"))
    command = build_phase_training_command(plain)
    assert "--reviewer" not in command
    assert trainer.build_parser().parse_args([]).reviewers is None
    _sources, regions, _ = trainer._load_sources_and_regions(
        [trainer.ReviewBundle(evidence.decisions, evidence.manifest)],
        withdrawal_snapshot=withdrawals.empty_snapshot(KEY))
    path = tmp_path / "region_inventory.csv"
    trainer._write_region_inventory(path, regions)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == LEGACY_INVENTORY_HEADER
    assert lines[1] == "0,r1:manual:a1,fftest,r1,0,0.0,gas,high,manual,Ada,1.0"
    # An empty list also means everyone: the derived inputs are those of no setting.
    for event in evidence.events:
        _write(evidence.root / "PhaseModel" / "Annotations" / "evidence_actions" / "events"
               / f"{event['event_id']}.json", event)
    absent, empty = _prepare(evidence, {}, "absent"), _prepare(evidence, {"reviewers": []}, "empty")
    for left, right in zip(absent["bundles"], empty["bundles"]):
        for role in ("decisions", "combined_manifest"):
            assert Path(left[role]["snapshot_path"]).read_bytes() == Path(right[role]["snapshot_path"]).read_bytes()
    assert [event["event_id"] for event in withdrawals.model_retractions(absent["output_root"])["events"]] == sorted(
        event["event_id"] for event in evidence.events)
    assert empty["settings"]["reviewers"] == []


def test_cli_option_and_request_field_round_trip(evidence):
    request = _prepare(evidence, {"reviewers": ["Bob", "Ada"]})
    command = build_phase_training_command(request)
    pairs = [command[index + 1] for index, value in enumerate(command) if value == "--reviewer"]
    assert pairs == ["ada", "bob"]
    stored = json.loads(Path(request["request_path"]).read_text(encoding="utf-8"))
    assert stored["settings"]["reviewers"] == ["ada", "bob"]
    arguments = trainer.build_parser().parse_args(
        [item for index, value in enumerate(command) if value == "--reviewer" for item in command[index:index + 2]])
    assert arguments.reviewers == stored["settings"]["reviewers"]


def test_frozen_annotation_set_refuses_a_reviewer_subset(evidence, tmp_path):
    frozen = {"v5_chemistry_fit_scope": {"excluded_runs": []}}
    with pytest.raises(ValueError, match="bound to a frozen annotation set"):
        _prepare(evidence, {"reviewers": ["Ada"], **frozen}, "refused")
    assert not (evidence.tmp / "refused").exists()
    assert scope.require_reviewer_scope_allowed(frozen) is None
    assert scope.require_reviewer_scope_allowed({"reviewers": ["ada"]}) is None
    with pytest.raises(ValueError, match="bound to a frozen annotation set"):
        build_phase_training_command({"settings": {"reviewers": ["ada"], "v5_review_fit_partition": {"x": 1}},
                                      "model_id": "m", "output_root": str(tmp_path), "cache_root": str(tmp_path),
                                      "bundles": []})
    output = _write(tmp_path / "frozen model" / "training_request.json", {"settings": frozen}).parent
    arguments = trainer.build_parser().parse_args([
        "--output-root", str(output), "--cache-root", str(tmp_path / "cache"),
        "--bundle", str(evidence.decisions), str(evidence.manifest), "--reviewer", "ada"])
    with pytest.raises(ValueError, match="bound to a frozen annotation set"):
        trainer.train_campaign(arguments)
    assert sorted(path.name for path in output.iterdir()) == ["training_request.json"]


@pytest.fixture
def studio(qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower.ui import model_training as module

    monkeypatch.setattr(module, "configured_series", lambda: ((KEY, "Test", tmp_path),))
    monkeypatch.setattr(module.SharedPhaseModelTrainingPage, "_start_review_session_discovery", lambda self: None)
    monkeypatch.setattr(module.SharedPhaseModelTrainingPage, "_request_async_model_refresh", lambda self, **kw: None)
    monkeypatch.setattr(module.SharedPhaseModelTrainingPage, "_session_matches_selected_model", lambda self, s: True)
    page = module.SharedPhaseModelTrainingPage()
    qtbot.addWidget(page)
    sessions = [
        {"reviewer": "Ada Example", "reviewer_id": "ada-example", "session_path": str(tmp_path / "s1"),
         "manual_regions": 2, "modified_timestamp": 3.0},
        {"reviewer": "Bob", "session_path": str(tmp_path / "s2"), "manual_regions": 1, "modified_timestamp": 2.0},
        {"reviewer": "Ada Example", "reviewer_id": "ada-example", "session_path": str(tmp_path / "s3"),
         "blind_region_labels": 4, "modified_timestamp": 1.0},
    ]
    page._install_review_sessions(sessions)
    return NS(page=page, module=module, sessions=sessions)


def test_annotators_list_defaults_to_everyone_filters_sessions_and_passes_the_choice(studio, monkeypatch, qtbot):
    from PySide6.QtCore import Qt

    page, module = studio.page, studio.module
    labels = [page.annotators.item(row).text() for row in range(page.annotators.count())]
    assert labels == ["Ada Example (ada-example)", "Bob (bob)"]
    assert all(page.annotators.item(row).checkState() == Qt.CheckState.Checked for row in range(2))
    assert page._selected_annotators() is None
    assert len(page.visible_review_sessions) == 3
    captured = []

    def capture(**kwargs):
        captured.append(kwargs["settings"])
        raise RuntimeError("stop after capturing the request settings")

    monkeypatch.setattr(module, "prepare_phase_training_request", capture)
    monkeypatch.setattr(module, "parse_training_bundles", lambda text: [])
    monkeypatch.setattr(module.QMessageBox, "critical", lambda *args: captured.append(args[-1]))
    page.training_series.setText(KEY)
    page.radiometry_mode.setCurrentIndex(page.radiometry_mode.findData("legacy"))
    page._start_training()
    assert isinstance(captured[0], dict) and "reviewers" not in captured[0]
    assert captured[1] == "stop after capturing the request settings"
    page.annotators.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert page._selected_annotators() == ["ada-example"]
    assert [row["session_path"] for row in page.visible_review_sessions] == [
        studio.sessions[0]["session_path"], studio.sessions[2]["session_path"]]
    page._start_training()
    assert captured[2]["reviewers"] == ["ada-example"]
    assert {key: value for key, value in captured[2].items() if key != "reviewers"} == captured[0]
    # A rediscovery keeps the choice; nobody ticked is refused before preparation.
    page._install_review_sessions(studio.sessions)
    assert page._selected_annotators() == ["ada-example"]
    page.annotators.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert page.visible_review_sessions == []
    page._start_training()
    assert captured[4:] == ["Tick at least one annotator before training"]


def test_frozen_route_is_refused_in_the_training_step(studio, monkeypatch, evidence):
    from PySide6.QtCore import Qt

    page, module = studio.page, studio.module
    messages, launches = [], []
    monkeypatch.setattr(module.QMessageBox, "critical", lambda *args: messages.append(args[-1]))
    monkeypatch.setattr(page, "_launch_training_job", lambda *args: launches.append(args))
    page._native_training_photometry = NS(
        prepare_settings=lambda settings, bundles: {**settings, "v5_chemistry_fit_scope": {"excluded_runs": []}})
    page.bundles.setPlainText(f"{evidence.decisions} | {evidence.manifest}")
    page.training_series.setText(KEY)
    page.radiometry_mode.setCurrentIndex(page.radiometry_mode.findData("legacy"))
    page.annotators.item(1).setCheckState(Qt.CheckState.Unchecked)
    page._start_training()
    assert messages == [scope.FROZEN_ANNOTATION_SET_REFUSAL] and launches == []


def test_native_training_step_shows_the_annotators_list(studio, qtbot, tmp_path):
    from darsia.fluidflower.native_phase_steps import NativePhaseSteps

    steps = NativePhaseSteps(studio.page, actor_provider=lambda: "local-test-user")
    qtbot.addWidget(steps)
    assert steps.groups[4].isAncestorOf(studio.page.annotators)
    assert steps.groups[4].isAncestorOf(studio.page.saved_sessions)
