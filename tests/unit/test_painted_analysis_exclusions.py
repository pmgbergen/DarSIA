"""Explicit synthetic gap/state/warning actions; no campaign DATA or GUI approval."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import corrected_setup as export
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import painted_analysis_exclusions as subject
from darsia.fluidflower.geometry.context import RigFrameContext

from .test_geometry_registered_export_apply import registered_setup  # noqa: F401
from .test_painted_adoption_table import imported, prepared_painting, synthetic_payload_admission  # noqa: F401


ACTION = dict(actor="synthetic human", approved_utc="2026-01-01T00:00:00Z",
              reason="Explicit test-only source/area review", decision_ref="synthetic-N438")


@pytest.fixture
def gap_case(registered_setup, tmp_path, monkeypatch):
    return _make_gap_case(registered_setup, tmp_path, monkeypatch)


def _make_gap_case(data, tmp_path, monkeypatch, **fixture_options):
    actual = adoption.prepare_adoption
    captured = {}
    def keep(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return actual(*args, **kwargs)
    monkeypatch.setattr(adoption, "prepare_adoption", keep)
    before = geometry.GeometryRevisionStore(data.project).current_path.read_bytes()
    with pytest.raises(subject.GapReviewRequired) as error:
        prepared_painting(data, tmp_path, outside=True, continuation=True, outside_targets=True, **fixture_options)
    assert geometry.GeometryRevisionStore(data.project).current_path.read_bytes() == before
    review = error.value.review
    assert not (Path(review["review_binding"]["path"]).parent / "index.json").exists()
    return SimpleNamespace(data=data, review=review, actual=actual, captured=captured, root=tmp_path)


@pytest.fixture
def previous_exclusion_case(imported, synthetic_payload_admission, tmp_path, monkeypatch):
    """Freeze synthetic old standalone metadata before any source SHA is captured."""
    import toml
    from . import test_geometry_registered_export_apply as registered
    old = tmp_path / "retained-previous-gap-approval.json"
    old.write_text('{"synthetic_previous_approval":true}\n', encoding="utf8")
    old_binding = subject._bound(old)
    original_dumps = toml.dumps
    def historical(document, *args, **kwargs):
        if isinstance(document, dict) and document.get("calibration") == {"marker": "unchanged"}:
            document = deepcopy(document)
            run = Path(document["data"]["results"]).name
            document.setdefault("rig", {})["analysis_exclusions"] = json.dumps(dict(
                format="darsia-approved-analysis-gap-native-v1", run_id=run,
                previous_synthetic_approval=old_binding), sort_keys=True, separators=(",", ":"))
        return original_dumps(document, *args, **kwargs)
    with pytest.MonkeyPatch.context() as initial_settings:
        initial_settings.setattr(toml, "dumps", historical)
        data = registered.registered_setup.__wrapped__(imported, monkeypatch)
    case = _make_gap_case(data, tmp_path, monkeypatch)
    case.old_artifact, case.old_bytes = old, old.read_bytes()
    return case


def _approve(review, *, previous=None, selection=None, **changes):
    keys = selection or [dict(run_id=row["run_id"], component_id=item["component_id"])
                        for row in review["runs"] for item in row["components"]]
    return subject.make_gap_approval(review, keys, classification="Annet",
        other_text="Synthetic unquantified remainder", previous=previous, **dict(ACTION, **changes))


def _fresh(case, approval):
    kwargs = dict(case.captured["kwargs"], analysis_exclusions=approval, review_findings=True)
    return case.actual(case.root / "fresh", *case.captured["args"][1:], **kwargs)


def test_new_review_writes_explicit_full_photo_detected_transfer_policy(gap_case):
    from darsia.fluidflower.painting_chart_transfer import fresh_transfer_policy, review_proposal_format, DETECTED_FORMAT
    review = gap_case.review
    assert {row['run_id'] for row in review['runs']} == {'one', 'two'}
    assert review['transfer_policy'] == fresh_transfer_policy()
    assert review_proposal_format(review) == DETECTED_FORMAT
    stored = json.loads(Path(review['review_binding']['path']).read_bytes())
    assert stored['transfer_policy'] == fresh_transfer_policy()


def _resolved(candidate, report, approval, state="indicator"):
    return subject.seal_findings_resolutions(candidate, report, dict(
        warnings=[dict(issue_index=index, issue_sha256=subject.issue_identity(issue), **ACTION)
                  for index, issue in enumerate(report["issues"]) if issue["severity"] == "warning"],
        acquisition=[dict(acquisition_state=state, **ACTION)], analysis_exclusions=approval))


@pytest.mark.parametrize('covered', ['some', 'all', 'new_gap'])
def test_chart_coverage_retains_original_area_approval_but_cannot_approve_new_gaps(gap_case, monkeypatch, covered):
    from darsia.fluidflower import painted_target_integration
    approval = _approve(gap_case.review)
    prepared = _fresh(gap_case, approval)
    original_product = painted_target_integration.effective_product
    prepared['chart_transfers'] = {row['run_id']: {} for row in prepared['runs']}
    def changed(preparation, folder, row):
        arrays, bound = original_product({key: value for key, value in preparation.items()
                                          if key != 'chart_transfers'}, folder, row)
        arrays = deepcopy(arrays)
        points = np.argwhere(arrays['gaps'])
        if covered == 'all':
            arrays['gaps'][:] = False
        elif covered == 'some' and len(points):
            arrays['gaps'][tuple(points[0])] = False
        elif covered == 'new_gap':
            points = np.argwhere(~arrays['gaps'])
            arrays['gaps'][tuple(points[0])] = True
        return arrays, bound
    monkeypatch.setattr(painted_target_integration, 'effective_product', changed)
    if covered == 'new_gap':
        with pytest.raises(ValueError, match='newly exposed'):
            subject.verify_analysis_exclusions(prepared, prepared['folder'], approval)
    else:
        masks = subject.verify_analysis_exclusions(prepared, prepared['folder'], approval)
        assert set(masks) == {row['run_id'] for row in approval['runs']}
        if covered == 'all':
            assert not any(mask.any() for mask in masks.values())


def test_explicit_components_keep_base_pixels_and_roundtrip_store_export(gap_case):
    case = gap_case
    approval = _approve(case.review)
    prepared = _fresh(case, approval)
    assert subject.read_gap_review(prepared["gap_review"]["path"])["runs"]
    for row in case.review["runs"]:
        with np.load(row["product"]["path"], allow_pickle=False) as old:
            with np.load(Path(prepared["folder"]) / row["run_id"] / "labels.npz", allow_pickle=False) as fresh:
                for key in row["scientific_arrays"]:
                    assert np.array_equal(old[key], fresh[key])
    declaration = subject.make_facies_source_acquisition(prepared, "indicator", **ACTION)
    candidate, _ = adoption.adoption_candidate(case.data.revision, prepared["folder"], facies_source_acquisition=declaration)
    report = geometry.validate_geometry_config(candidate)
    assert not [issue for issue in report["issues"] if issue["severity"] == "error"]
    assert [issue for issue in report["issues"] if issue["code"] == "indicator_facies_carryover_risk"]
    resolved = _resolved(candidate, report, approval)
    saved = geometry.GeometryRevisionStore(case.data.project).save_revision(candidate,
        actor="synthetic", reason="explicit synthetic findings", validation=report,
        review_resolutions=resolved, preserve_failed_artifacts=True)
    assert "review_resolutions" not in saved["configuration"]
    assert saved["review_resolutions"]["input"] == resolved
    assert saved["review_resolutions"]["materialized"]["records"] == resolved["records"]
    with pytest.MonkeyPatch.context() as metadata_only:
        checked = subject._checked
        def only_metadata(bound, *, metadata=False):
            # The typed review reader hashes this exact JSON before decoding.
            assert metadata or bound == case.review["review_binding"], "Native preclaim planning hashed a binary artifact"
            return checked(bound, metadata=metadata)
        metadata_only.setattr(subject, "_checked", only_metadata)
        metadata_only.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Preclaim metadata decoded an NPZ"))
        projected = subject.adopted_run_native_product(saved, "two", project_root=case.data.project.root)
        assert projected["analysis_exclusions"] is not None
        assert subject.native_analysis_exclusion_dependencies(projected)
    for run in ("one", "two"):
        exported = export.export_corrected_setup(case.data.project,
            reference=SimpleNamespace(run_id=run), actor="synthetic export")
        mask = subject.configuration_exclusion(candidate, run)
        if mask is not None:
            contract = exported.receipt["binding"]["analysis_exclusions"]
            import tomllib
            snippet = tomllib.loads((exported.destination / "native_setup.toml").read_text(encoding="utf8"))
            assert isinstance(snippet["rig"]["analysis_exclusions"], str)
            assert json.loads(snippet["rig"]["analysis_exclusions"]) == contract
            assert contract["identity"] == subject._identity(mask)
            assert contract["source_to_export_native_h"] == np.eye(3).tolist()
            assert contract["meaning"] == "outside-analysis-unquantified"
            with np.load(exported.destination / "native_facies.npz", allow_pickle=True) as facies:
                assert not facies["array"][mask].any()
            native = subject.read_native_analysis_exclusions(contract,
                actual_labels=geometry._load_array(exported.destination / "native_labels.npz"), run_id=run)
            assert np.array_equal(native["mask"], mask) and native["dependencies"]
            assert subject.adopted_run_analysis_exclusions(saved, run, project_root=case.data.project.root) == contract
            # The real selected native replay must not touch another run's
            # product or mask even though the bound metadata lists the cohort.
            other_products = {str((Path(prepared["folder"]) / item["labels"]["path"]).resolve())
                for item in prepared["runs"] if item["run_id"] != run}
            other_products.update(str(Path(item["artifact"]["path"]).resolve())
                for key, item in prepared["analysis_exclusions"]["runs"].items() if key != run)
            other_products.update(str(Path(item["product"]["path"]).resolve())
                for item in case.review["runs"] if item["run_id"] != run)
            from darsia.fluidflower import painted_adoption_table as table
            original_checked, original_load = table._checked, np.load
            actual_labels = geometry._load_array(exported.destination / "native_labels.npz")
            with pytest.MonkeyPatch.context() as selected_only:
                def checked_selected(bound, *, metadata=False):
                    assert str(Path(bound["path"]).resolve()) not in other_products, "Unrelated run artifact was hashed"
                    return original_checked(bound, metadata=metadata)
                def load_selected(path, *args, **kwargs):
                    if isinstance(path, (str, Path)):
                        assert str(Path(path).resolve()) not in other_products, "Unrelated run artifact was decoded"
                    return original_load(path, *args, **kwargs)
                selected_only.setattr(table, "_checked", checked_selected)
                selected_only.setattr(np, "load", load_selected)
                product = subject.read_native_product(exported.receipt["binding"]["native_product"],
                    actual_labels=actual_labels, run_id=run)
            assert np.array_equal(product["mask"], mask)
            assert mask.any()


def test_area_actions_are_pure_metadata_and_preserve_other_actions(gap_case, monkeypatch):
    review = gap_case.review
    monkeypatch.setattr(subject, "_checked", lambda *args, **kwargs: pytest.fail("GUI action creation performed I/O"))
    approval = _approve(review)
    again = _approve(review, previous=approval, reason="A later explicit reason")
    assert approval["runs"][0]["reason"] == ACTION["reason"]
    assert again["runs"][0]["reason"] == "A later explicit reason"
    assert approval["runs"][0]["classification"] == "other"
    assert approval["runs"][0]["component_id"]


def test_retained_legacy_class_preserves_exact_findings_store_and_export_records(gap_case):
    """Read literal historical choices without translating their bound proof bytes."""
    case = gap_case
    approval = _approve(case.review)
    for row in approval["runs"]:
        row["classification"] = "Annet"
    retained_choice = deepcopy(approval)
    prepared = _fresh(case, approval)
    binding = prepared["analysis_exclusions"]["approval"]
    path = Path(binding["path"])
    original_bytes = path.read_bytes()
    assert json.loads(original_bytes) == retained_choice
    assert subject._bound(path) == binding
    declaration = subject.make_facies_source_acquisition(prepared, "indicator", **ACTION)
    candidate, _ = adoption.adoption_candidate(case.data.revision, prepared["folder"],
                                               facies_source_acquisition=declaration)
    report = geometry.validate_geometry_config(candidate)
    assert not [issue for issue in report["issues"] if issue["severity"] == "error"]
    resolved = _resolved(candidate, report, retained_choice)
    assert resolved["records"]["analysis_exclusions"] == retained_choice

    translated = deepcopy(resolved)
    translated["records"]["analysis_exclusions"]["runs"][0]["classification"] = "other"
    with pytest.raises(ValueError, match="exact approved connected areas"):
        subject.verify_findings_resolutions(candidate, report, translated)
    assert subject.verify_findings_resolutions(candidate, report, resolved) == resolved
    saved = geometry.GeometryRevisionStore(case.data.project).save_revision(candidate,
        actor="synthetic", reason="Retain literal historical area choices", validation=report,
        review_resolutions=resolved, preserve_failed_artifacts=True)
    assert saved["review_resolutions"]["input"] == resolved
    assert saved["review_resolutions"]["materialized"]["records"]["analysis_exclusions"] == retained_choice
    display_run = saved["configuration"]["frame_binding"]["display_reference_run"]
    export.export_corrected_setup(case.data.project,
        reference=SimpleNamespace(run_id=display_run), actor="synthetic export")
    run = retained_choice["runs"][0]["run_id"]
    exported = export.export_corrected_setup(case.data.project,
        reference=SimpleNamespace(run_id=run), actor="synthetic export")
    assert exported.receipt["binding"]["analysis_exclusions"]["approval"] == binding
    assert subject._checked(binding, metadata=True) == retained_choice
    assert path.read_bytes() == original_bytes
    assert subject._bound(path) == binding
    assert approval == retained_choice


@pytest.mark.parametrize("change", ["no_reason", "no_class", "empty_other", "wrong_component", "row_drift"])
def test_incomplete_or_stale_area_actions_never_approve(gap_case, change):
    review = deepcopy(gap_case.review)
    keys = [dict(run_id=row["run_id"], component_id=item["component_id"])
            for row in review["runs"] for item in row["components"]]
    args = dict(ACTION, classification="Annet", other_text="Synthetic")
    if change == "no_reason":
        args["reason"] = ""
    elif change == "no_class":
        args["classification"] = None
    elif change == "empty_other":
        args["other_text"] = ""
    elif change == "wrong_component":
        keys[0]["component_id"] = "0" * 64
    else:
        review["runs"][0]["gap"]["pixels"] += 1
    with pytest.raises(ValueError):
        subject.make_gap_approval(review, keys, **args)


@pytest.mark.parametrize("field", ["array", "filled", "support", "inside", "continued", "effective_coverage", "gaps"])
def test_rebound_old_artifact_tampering_is_refused(gap_case, field):
    approval = _approve(gap_case.review)
    row = next(row for row in gap_case.review["runs"] if row["components"])
    path = Path(row["product"]["path"])
    with np.load(path, allow_pickle=True) as archive:
        values = {key: np.array(archive[key], copy=True) for key in archive.files}
    values[field].flat[0] = (int(values[field].flat[0]) + 1 if field == "array" else not values[field].flat[0])
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError):
        _fresh(gap_case, approval)


def test_inside_gap_cannot_save_without_all_findings_resolutions(gap_case):
    prepared = _fresh(gap_case, _approve(gap_case.review))
    candidate, _ = adoption.adoption_candidate(gap_case.data.revision, prepared["folder"])
    with pytest.raises(ValueError, match="complete explicit findings resolutions from Use the map in the series"):
        geometry.GeometryRevisionStore(gap_case.data.project).save_revision(candidate,
            actor="synthetic", reason="must refuse", preserve_failed_artifacts=True)


def test_zero_gap_review_requires_warning_closure_and_saves_explicit_gui_seal(
        registered_setup, tmp_path, monkeypatch):
    """Actual supported V2 route: no exclusion/edit key exists to trigger a gate."""
    original_prepare = adoption.prepare_adoption
    def reviewed(*args, **kwargs):
        return original_prepare(*args, **dict(kwargs, review_findings=True))
    monkeypatch.setattr(adoption, "prepare_adoption", reviewed)
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path, continuation=True)
    review = subject.read_gap_review(prepared["gap_review"]["path"])
    assert all(not row["components"] and row["gap"]["pixels"] == 0 for row in review["runs"])
    assert "analysis_exclusions" not in prepared and "target_corrections" not in prepared
    declaration = subject.make_facies_source_acquisition(prepared, "indicator", **ACTION)
    candidate, _ = adoption.adoption_candidate(data.revision, prepared["folder"],
        facies_source_acquisition=declaration)
    assert "analysis_exclusions" not in candidate and "target_corrections" not in candidate
    report = geometry.validate_geometry_config(candidate)
    assert not [issue for issue in report["issues"] if issue["severity"] == "error"]
    assert [issue for issue in report["issues"] if issue["code"] == "indicator_facies_carryover_risk"]
    store = geometry.GeometryRevisionStore(data.project)
    before_current = store.current_path.read_bytes()
    before_artifacts = sorted(path.name for path in store.artifact_root.iterdir())
    with pytest.raises(ValueError, match="complete explicit findings resolutions from Use the map in the series"):
        store.save_revision(candidate, actor="synthetic", reason="No warning actions",
            validation=report, preserve_failed_artifacts=True)
    assert store.current_path.read_bytes() == before_current
    assert sorted(path.name for path in store.artifact_root.iterdir()) == before_artifacts
    sealed = _resolved(candidate, report, None)
    saved = store.save_revision(candidate, actor="synthetic", reason="Explicit warnings-only review",
        validation=report, preserve_failed_artifacts=True, review_resolutions=sealed)
    assert saved["review_resolutions"]["input"] == sealed
    assert saved["review_resolutions"]["materialized"]["records"] == sealed["records"]
    assert saved["validation"]["issues"] == report["issues"]
    assert "review_resolutions" not in saved["configuration"]


def test_historical_preparation_without_review_keeps_warning_save_behavior(registered_setup, tmp_path):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path, continuation=True)
    assert "gap_review" not in prepared
    candidate, _ = adoption.adoption_candidate(data.revision, prepared["folder"])
    report = geometry.validate_geometry_config(candidate)
    assert not [issue for issue in report["issues"] if issue["severity"] == "error"]
    assert [issue for issue in report["issues"] if issue["severity"] == "warning"]
    saved = geometry.GeometryRevisionStore(data.project).save_revision(candidate,
        actor="synthetic", reason="Original historical warning path", validation=report,
        preserve_failed_artifacts=True)
    assert "review_resolutions" not in saved and "gap_review" not in prepared


def test_warning_resolutions_bind_duplicate_indices_errors_and_candidate():
    candidate = dict(run_geometry={})
    warning = dict(severity="warning", code="synthetic", message="Same literal warning")
    report = dict(config_sha256=geometry.geometry_config_digest(candidate), issues=[warning, deepcopy(warning)])
    records = dict(warnings=[dict(issue_index=index, issue_sha256=subject.issue_identity(warning), **ACTION) for index in (0, 1)],
                   acquisition=[], analysis_exclusions=None)
    sealed = subject.seal_findings_resolutions(candidate, report, records)
    assert subject.verify_findings_resolutions(candidate, report, sealed) == sealed
    bad = deepcopy(records)
    bad["warnings"][1]["issue_index"] = 0
    with pytest.raises(ValueError):
        subject.seal_findings_resolutions(candidate, report, bad)
    changed = deepcopy(report)
    changed["issues"][0]["severity"] = "error"
    with pytest.raises(ValueError, match="Errors cannot"):
        subject.seal_findings_resolutions(candidate, changed, records)
    with pytest.raises(ValueError):
        subject.verify_findings_resolutions(dict(candidate, width_m=2.), report, sealed)


@pytest.mark.parametrize("change", ["missing_choice", "other_state", "external_origin", "other_row"])
def test_warning_approval_cannot_replace_explicit_gui_source_choice(change):
    choice = dict(acquisition_state="indicator", **ACTION)
    candidate = dict(painted_adoption_table={"synthetic_source": True},
        run_geometry=dict(runs=[dict(run_id="one", acquisition_state="indicator", enabled=True)]),
        provenance=dict(facies_source_acquisition=dict(acquisition_state="indicator",
            decision=dict(ACTION, origin="explicit-step5-gui"))))
    records = dict(warnings=[], acquisition=[choice], analysis_exclusions=None)
    if change == "missing_choice":
        records["acquisition"] = []
    elif change == "other_state":
        records["acquisition"][0]["acquisition_state"] = "clearwater"
    elif change == "external_origin":
        candidate["provenance"]["facies_source_acquisition"]["decision"]["origin"] = "chat"
    else:
        candidate["run_geometry"]["runs"][0]["acquisition_state"] = "unknown"
    report = dict(config_sha256=geometry.geometry_config_digest(candidate), issues=[])
    with pytest.raises(ValueError, match="GUI"):
        subject.seal_findings_resolutions(candidate, report, records)


def test_phase_mask_preserves_physical_locations_and_original_buffers():
    shape = (2, 3)
    domain = np.ones(shape, bool)
    gap = np.zeros(shape, bool)
    gap[0, 1] = True
    area = np.arange(6, dtype=np.float32).reshape(shape) + 1
    original = RigFrameContext(domain, domain.copy(), np.zeros(shape, bool), np.zeros(shape, bool),
        np.zeros(shape, bool), {}, (0, 0), shape, (0, 0, 3, 2), area, {"original": True})
    frame = dict(context=original, untouched=area)
    assert subject.restrict_phase_domain(frame, None, None) is frame
    result = subject.restrict_phase_domain(frame, gap, {"explicit": "synthetic"})
    assert result["context"].pixel_area_mm2 is area and result["untouched"] is area
    assert np.array_equal(result["physical_rig_domain"], domain)
    assert not result["context"].rig_domain[gap].any()
    assert not result["context"].modelled[gap].any()
    assert original.rig_domain.all() and original.modelled.all()
    assert "approved_analysis_gap" not in original.parameters
    fixtures = deepcopy(original)
    fixtures.hardware_excluded[0, 1] = True
    with pytest.raises(ValueError, match="fixture exclusion"):
        subject.restrict_phase_domain(dict(context=fixtures), gap, {})


def test_connected_review_areas_use_exact_masks_and_eight_neighbours():
    mask = np.array([[1, 0, 0, 0], [0, 1, 0, 1]], bool)
    parts = list(subject._components(mask))
    assert [item["pixels"] for item, _ in parts] == [2, 1]
    assert np.array_equal(np.logical_or.reduce([value for _, value in parts]), mask)
    assert all(item["identity"] == subject._identity(value) for item, value in parts)


def _paint_component(case, *, run_id=None):
    from darsia.fluidflower.painted_target_corrections import TargetCorrectionDraft
    row = next(row for row in case.review["runs"] if row["components"] and (run_id is None or row["run_id"] == run_id))
    packet = subject.read_target_paint(case.review, row["run_id"], row["components"][0]["component_id"])
    assert packet["editable_mask"].dtype == bool
    assert packet["target_labels"].dtype == np.uint8
    assert all(item["role"] == "porous_medium" for item in packet["allowed_layers"])
    draft = TargetCorrectionDraft(labels=packet["target_labels"], editable_mask=packet["editable_mask"],
        allowed_ids=[item["layer_id"] for item in packet["allowed_layers"]], basis=packet["target_correction_basis"])
    for y, x in np.argwhere(packet["editable_mask"]):
        draft.paint("brush", [[int(x), int(y)]], packet["allowed_layers"][0]["layer_id"], radius_px=1)
    binding = draft.save_and_approve(case.root / "target_corrections", actor=ACTION["actor"],
        approved_utc=ACTION["approved_utc"], reason=ACTION["reason"])
    return row, packet, binding


def test_target_paint_replays_separately_and_keeps_every_original_base_field(gap_case):
    from darsia.fluidflower import painted_target_integration as target
    case = gap_case
    row, packet, binding = _paint_component(case)
    original_bytes = Path(row["product"]["path"]).read_bytes()
    kwargs = dict(case.captured["kwargs"], target_corrections=[binding], review_findings=True)
    prepared = case.actual(case.root / "fresh_target", *case.captured["args"][1:], **kwargs)
    entry = next(item for item in prepared["runs"] if item["run_id"] == row["run_id"])
    effective, artifact = target.effective_product(prepared, prepared["folder"], entry)
    base = subject._arrays(dict(entry["labels"], path=str(Path(prepared["folder"]) / entry["labels"]["path"])))
    assert Path(row["product"]["path"]).read_bytes() == original_bytes
    assert artifact["path"].endswith("target_labels.npz") and entry["labels"]["path"].endswith("labels.npz")
    for key in row["base_scientific_arrays"]:
        assert subject._identity(base[key]) == row["base_scientific_arrays"][key]
    for key in ("filled", "support", "inside", "continued"):
        assert np.array_equal(effective[key], base[key])
    assert np.array_equal(effective["array"][~packet["editable_mask"]], base["array"][~packet["editable_mask"]])
    assert not effective["gaps"].any() and not base["support"][packet["editable_mask"]].any()
    assert not base["array"][packet["editable_mask"]].any()
    assert "analysis_exclusions" not in prepared
    assert adoption.read_preparation(prepared["folder"])["target_corrections"] == prepared["target_corrections"]
    declaration = subject.make_facies_source_acquisition(prepared, "indicator", **ACTION)
    candidate, _ = adoption.adoption_candidate(case.data.revision, prepared["folder"], facies_source_acquisition=declaration)
    report = geometry.validate_geometry_config(candidate)
    assert not [issue for issue in report["issues"] if issue["severity"] == "error"]
    sealed = _resolved(candidate, report, None)
    revision = geometry.GeometryRevisionStore(case.data.project).save_revision(candidate,
        actor="synthetic", reason="explicit target correction", validation=report,
        review_resolutions=sealed, preserve_failed_artifacts=True)
    assert revision["configuration"]["target_corrections"] == prepared["target_corrections"]
    native_product = subject.adopted_run_native_product(revision, row["run_id"], project_root=case.data.project.root)
    assert native_product["analysis_exclusions"] is None and native_product["target_corrections"] == [binding]
    with pytest.MonkeyPatch.context() as metadata_only:
        metadata_only.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Painted-run planning decoded NPZ"))
        assert subject.adopted_run_native_product(revision, row["run_id"], project_root=case.data.project.root) == native_product
        assert subject.native_analysis_exclusion_dependencies(native_product)
    for run in ("one", "two"):
        exported = export.export_corrected_setup(case.data.project,
            reference=SimpleNamespace(run_id=run), actor="synthetic painted target export")
        assert "analysis_exclusions" not in exported.receipt["binding"]
        import tomllib
        snippet = tomllib.loads((exported.destination / "native_setup.toml").read_text(encoding="utf8"))
        assert isinstance(snippet["rig"]["native_product"], str)
        assert json.loads(snippet["rig"]["native_product"]) == exported.receipt["binding"]["native_product"]
        assert json.loads(snippet["rig"]["native_product"])["analysis_exclusions"] is None
        attempt = export.apply_corrected_selected_run(case.data.project, exported, rebuild_facies_by_name=True)
        for path in exported.receipt["binding"]["configuration_bundles"][run]:
            document = tomllib.loads(Path(path).read_text(encoding="utf8"))
            assert isinstance(document["rig"]["native_product"], str)
            assert json.loads(document["rig"]["native_product"]) == exported.receipt["binding"]["native_product"]
            assert json.loads(document["rig"]["native_product"])["analysis_exclusions"] is None
        assert export._apply_completion(exported.destination, exported.receipt)["complete"] is True
        assert export.apply_corrected_selected_run(case.data.project, exported, rebuild_facies_by_name=True) == attempt
        if run == row["run_id"]:
            assert np.array_equal(geometry._load_array(exported.destination / "native_labels.npz"), effective["array"])
    lost = deepcopy(candidate)
    lost.pop("target_corrections")
    with pytest.raises(ValueError, match="target corrections"):
        sources.canonical_parent_payload(lost)


def _adopt_target_fixture(case, bindings, name):
    kwargs = dict(case.captured["kwargs"], target_corrections=bindings, review_findings=True)
    prepared = case.actual(case.root / name, *case.captured["args"][1:], **kwargs)
    declaration = subject.make_facies_source_acquisition(prepared, "indicator", **ACTION)
    candidate, _ = adoption.adoption_candidate(case.data.revision, prepared["folder"],
        facies_source_acquisition=declaration)
    report = geometry.validate_geometry_config(candidate)
    assert not [issue for issue in report["issues"] if issue["severity"] == "error"]
    revision = geometry.GeometryRevisionStore(case.data.project).save_revision(candidate,
        actor="synthetic", reason="Explicit target test fixture", validation=report,
        review_resolutions=_resolved(candidate, report, None), preserve_failed_artifacts=True)
    return prepared, revision


def test_explicit_fully_painted_apply_retires_old_standalone_exclusion_and_replays_completion(previous_exclusion_case):
    """Real Store/export/Apply; only the old owned TOML value is retired."""
    import tomllib
    case = previous_exclusion_case
    _, _, binding = _paint_component(case)
    _, revision = _adopt_target_fixture(case, [binding], "target_after_previous_exclusion")
    for run in ("one", "two"):
        exported = export.export_corrected_setup(case.data.project,
            reference=SimpleNamespace(run_id=run), actor="synthetic transition")
        paths = exported.receipt["binding"]["configuration_bundles"][run]
        for path in paths:
            assert "analysis_exclusions" in tomllib.loads(Path(path).read_text(encoding="utf8"))["rig"]
        attempt = export.apply_corrected_selected_run(case.data.project, exported, rebuild_facies_by_name=True)
        for path in paths:
            rig = tomllib.loads(Path(path).read_text(encoding="utf8"))["rig"]
            assert "analysis_exclusions" not in rig
            assert json.loads(rig["native_product"])["analysis_exclusions"] is None
        complete = export._apply_completion(exported.destination, exported.receipt)
        assert complete["complete"] is True
        assert export.apply_corrected_selected_run(case.data.project, exported, rebuild_facies_by_name=True) == attempt
        assert case.old_artifact.read_bytes() == case.old_bytes
    assert "analysis_exclusions" not in revision["configuration"]


def test_two_target_runs_restore_reads_only_the_selected_version_and_approval(registered_setup, tmp_path, monkeypatch):
    """Full publication verifies both; selected native replay cannot open the other."""
    from darsia.fluidflower import painted_target_corrections as backend
    def full_own_frames(frames):
        for frame in frames.values():
            frame["corners"] = dict(top_left=[0., 0.], top_right=[0., 2.],
                                    bottom_right=[1., 2.], bottom_left=[1., 0.])
    case = _make_gap_case(registered_setup, tmp_path, monkeypatch, own_frame_change=full_own_frames)
    assert {row["run_id"] for row in case.review["runs"] if row["components"]} == {"one", "two"}
    bindings = {run: _paint_component(case, run_id=run)[2] for run in ("one", "two")}
    _, revision = _adopt_target_fixture(case, list(bindings.values()), "two_target_runs")
    for run in ("one", "two"):
        foreign = "two" if run == "one" else "one"
        forbidden = {str(Path(bound["path"]).resolve()) for bound in bindings[foreign].values()}
        contract = subject.adopted_run_native_product(revision, run, project_root=case.data.project.root)
        actual = geometry._load_array(Path(revision["artifacts"]["run_geometry"][run]["labels_path"]))
        original_common, original_backend = subject._checked, backend._checked
        with pytest.MonkeyPatch.context() as selected_only:
            def selected_common(bound, **kwargs):
                assert str(Path(bound["path"]).resolve()) not in forbidden, "Foreign target version/approval was read"
                return original_common(bound, **kwargs)
            def selected_backend(bound, **kwargs):
                assert str(Path(bound["path"]).resolve()) not in forbidden, "Foreign target version/approval was read"
                return original_backend(bound, **kwargs)
            selected_only.setattr(subject, "_checked", selected_common)
            selected_only.setattr(backend, "_checked", selected_backend)
            result = subject.read_native_product(contract, actual_labels=actual, run_id=run)
        assert result["mask"] is None and result["contract"]["target_corrections"] == [bindings[run]]


def test_target_correction_refuses_tampered_approval_and_duplicate_version(gap_case):
    _, _, binding = _paint_component(gap_case)
    bad = deepcopy(binding)
    bad["approval"]["sha256"] = "0" * 64
    kwargs = dict(gap_case.captured["kwargs"], target_corrections=[bad], review_findings=True)
    with pytest.raises(ValueError):
        gap_case.actual(gap_case.root / "bad_target", *gap_case.captured["args"][1:], **kwargs)
    kwargs["target_corrections"] = [binding, binding]
    with pytest.raises(ValueError, match="unique explicit"):
        gap_case.actual(gap_case.root / "duplicate_target", *gap_case.captured["args"][1:], **kwargs)


@pytest.mark.parametrize("denied", [None, "index.json", "nested.json", "labels.npz", "continuation.json"])
def test_native_dependency_admission_precedes_every_read_and_json_expansion(tmp_path, monkeypatch, denied):
    """Admit selected JSON before reads; never sweep unrelated RAW/archives."""
    from darsia.fluidflower import painted_native_product as native
    from darsia.fluidflower import photo_edge_continuation as continuation
    folder = tmp_path / "retained"
    def pin(name):
        return dict(path=str(folder / name), sha256="1" * 64, size_bytes=1)
    def relative(name):
        return dict(path=name, sha256="1" * 64, size_bytes=1)
    preparation = dict(runs=[dict(run_id="one", labels=relative("one/labels.npz"),
            reference_model=pin("nested.json"), target_model=pin("nested.json")),
        dict(run_id="unrelated", labels=relative("unrelated/labels.npz"),
             target_model=pin("unrelated/model.json"))],
        painting_facies=dict(files={"map.npz": pin("saved_map.npz")}), geometry_index=pin("geometry_index.json"),
        source_map=relative("source_map.npz"), continuation=dict(file="continuation.json", sha256="1" * 64, size_bytes=1))
    documents = {str(folder / "index.json"): preparation,
        str(folder / "nested.json"): dict(reference_image=pin("selected_raw.nef"),
            baseline_native_grid=dict(renderer=pin("renderer.json"),
                parent_grid_compatibility=dict(configuration=pin("historical.toml"))),
            unrelated_ancestry=pin("unrelated/large.npz"), baseline_configuration=pin("historical.toml")),
        str(folder / "renderer.json"): dict(corrected_baseline=dict(archive=pin("native_baseline.npz")),
            inputs=dict(raw=pin("selected_raw.nef"), profile=pin("profile.json"), operation_request=pin("request.json"),
                configuration_files=[pin("historical.toml")], series_context=dict(manifest=pin("series.toml")),
                protocol_inputs=[dict(source=pin("protocol.xlsx"))]),
            unused_ancestry=pin("unrelated/large.nef")),
        str(folder / "continuation.json"): dict(runs={"one": dict(
            source_model=dict(descriptor=pin("nested.json"), photograph=pin("selected_raw.nef"),
                document=dict(baseline_configuration=pin("historical.toml"))),
            target_model=dict(descriptor=pin("nested.json"), photograph=pin("selected_raw.nef")),
            frame_authority=dict(project=pin("historical-project.json"))),
            "unrelated": dict(model=pin("unrelated/model.json"), raw=pin("unrelated/large.nef"))})}
    admitted, reads = set(), []
    class Reader:
        def path(self, value):
            path = value["path"]
            assert "unrelated" not in Path(path).parts, "Catalogue traversed an unrelated run/ancestor"
            assert Path(path).name not in {"historical.toml", "series.toml", "protocol.xlsx", "historical-project.json"}, (
                "Passive historical configuration provenance became a restore-time file dependency")
            if Path(path).name == denied:
                raise PermissionError("Synthetic original-root admission refused")
            admitted.add(path)
            return Path(path)
    def checked(value, *, metadata=False):
        assert value["path"] in admitted, "A descriptor was read before original-root admission"
        assert metadata, "Catalogue hashed a binary descriptor instead of returning its literal identity"
        reads.append(value["path"])
        return deepcopy(documents[value["path"]])
    def read_contracts(path):
        assert str(path) in admitted, "Pooled sidecar was expanded before root admission"
        assert str(path) in reads, "Pooled sidecar was expanded before its metadata SHA check"
        return deepcopy(documents[str(path)])
    monkeypatch.setattr(subject, "_checked", checked)
    monkeypatch.setattr(continuation, "read_contracts", read_contracts)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Metadata admission decoded NPZ"))
    contract = dict(preparation=pin("index.json"), run_id="one", adopted_revision=pin("revision.json"),
        native_grid=dict(parent_grid_compatibility=dict(configuration=pin("historical.toml"))))
    if denied is None:
        result = native.native_analysis_exclusion_dependencies(contract, Reader())
        assert result and all(value["path"] in admitted for value in result)
        assert str(folder / "selected_raw.nef") in admitted and str(folder / "native_baseline.npz") in admitted
        assert str(folder / "request.json") in admitted and str(folder / "profile.json") in admitted
        assert str(folder / "revision.json") in admitted and str(folder / "revision.json") not in reads
        assert {Path(path).name for path in reads} == {"index.json", "continuation.json", "nested.json", "renderer.json"}
    else:
        with pytest.raises(PermissionError, match="original-root admission"):
            native.native_analysis_exclusion_dependencies(contract, Reader())
        assert all(Path(path).name != denied for path in reads)


def test_omitted_native_rig_keeps_original_four_line_bytes():
    before = (b'# original comments and CRLF\r\n[labeling]\r\nlabels = "old-labels"\r\n'
              b'frame = "camera"\r\n[facies]\r\npath = "old-facies"\r\nprops = "old-props"\r\n'
              b'[rig]\r\npath = "untouched-rig"\r\nanalysis_exclusions = "untouched-old-metadata"\r\n')
    replacements = dict(labeling=dict(labels="new-labels", frame="rig-corrected"),
                        facies=dict(path="new-facies", props="new-props"))
    expected = (b'# original comments and CRLF\r\n[labeling]\r\nlabels = "new-labels"\r\n'
                b'frame = "rig-corrected"\r\n[facies]\r\npath = "new-facies"\r\nprops = "new-props"\r\n'
                b'[rig]\r\npath = "untouched-rig"\r\nanalysis_exclusions = "untouched-old-metadata"\r\n')
    assert export.corrected_config_bytes(before, replacements) == expected
    assert export.corrected_config_bytes(before, replacements, native_rig=None) == expected


@pytest.mark.parametrize("with_exclusion", [False, True])
def test_raw_toml_native_contract_roundtrip_preserves_nulls_and_original_settings_bytes(tmp_path, with_exclusion):
    """Actual TOML encoding/parse; contracts are explicit synthetic literals."""
    import toml
    import tomllib
    from darsia.fluidflower import painted_setup_apply as rebuilt
    camera = dict(configuration=str(tmp_path / "camera.json"), sha256="1" * 64,
                  profile_id="explicit", run_id="one")
    settings = dict(run_id="one", raw_camera=camera, rig_path=str(tmp_path / "rig"))
    old_exclusion = '{"previous_synthetic_exclusion":true}'
    before = ('[data]\nuse_cache = true\n[rig]\npath = "old"\nanalysis_exclusions = '
              + json.dumps(old_exclusion) + '\n[extra]\nvalue = 2\n').encode("utf8")
    legacy_expected = toml.dumps(dict(data=dict(use_cache=False, raw_camera=camera),
        rig=dict(path=settings["rig_path"], analysis_exclusions=old_exclusion), extra=dict(value=2))).encode("utf8")
    assert rebuilt.raw_settings_for_run(settings, run_id="one") == settings
    assert rebuilt.raw_settings_bytes(before, settings) == legacy_expected
    exclusion = (dict(format="darsia-approved-analysis-gap-native-v1", run_id="one",
                      nullable_metadata=None) if with_exclusion else None)
    native = dict(format="darsia-painted-native-product-v1", run_id="one",
                  analysis_exclusions=exclusion, native_grid=dict(authority=None, shape_hw=[2, 3]),
                  source_to_export_native_h=np.eye(3).tolist())
    requested = dict(settings, native_product=native)
    if with_exclusion:
        requested["analysis_exclusions"] = exclusion
    assert rebuilt.raw_settings_for_run(requested, run_id="one") == requested
    encoded = rebuilt.raw_settings_bytes(before, requested)
    document = tomllib.loads(encoded.decode("utf8"))
    assert document["data"] == dict(use_cache=False, raw_camera=camera)
    assert document["extra"] == dict(value=2) and document["rig"]["path"] == settings["rig_path"]
    assert isinstance(document["rig"]["native_product"], str)
    assert export._typed_equal(json.loads(document["rig"]["native_product"]), native)
    assert json.loads(document["rig"]["native_product"])["native_grid"]["authority"] is None
    if with_exclusion:
        assert isinstance(document["rig"]["analysis_exclusions"], str)
        assert json.loads(document["rig"]["analysis_exclusions"]) == exclusion
    else:
        assert "analysis_exclusions" not in document["rig"]
    assert rebuilt.raw_settings_bytes(encoded, requested) == encoded
