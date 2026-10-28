"""Backend joins with a synthetic metadata-only display-provider double.

The genuine tiny Prepare/Check/Store paths remain active. These cases qualify
the glue, not the separate provider's camera-position or water mathematics.
No campaign inputs, actual human decisions, photographs or RAW are used.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import painted_analysis_exclusions as subject
from .test_geometry_registered_export_apply import registered_setup  # noqa: F401
from .test_painted_adoption_table import imported, prepared_painting, synthetic_payload_admission  # noqa: F401
from .test_painted_analysis_exclusions import ACTION


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()


@pytest.fixture
def display_provider(tmp_path, monkeypatch):
    """Explicit supplier double; validates a real bound source JSON, no arrays."""
    source = tmp_path / "synthetic-display-source.json"
    source.write_text('{"runs":["one","two"]}\n', encoding="utf8")
    context = dict(format="synthetic-display-context", source=subject._bound(source),
        declared_run_order=["one", "two"], anchor_metric_mm=[0., 0.], capture_name="initial")
    provider = ModuleType("darsia.fluidflower.painted_review_comparison")
    def validate(value, review=None):
        value = deepcopy(value)
        source_document = subject._checked(value["source"], metadata=True)
        if (value["format"] != "synthetic-display-context"
                or value["declared_run_order"] != source_document["runs"]):
            raise ValueError("The actual declared source order changed")
        if review is not None and any(row["run_id"] not in value["declared_run_order"] for row in review["runs"]):
            raise ValueError("A reviewed run is outside the actual declared order")
        return value
    def source_sha(value):
        value = validate(value)
        return _digest({key: value[key] for key in ("format", "source", "declared_run_order", "anchor_metric_mm")})
    def findings(review, attachments=None):
        actual = validate(review["display_context"], review)
        expected = dict(format="darsia-painted-review-display-attachments-v1",
            review=deepcopy(review["review_binding"]), display_context_sha256=source_sha(actual),
            position=dict(method="camera registration (map placement)", shift_tolerance_mm=2., angle_tolerance_degrees=.5),
            water=dict(height_tolerance_mm=3., angle_tolerance_degrees=.5, actions={}))
        value = expected if attachments is None else deepcopy(attachments)
        if (set(value) != set(expected) or value["format"] != expected["format"]
                or value["review"] != expected["review"]
                or value["display_context_sha256"] != expected["display_context_sha256"]):
            raise ValueError("The actual review/source binding differs")
        if value["water"]["actions"]:
            raise ValueError("A genuine clearwater water-line basis is unavailable")
        issues = [dict(severity="info", code="synthetic_clearwater_unavailable",
                       message="No human water action or physical verdict was generated")]
        if value["position"]["shift_tolerance_mm"] < 3.:
            issues.append(dict(severity="warning", code="synthetic_camera_registration",
                message="Synthetic metadata advisory", run_id="two", source_sha256=source_sha(actual)))
        return dict(display_attachments=value, issues=issues,
            display_findings_sha256=_digest(dict(display_attachments=value, issues=issues)))
    provider.validate_display_context = validate
    provider.context_source_sha256 = source_sha
    provider.display_findings = findings
    provider.display_context_digest = _digest
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    return SimpleNamespace(context=context, module=provider, source=source)


@pytest.fixture
def displayed_case(registered_setup, tmp_path, monkeypatch, display_provider):
    original = adoption.prepare_adoption
    def reviewed(*args, **kwargs):
        return original(*args, **dict(kwargs, display_context=display_provider.context, review_findings=True))
    monkeypatch.setattr(adoption, "prepare_adoption", reviewed)
    prepared, _, _, _, _ = prepared_painting(registered_setup, tmp_path, continuation=True)
    review = subject.read_gap_review(prepared["gap_review"]["path"])
    declaration = subject.make_facies_source_acquisition(prepared, "indicator", **ACTION)
    candidate, _ = adoption.adoption_candidate(registered_setup.revision, prepared["folder"],
        facies_source_acquisition=declaration)
    report = geometry.validate_geometry_config(candidate)
    assert not [issue for issue in report["issues"] if issue["severity"] == "error"]
    return SimpleNamespace(data=registered_setup, prepared=prepared, review=review,
        candidate=candidate, report=report, provider=display_provider)


def _records(report):
    return dict(warnings=[dict(issue_index=index, issue_sha256=subject.issue_identity(issue), **ACTION)
        for index, issue in enumerate(report["issues"]) if issue["severity"] == "warning"],
        acquisition=[dict(acquisition_state="indicator", **ACTION)], analysis_exclusions=None)


def test_historical_no_context_seal_shape_never_imports_display_provider(monkeypatch):
    bomb = ModuleType("darsia.fluidflower.painted_review_comparison")
    def fail_api(name):
        if name in {"validate_display_context", "context_source_sha256", "display_findings", "display_context_digest"}:
            pytest.fail("Historical None path imported the display provider")
        raise AttributeError(name)
    bomb.__getattr__ = fail_api
    monkeypatch.setitem(sys.modules, bomb.__name__, bomb)
    candidate = dict(run_geometry={})
    report = dict(config_sha256=geometry.geometry_config_digest(candidate), issues=[])
    sealed = subject.seal_findings_resolutions(candidate, report,
        dict(warnings=[], acquisition=[], analysis_exclusions=None))
    assert set(sealed) == {"format", "candidate_sha256", "issues_sha256", "records"}
    assert subject.verify_findings_resolutions(candidate, report, sealed) == sealed


def test_real_prepare_check_store_binds_compact_default_attachment_without_decoding_in_glue(displayed_case, monkeypatch):
    case = displayed_case
    assert case.prepared["display_context"] == case.provider.context == case.review["display_context"]
    assert all(row["gap"]["pixels"] == 0 for row in case.review["runs"])
    assert "analysis_exclusions" not in case.candidate and "target_corrections" not in case.candidate
    with pytest.MonkeyPatch.context() as metadata_only:
        metadata_only.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Display glue decoded an array"))
        findings = subject.check_review_findings(case.candidate)
        sealed = subject.seal_findings_resolutions(case.candidate, case.report, _records(case.report))
    assert findings["display_findings_sha256"] == case.report["display_findings_sha256"]
    assert findings["display_attachments"] == case.report["display_attachments"]
    assert set(sealed["records"]) == {"warnings", "acquisition", "analysis_exclusions"}
    assert set(sealed["display_attachments"]) == {"format", "review", "display_context_sha256", "position", "water"}
    assert sealed["display_attachments"]["water"]["actions"] == {}
    assert any(issue["code"] == "indicator_facies_carryover_risk" for issue in case.report["issues"])
    saved = geometry.GeometryRevisionStore(case.data.project).save_revision(case.candidate,
        actor="synthetic", reason="Explicit synthetic display findings", validation=case.report,
        review_resolutions=sealed, preserve_failed_artifacts=True)
    assert saved["review_resolutions"]["input"] == sealed
    materialized = saved["review_resolutions"]["materialized"]
    assert materialized["records"] == sealed["records"]
    assert materialized["display_attachments"] == sealed["display_attachments"]
    assert materialized["display_findings_sha256"] == sealed["display_findings_sha256"]
    assert "display_attachments" not in saved["configuration"]


def test_preference_change_requires_fresh_check_and_warning_resolutions(displayed_case):
    case = displayed_case
    sealed = subject.seal_findings_resolutions(case.candidate, case.report, _records(case.report))
    changed = deepcopy(sealed["display_attachments"])
    changed["position"]["shift_tolerance_mm"] = 99.
    with pytest.raises(ValueError, match="since Check"):
        subject.seal_findings_resolutions(case.candidate, case.report, _records(case.report), display_attachments=changed)
    fresh = geometry.validate_geometry_config(case.candidate, display_attachments=changed)
    assert fresh["display_findings_sha256"] != case.report["display_findings_sha256"]
    with pytest.raises(ValueError):
        subject.verify_findings_resolutions(case.candidate, fresh, sealed)
    current = subject.seal_findings_resolutions(case.candidate, fresh, _records(fresh), display_attachments=changed)
    assert current["display_attachments"] == changed
    assert current["records"]["acquisition"] == sealed["records"]["acquisition"]


def test_unavailable_water_action_is_unapprovable_and_original_findings_survive(displayed_case):
    case = displayed_case
    attachments = subject.check_review_findings(case.candidate)["display_attachments"]
    attachments["water"]["actions"]["one"] = dict(run_id="one", action="untrusted supplied action")
    original = case.review["issues"]
    assert original, "This real two-run fixture must contain an original frame disagreement"
    report = geometry.validate_geometry_config(case.candidate, display_attachments=attachments)
    assert report["summary"]["error_count"] > 0
    assert all(issue in report["issues"] for issue in original)
    assert any(issue["code"] == "painted_findings_changed" for issue in report["issues"])
    with pytest.raises(ValueError, match="Errors cannot"):
        subject.seal_findings_resolutions(case.candidate, report, _records(report), display_attachments=attachments)


@pytest.mark.parametrize("fault", ["context_validation", "missing_provider"])
def test_display_provider_failure_preserves_verified_original_issues_and_refuses_approval(displayed_case, monkeypatch, fault):
    case = displayed_case
    original = case.review["issues"]
    assert original, "The genuine tiny review must contain original frame findings"
    if fault == "context_validation":
        def stale_context(*args, **kwargs):
            raise ValueError("Synthetic actual display context is no longer available")
        monkeypatch.setattr(case.provider.module, "validate_display_context", stale_context)
    else:
        monkeypatch.setitem(sys.modules, case.provider.module.__name__, None)
    # Public native-approval/UI entrypoints still validate the positive context.
    with pytest.raises(subject.DisplayFindingsChanged) as failure:
        subject.read_gap_review(case.prepared["gap_review"]["path"])
    assert failure.value.original_issues == original
    report = geometry.validate_geometry_config(case.candidate)
    assert all(issue in report["issues"] for issue in original)
    assert report["summary"]["error_count"] > 0
    assert any(issue["severity"] == "error" and issue["code"] == "painted_findings_changed"
               for issue in report["issues"])
    assert "display_attachments" not in report and "display_findings_sha256" not in report
    with pytest.raises(ValueError, match="Errors cannot"):
        subject.seal_findings_resolutions(case.candidate, report, _records(report))


def test_scientific_review_source_mismatch_fails_before_optional_provider(displayed_case, monkeypatch, tmp_path):
    case = displayed_case
    candidate = deepcopy(case.candidate)
    table = candidate["painted_adoption_table"]
    preparation = subject._checked(table["preparation"], metadata=True)
    preparation["reference_run"] = "foreign-scientific-source"
    foreign = tmp_path / "foreign-preparation.json"
    foreign.write_text(json.dumps(preparation, allow_nan=False), encoding="utf8")
    table["preparation"] = subject._bound(foreign)
    monkeypatch.setattr(case.provider.module, "validate_display_context",
        lambda *args, **kwargs: pytest.fail("Scientific source mismatch reached the optional provider"))
    with pytest.raises(ValueError, match="another approved source or geometry") as failure:
        subject.check_review_findings(candidate)
    assert not isinstance(failure.value, subject.DisplayFindingsChanged)


@pytest.mark.parametrize("fault", ["missing_sibling", "changed_tolerance", "foreign_review"])
def test_store_rechecks_actual_attachments_before_any_publication(displayed_case, fault):
    case = displayed_case
    sealed = subject.seal_findings_resolutions(case.candidate, case.report, _records(case.report))
    if fault == "missing_sibling":
        sealed.pop("display_findings_sha256")
    elif fault == "changed_tolerance":
        sealed["display_attachments"]["position"]["shift_tolerance_mm"] = 99.
    else:
        sealed["display_attachments"]["review"]["sha256"] = "0" * 64
    store = geometry.GeometryRevisionStore(case.data.project)
    before, names = store.current_path.read_bytes(), sorted(path.name for path in store.artifact_root.iterdir())
    with pytest.raises(ValueError):
        store.save_revision(case.candidate, actor="synthetic", reason="Must refuse changed display evidence",
            validation=case.report, review_resolutions=sealed, preserve_failed_artifacts=True)
    assert store.current_path.read_bytes() == before
    assert sorted(path.name for path in store.artifact_root.iterdir()) == names


def test_source_identity_ignores_capture_metadata_but_refuses_changed_actual_source(display_provider, tmp_path):
    first = display_provider.context
    second = dict(deepcopy(first), capture_name="fresh-folder-and-time")
    assert subject._same_display_source(first, second)
    changed = tmp_path / "changed-source.json"
    changed.write_text('{"runs":["one","two"],"changed_scientific_source":true}\n', encoding="utf8")
    second["source"] = subject._bound(changed)
    assert not subject._same_display_source(first, second)
    assert subject._same_display_source(None, None)
    assert not subject._same_display_source(first, None)


def test_positive_display_context_cannot_masquerade_as_an_unsupported_native_review(display_provider, tmp_path):
    from darsia.fluidflower import map_placement
    with pytest.raises(ValueError, match="genuine native"):
        map_placement.place_map_in_runs(tmp_path / "not-created", np.ones((2, 3), np.uint8),
            reference_run="one", geometry_index=tmp_path / "absent.json", runs=["one"],
            photo_loader=lambda *args: pytest.fail("Unsupported review read a photograph"),
            roles={1: "porous_medium"}, version={}, display_context=display_provider.context)
    with pytest.raises(ValueError, match="genuine saved V2"):
        adoption.prepare_adoption(tmp_path / "also-not-created", np.ones((2, 3), np.uint8),
            reference_run="one", geometry_index=tmp_path / "absent.json", runs=["one"],
            roles={1: "porous_medium"}, version={}, display_context=display_provider.context)
    assert not (tmp_path / "not-created").exists() and not (tmp_path / "also-not-created").exists()


@pytest.mark.parametrize("kind", ["analysis_exclusions", "target_corrections"])
def test_new_display_subtree_cannot_add_scientific_dependencies_but_all_legacy_pins_remain(tmp_path, monkeypatch, kind):
    """Collector boundary double: no path-based TOML exemption or legacy bypass."""
    from darsia.fluidflower import painted_target_integration as target
    def pin(name):
        return dict(path=str(tmp_path / name), sha256="1" * 64, size_bytes=1)
    review = dict(review_binding=pin("review.json"), original_product=pin("original.npz"),
        original_provenance=dict(configuration=pin("legacy-still-required.toml")),
        display_context=dict(declared_runs=dict(configuration=pin("display-only.toml")),
                             protocol=pin("display-only-protocol.csv")))
    version = dict(basis=dict(review=pin("review.json")), product=pin("paint.npz"))
    seen = set()
    def checked(bound, *, metadata=False):
        name = Path(bound["path"]).name
        assert not name.startswith("display-only"), "Display metadata became a scientific live-file pin"
        seen.add(name)
        if metadata:
            return deepcopy(version if name == "version.json" else dict(review=pin("review.json")))
        return deepcopy(bound)
    monkeypatch.setattr(subject, "_checked", checked)
    monkeypatch.setattr(subject, "read_gap_review", lambda path: deepcopy(review))
    if kind == "analysis_exclusions":
        preparation = dict(analysis_exclusions=dict(approval=pin("gap-approval.json"),
            runs=dict(one=dict(artifact=pin("mask.npz")))))
        result = subject.exclusion_sources(preparation)
        assert "mask.npz" in seen
    else:
        preparation = dict(target_corrections=dict(bindings=[dict(version=pin("version.json"), approval=pin("approval.json"))]))
        result = target.target_sources(preparation)
        assert "paint.npz" in seen
    assert {"review.json", "original.npz", "legacy-still-required.toml"} <= seen
    assert result and all(not Path(bound["path"]).name.startswith("display-only") for bound in result)
