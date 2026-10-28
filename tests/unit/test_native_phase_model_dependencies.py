"""Explicit metadata dependency reporting; no deletion or scientific execution."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import native_phase_model_dependencies as dependencies


@pytest.fixture
def scope(tmp_path, monkeypatch):
    from darsia.fluidflower import storage_paths

    monkeypatch.setattr(storage_paths, "storage_path_remapper", lambda: lambda value: str(value))
    root = tmp_path / "model-a"
    # Model bytes deliberately need not exist: inspection only opens explicit
    # metadata records and reports the cached identity as unverified.
    entry = dict(root=str(root), model_id="model-a", library_source_identity=dict(
        model_id="model-a", version_sha256="a" * 64,
        metadata_sha256={"training_request.json": "b" * 64}, complete_heads=True))
    return tmp_path, root, entry


def write(scope, name, value, kind, **extra):
    path = scope[0] / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return dict(kind=kind, path=str(path), **extra)


def inspect(scope, records=(), **kwargs):
    return dependencies.inspect_model_dependencies(scope[2], scope_records=records,
                                                   series_root=scope[0], **kwargs)


def test_empty_scope_never_means_unused_or_deletable(scope):
    result = inspect(scope)
    assert result["dependency_status"] == "unknown"
    assert result["deletion_allowed"] is False
    assert result["coverage"]["complete"] is False
    assert result["coverage"]["requested_files"] == 0
    assert any(row["code"] == "incomplete_scope" for row in result["unknown"])
    assert not scope[1].exists()


def test_original_continuation_and_frozen_evidence_have_source_provenance(scope):
    root = scope[1]
    record = write(scope, "child-training.json", dict(
        format="darsia-fluidflower-phase-training-request-v1", model_id="child",
        continuation=dict(contract="darsia-phase-continuation-v1", parent_model_id="model-a",
            parent_request=dict(path=str(root / "training_request.json"), sha256="b" * 64),
            parent_heads={"co2_presence": dict(path=str(root / "models/generic/co2_presence_signal.json"), sha256="c" * 64)}),
        bundles=[dict(decisions=dict(source_path=str(root / "frozen/decisions.json"), sha256="d" * 64),
                      combined_manifest=dict(snapshot_path=str(scope[0] / "child/manifest.json")))]), "training_request")
    result = inspect(scope, [record])
    refs = result["refs"]
    assert result["dependency_status"] == "referenced"
    assert {row["relation"] for row in refs} >= {"continued from model", "saved parent head", "retained training evidence"}
    digest = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    assert all(row["source_sha256"] == digest for row in refs)
    assert all(row["path"] == record["path"].replace("\\", "/") for row in refs)
    assert not root.exists()


def test_library_target_and_previous_target_are_retained(scope):
    record = write(scope, "library.json", dict(format="phase-model-library-reference-v1",
        source_identity=scope[2]["library_source_identity"],
        target=dict(path="model-a", relative_to="series_root"),
        previous_targets=[dict(path=str(scope[1] / "old"), relative_to="absolute")]), "library_reference")
    result = inspect(scope, [record])
    assert {row["relation"] for row in result["refs"]} >= {"library target", "historical library target"}
    assert any(row["match"] == "recorded_version" for row in result["refs"])
    assert any(row["code"] == "model_identity_unverified" for row in result["unknown"])


def test_derived_training_paths_keep_their_own_recorded_digests(scope):
    root = scope[1]
    decisions = dict(snapshot_path=str(root / "filtered-decisions.json"), sha256="1" * 64,
        source_path=str(root / "live-decisions.json"), source_sha256="2" * 64,
        original_snapshot_path=str(root / "original-decisions.json"), original_snapshot_sha256="3" * 64)
    # Legacy metadata can record a source path without its digest. A frozen
    # snapshot's digest must never be substituted for that unknown identity.
    combined = dict(snapshot_path=str(root / "frozen-manifest.json"), sha256="4" * 64,
        source_path=str(root / "live-manifest.json"),
        original_snapshot_path=str(root / "original-manifest.json"))
    record = write(scope, "derived-training.json", dict(
        format="darsia-fluidflower-phase-training-request-v1",
        bundles=[dict(decisions=decisions, combined_manifest=combined)]), "training_request")
    result = inspect(scope, [record])
    digests = {row["field"]: row["recorded_sha256"] for row in result["refs"]}
    assert digests == {
        "bundles[0].decisions.snapshot_path": "1" * 64,
        "bundles[0].decisions.source_path": "2" * 64,
        "bundles[0].decisions.original_snapshot_path": "3" * 64,
        "bundles[0].combined_manifest.snapshot_path": "4" * 64,
        "bundles[0].combined_manifest.source_path": None,
        "bundles[0].combined_manifest.original_snapshot_path": None,
    }
    assert result["sources"][0]["status"] == "read"
    assert result["deletion_allowed"] is False and result["coverage"]["complete"] is False


def test_discarded_review_still_records_historical_model_reference(scope):
    record = write(scope, "review.json", dict(format="fluidflower-local-review-session-v1",
        target_model_id="model-a", include_in_training=False, editing_locked=True,
        training_locks=[dict(model_id="model-a")]), "review_session")
    before = Path(record["path"]).read_bytes()
    result = inspect(scope, [record])
    assert {row["relation"] for row in result["refs"]} == {"retained review evidence", "review training lock"}
    assert Path(record["path"]).read_bytes() == before


def test_native_review_binding_is_exact_path_reference(scope):
    record = write(scope, "binding.json", dict(format="native-phase-review-source-binding-v1",
        model_root=str(scope[1]), target_model_id="model-a"), "review_source_binding")
    result = inspect(scope, [record])
    assert any(row["match"] == "path" and row["field"] == "model_root" for row in result["refs"])


def test_inactive_saved_series_model_still_retained(scope):
    path = scope[0] / "series.toml"
    path.write_text('[method]\nactive = "colour_path"\n[method.phase_model]\nmodel_version = "model-a"\n', encoding="utf-8")
    result = inspect(scope, [dict(kind="series_manifest", path=str(path))])
    assert any(row["relation"] == "saved series selection" for row in result["refs"])


def test_bare_saved_library_pointer_is_explicitly_unresolved_without_following_it(scope):
    path = scope[0] / "series.toml"
    path.write_text('[method]\nactive = "phase_model"\n[method.phase_model]\n'
                    'model_version = "model-a.json"\n', encoding="utf-8")
    result = inspect(scope, [dict(kind="series_manifest", path=str(path))])
    pointer = scope[0] / "model-a.json"
    unresolved = [row for row in result["unknown"] if row.get("target_path") == str(pointer).replace("\\", "/")]
    assert len(unresolved) == 1 and "not followed" in unresolved[0]["reason"]
    assert result["coverage"]["read_files"] == 1 and not pointer.exists()
    assert result["dependency_status"] == "unknown" and result["deletion_allowed"] is False


def test_published_mass_receipt_records_payload_without_opening_it(scope):
    source = scope[1] / "models/generic/co2_presence_signal.json"
    record = write(scope, "delivery.json", dict(format="studio-mass-delivery-v1",
        result=str(scope[0] / "missing-result.json"), bindings={str(source): dict(sha256="c" * 64, stamp=[3, 4])}),
        "studio_mass_receipt")
    result = inspect(scope, [record])
    assert result["coverage"]["read_files"] == 1
    assert any(row["recorded_sha256"] == "c" * 64 for row in result["refs"])
    assert not source.exists()
    assert any(row["code"] == "result_not_verified" for row in result["unknown"])


def test_original_result_source_map_keeps_exact_payload_digest(scope):
    record = write(scope, "sources.json", {str(scope[1] / "phase_physics.json"): "d" * 64}, "result_source_manifest")
    result = inspect(scope, [record])
    assert result["refs"][0]["recorded_sha256"] == "d" * 64
    assert result["refs"][0]["relation"] == "retained result source"


def catalog(scope, roots=True):
    reference = dict(root_alias="models", relative_path="model-a/summary.json", sha256="d" * 64, bytes=1, role="model")
    return write(scope, "catalog.json", dict(format="darsia-role-input-catalog-v1", complete=True,
        configuration=reference, inputs=[dict(reference=reference, uses=["phase.model.native.summary"])]),
        "job_input_catalog", **({"roots": {"models": str(scope[0])}} if roots else {}))


def test_original_job_catalog_maps_only_explicit_alias_roots(scope):
    result = inspect(scope, [catalog(scope)])
    assert any(row["relation"] == "recorded job input: phase.model.native.summary" for row in result["refs"])
    assert any(row["code"] == "job_retention_unknown" for row in result["unknown"])


def test_job_catalog_missing_alias_is_unknown_without_guessing(scope):
    result = inspect(scope, [catalog(scope, roots=False)])
    assert not result["refs"]
    assert any(row["code"] == "missing_root_alias" for row in result["unknown"])


def test_local_job_registry_never_claims_full_retention_coverage(scope):
    record = write(scope, "jobs.json", dict(format="darsia-native-jobs-registry-v1",
        jobs=[dict(root=str(scope[0] / "unopened-job"))], plans=[]), "job_registry")
    result = inspect(scope, [record])
    assert result["coverage"]["read_files"] == 1
    assert {row["code"] for row in result["unknown"]} >= {"local_registry_incomplete", "registered_job_not_inspected"}


@pytest.mark.parametrize("raw", ['{"format":"x","format":"y"}', '{"bad":NaN}', '{', '[]'])
def test_corrupt_documents_are_unknown_never_absence(scope, raw):
    path = scope[0] / "bad.json"
    path.write_text(raw, encoding="utf-8")
    result = inspect(scope, [dict(kind="review_session", path=str(path))])
    assert result["sources"][0]["status"] == "unknown"
    assert result["dependency_status"] == "unknown"
    assert result["deletion_allowed"] is False


def test_explicit_expected_sha_mismatch_refuses_parsing(scope):
    record = write(scope, "review.json", dict(format="fluidflower-local-review-session-v1", target_model_id="model-a"),
                   "review_session", sha256="0" * 64)
    result = inspect(scope, [record])
    assert not result["refs"]
    assert any("SHA-256" in row["reason"] for row in result["unknown"])


def test_file_and_byte_limits_keep_unknown_coverage(scope):
    first = write(scope, "one.json", dict(format="fluidflower-local-review-session-v1"), "review_session")
    second = dict(kind="review_session", path=str(scope[0] / "never-opened.json"))
    result = inspect(scope, [first, second], max_files=1)
    assert result["coverage"]["requested_files"] == 2
    assert result["coverage"]["read_files"] == 1
    assert any(row["code"] == "file_limit" for row in result["unknown"])
    result = inspect(scope, [first], max_file_bytes=8)
    assert result["coverage"]["read_files"] == 0
    assert result["sources"][0]["status"] == "unknown"


def test_cancel_before_read_returns_partial_unknown_and_opens_no_scope(scope):
    result = inspect(scope, [dict(kind="review_session", path=str(scope[0] / "never-opened.json"))], cancelled=lambda: True)
    assert result["coverage"]["cancelled"] is True
    assert result["coverage"]["read_files"] == 0
    assert result["dependency_status"] == "unknown"


def test_cancel_between_metadata_reads_keeps_first_reference(scope):
    record = write(scope, "first.json", dict(format="fluidflower-local-review-session-v1", target_model_id="model-a"),
                   "review_session")
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 4  # before the second file; first file has three checks

    result = inspect(scope, [record, dict(kind="review_session", path=str(scope[0] / "unopened.json"))], cancelled=cancelled)
    assert result["coverage"]["cancelled"] is True
    assert result["coverage"]["read_files"] == 1
    assert result["refs"][0]["model_id"] == "model-a"
    assert result["deletion_allowed"] is False


def test_concurrent_metadata_change_is_reported_unknown(scope, monkeypatch):
    record = write(scope, "racing.json", dict(format="fluidflower-local-review-session-v1", target_model_id="model-a"),
                   "review_session")
    original = dependencies.os.fstat
    calls = 0

    def changed(fd):
        nonlocal calls
        calls += 1
        info = original(fd)
        if calls != 2:
            return info
        return SimpleNamespace(st_mode=info.st_mode, st_dev=info.st_dev, st_ino=info.st_ino,
            st_size=info.st_size, st_mtime_ns=info.st_mtime_ns + 1)

    monkeypatch.setattr(dependencies.os, "fstat", changed)
    result = inspect(scope, [record])
    assert not result["refs"]
    assert any("changed while" in row["reason"] for row in result["unknown"])


def test_malformed_result_frame_does_not_escape_unknown_report(scope):
    record = write(scope, "broken-result.json", dict(complete=True, frames=[{}],
        selected_inputs=dict(candidate={}, frames=[1])), "studio_mass_result")
    result = inspect(scope, [record])
    assert result["sources"][0]["status"] == "unknown"
    assert result["deletion_allowed"] is False


def test_protected_evidence_overlap_blocks_even_an_empty_scope(scope):
    result = inspect(scope, protected_roots=[scope[1] / "validation_boxes"])
    assert any(row["relation"] == "protected evidence" for row in result["refs"])
    assert result["deletion_allowed"] is False


def test_configured_cross_host_storage_alias_is_applied_lexically(scope, monkeypatch):
    from darsia.fluidflower import storage_paths

    monkeypatch.setattr(storage_paths, "storage_path_remapper", lambda: lambda value:
        str(value).replace("Z:/retained-model", str(scope[1])))
    record = write(scope, "alias-binding.json", dict(format="native-phase-review-source-binding-v1",
        model_root="Z:/retained-model"), "review_source_binding")
    result = inspect(scope, [record])
    assert result["refs"][0]["match"] == "path"


def test_library_relative_traversal_is_refused(scope):
    record = write(scope, "bad-pointer.json", dict(format="phase-model-library-reference-v1",
        source_identity=dict(model_id="other"), target=dict(path="../model-a", relative_to="series_root")),
        "library_reference")
    result = inspect(scope, [record])
    assert result["sources"][0]["status"] == "unknown"
    assert not result["refs"]


def test_path_prefix_sibling_is_not_a_dependency(scope):
    record = write(scope, "sibling.json", dict(format="native-phase-review-source-binding-v1",
        model_root=str(scope[0] / "model-a-other")), "review_source_binding")
    result = inspect(scope, [record])
    assert not result["refs"]
    assert result["dependency_status"] == "unknown"
