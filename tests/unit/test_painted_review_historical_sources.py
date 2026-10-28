"""Synthetic historical review DATA; all scientific file/proof guards stay real."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import corrected_setup as export
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import painted_adoption_table as table
from darsia.fluidflower import painted_analysis_exclusions as subject
from darsia.fluidflower import painted_target_integration as target
from darsia.fluidflower import photo_edge_continuation as continuation

from .test_geometry_registered_export_apply import registered_setup  # noqa: F401
from .test_painted_adoption_table import imported, synthetic_payload_admission  # noqa: F401
from .test_painted_analysis_exclusions import gap_case, _adopt_target_fixture, _paint_component  # noqa: F401


def _collector_case(tmp_path, monkeypatch, kind, authority_kind="series configuration"):
    """Only review loading is an explicit collector double; _checked stays real.

    Stale live paths represent embedded historical DATA at this boundary.
    The separate integration case below uses actual proofs and owned Apply.
    """
    pins = {}

    def pin(name, payload=None):
        path = tmp_path / name
        raw = (name + " original bytes\n").encode() if payload is None else json.dumps(payload).encode()
        path.write_bytes(raw)
        pins[name] = subject._bound(path)
        return deepcopy(pins[name])

    def historical(name):
        bound = pin(name)
        before = Path(bound["path"]).read_text(encoding="utf8")
        Path(bound["path"]).write_bytes((name + " owned current postimage\n").encode())
        return dict(bound, text=before)

    old_configuration = historical("historical-baseline.toml")
    model = dict(descriptor=pin("source-model.json"),
        photograph=pin("source-photo.npz"), run_id="one", shape_hw=[2, 3],
        metric_to_grid=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        document=dict(baseline_configuration={key: old_configuration[key]
            for key in ("path", "sha256", "size_bytes")}),
        future_model_input=pin("future-model.pin"))
    other_model = deepcopy(model)
    other_model.update(descriptor=pin("target-model.json"), photograph=pin("target-photo.npz"), run_id="two")
    frame_authority = dict(kind=authority_kind, future_authority_input=pin("future-authority.pin"))
    names = ("project",) if authority_kind == "saved rig frame" else ("project", "series", "common", "run")
    for name in names:
        frame_authority[name] = dict(historical("historical-" + name + ".toml"),
            future_snapshot_input=pin("future-" + name + "-snapshot.pin"))
    contract = dict(format=continuation.FORMAT, policy=continuation.POLICY, target_grid="native",
        source_model=model, target_model=other_model, frame_authority=frame_authority,
        geometry_index=pin("geometry-index.json"), future_contract_input=pin("future-contract.pin"),
        active_scientific_closure=dict(raw=pin("source.ARW"), mask=pin("mask.npz")))
    review_binding = pin("review.json", {})
    review = dict(format=subject.REVIEW_FORMAT, review_binding=review_binding,
        original_product=pin("original.npz"),
        original_provenance=dict(configuration=pin("legacy-still-required.toml")),
        future_review_input=pin("future-review.pin"),
        runs=[dict(run_id="two", contract=contract, product=pin("review-product.npz"),
                   future_row_input=pin("future-row.pin"))],
        display_context=dict(configuration=historical("display-only.toml")))
    # This does not stand in for review/proof validation. It isolates the two
    # metadata dependency collectors with real bound-file verification intact.
    monkeypatch.setattr(subject, "read_gap_review", lambda path: review)
    if kind == "analysis_exclusions":
        preparation = dict(analysis_exclusions=dict(
            approval=pin("gap-approval.json", dict(review=review_binding)),
            runs=dict(two=dict(artifact=pins["mask.npz"]))))
        collect = lambda: subject.exclusion_sources(preparation)
    else:
        version = pin("version.json", dict(basis=dict(review=review_binding), product=pin("paint.npz")))
        approval = pin("target-approval.json", {})
        preparation = dict(target_corrections=dict(bindings=[dict(version=version, approval=approval)]))
        collect = lambda: target.target_sources(preparation)
    return SimpleNamespace(review=review, collect=collect, pins=pins)


@pytest.mark.parametrize("kind", ["analysis_exclusions", "target_corrections"])
@pytest.mark.parametrize("authority_kind", ["saved rig frame", "series configuration"])
def test_typed_review_uses_frozen_data_and_keeps_all_other_live_pins(tmp_path, monkeypatch, kind, authority_kind):
    case = _collector_case(tmp_path, monkeypatch, kind, authority_kind)
    before = deepcopy(case.review)
    result = case.collect()
    names = {Path(bound["path"]).name for bound in result}
    expected = {name for name in case.pins if not name.startswith(("historical-", "display-only"))}
    assert names == expected
    assert case.review == before


@pytest.mark.parametrize("kind", ["analysis_exclusions", "target_corrections"])
def test_unknown_or_malformed_contract_data_cannot_hide_a_live_file_pin(tmp_path, monkeypatch, kind):
    case = _collector_case(tmp_path, monkeypatch, kind)
    original = deepcopy(case.review["runs"][0]["contract"])
    variants = []
    for key, value in (("format", "unknown-continuation"), ("policy", "unknown-policy"),
                       ("target_grid", "camera_u")):
        variant = deepcopy(original)
        variant[key] = value
        variants.append(variant)
    variant = deepcopy(original)
    variant["source_model"].pop("descriptor")
    variants.append(variant)
    variant = deepcopy(original)
    variant["frame_authority"]["kind"] = "unknown-authority"
    variants.append(variant)
    variant = deepcopy(original)
    variant["frame_authority"]["project"].pop("text")
    variants.append(variant)
    for variant in variants:
        case.review["runs"][0]["contract"] = variant
        with table._FILES.fresh(), pytest.raises(ValueError):
            case.collect()


@pytest.mark.parametrize("kind", ["analysis_exclusions", "target_corrections"])
def test_scientific_model_photo_raw_and_unknown_legacy_pins_still_refuse_tampering(tmp_path, monkeypatch, kind):
    case = _collector_case(tmp_path, monkeypatch, kind)
    # Unknown keys at every structural level remain guarded; neither filename
    # suffix nor resemblance to an embedded snapshot grants an exemption.
    for name in ("legacy-still-required.toml", "source-model.json", "target-model.json",
                 "source-photo.npz", "target-photo.npz", "source.ARW", "mask.npz",
                 "future-review.pin", "future-row.pin", "future-contract.pin",
                 "future-model.pin", "future-authority.pin", "future-project-snapshot.pin"):
        path = Path(case.pins[name]["path"])
        original = path.read_bytes()
        path.write_bytes(original + b"tampered\n")
        with table._FILES.fresh(), pytest.raises(ValueError):
            case.collect()
        path.write_bytes(original)


def _forget_local_replay_caches():
    table._FILES.clear()
    with table._DERIVED_LOCK:
        table._DERIVED.clear()
    with continuation._NATIVE_PROOFS_LOCK:
        continuation._NATIVE_PROOFS.clear()


def test_owned_apply_cold_replay_uses_exact_configuration_copy_and_refuses_its_tampering(gap_case):
    """Actual Store/export/Apply, then original cold proof/copy guards, no doubles."""
    case = gap_case
    row, _, binding = _paint_component(case)
    run = row["run_id"]
    prepared, revision = _adopt_target_fixture(case, [binding], "cold_target_apply")
    copied = prepared["painting_geometry"]["baseline_configurations"][run]
    original = deepcopy(copied["original"])
    copy = table._local(prepared["folder"], copied["copy"])
    before = Path(original["path"]).read_bytes()
    assert Path(copy["path"]).read_bytes() == before
    assert all(copy[key] == original[key] for key in ("sha256", "size_bytes"))
    if run != prepared["reference_run"]:
        export.export_corrected_setup(case.data.project,
            reference=SimpleNamespace(run_id=prepared["reference_run"]), actor="synthetic reference export")
    exported = export.export_corrected_setup(case.data.project,
        reference=SimpleNamespace(run_id=run), actor="synthetic cold replay")
    assert any(Path(path).resolve() == Path(original["path"]).resolve()
               for path in exported.receipt["binding"]["configuration_bundles"][run])
    attempt = export.apply_corrected_selected_run(case.data.project, exported, rebuild_facies_by_name=True)
    after = Path(original["path"]).read_bytes()
    assert after != before and Path(copy["path"]).read_bytes() == before
    assert export._apply_completion(exported.destination, exported.receipt)["complete"] is True

    _forget_local_replay_caches()
    registered = sources.registered_run_sources(case.data.project.root, revision, run)
    sources.verify_registered_sources(registered)
    guards = registered["guarded_sources"]
    assert copy in guards
    assert not any(bound["path"] == original["path"] for bound in guards)
    assert prepared["painting_geometry"]["baseline_configurations"][run]["original"] == original
    assert export.apply_corrected_selected_run(case.data.project, exported, rebuild_facies_by_name=True) == attempt
    assert Path(original["path"]).read_bytes() == after

    Path(copy["path"]).write_bytes(before + b"\n# changed retained source copy\n")
    _forget_local_replay_caches()
    with pytest.raises(ValueError):
        sources.registered_run_sources(case.data.project.root, revision, run)
    assert Path(original["path"]).read_bytes() == after
