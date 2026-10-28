"""Tiny carried-source contracts; no campaign maps or scientific acceptance.

The synthetic gate files exercise admission only. They are not N12 results.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from tests.unit.test_geometry_imported_rasters import imported, save, snapshot


def matrix(dx=0.0, dy=0.0):
    return [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]]


def shift(dx=0.0, dy=0.0):
    return sources.registration_pixel_shift(dx_mm=dx, dy_mm=dy, pixel_size_mm=1.0)


def ref(document, pointer=""):
    return dict(document=deepcopy(document), pointer=pointer)


def write_json(path, value):
    path.open("xb").write((json.dumps(value, allow_nan=False) + "\n").encode())
    return sources.descriptor(path)


def archive(path, value, metadata):
    with path.open("xb") as stream:
        np.savez(stream, array=value, metadata=metadata)
    return sources.descriptor(path)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32])
def test_lossless_identity_preserves_zero_255_dtype_and_unknown_source_fill(dtype):
    values = np.array([[0, 1, 255], [4, 8, 12]], dtype=dtype)
    original = values.copy()
    reference = sources.carried_label_reference(values, reference_shift=shift(),
        source_fill_status="NOT_RECORDED")
    result = sources.carry_label_reference(reference["labels"], reference_matrix=matrix(),
        target_matrix=matrix(), target_shape_hw=[2, 3], reference_filled=reference["geometry_filled"],
        source_fill_status="NOT_RECORDED")
    assert np.array_equal(result["labels"], values)
    assert result["labels"].dtype == values.dtype
    assert np.array_equal(original, values) and not np.shares_memory(result["labels"], values)
    assert not result["geometry_filled"].any() and result["support"] is None
    assert result["metrics"]["source_fill_status"] == "NOT_RECORDED"
    assert result["metrics"]["uncovered_pixel_count"] == 0


def test_forward_composition_and_edge_flags_agree_with_original_integer_primitive():
    labels = np.array([[0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 255]], np.int32)
    # The nonidentity reference cancels: target @ inverse(reference) is +1 x.
    result = sources.carry_label_reference(labels, reference_matrix=matrix(2.0),
        target_matrix=matrix(3.0), target_shape_hw=[3, 4],
        reference_filled=np.zeros(labels.shape, bool), source_fill_status="NOT_RECORDED")
    expected, _ = sources.integer_label_edge_replication(labels,
        row_shift=0, column_shift=1, pixel_size_mm=1.0)
    assert np.array_equal(result["labels"], expected)
    assert result["metrics"]["composition"] == matrix(1.0)
    assert np.array_equal(result["geometry_filled"],
        [[True, False, False, False], [True, False, False, False], [True, False, False, False]])


def test_nontranslation_nearest_map_has_explicit_known_samples():
    labels = np.array([[0, 10, 20, 255], [30, 40, 50, 60], [70, 80, 90, 100]], np.uint8)
    transform = [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    result = sources.carry_label_reference(labels, reference_matrix=matrix(),
        target_matrix=transform, target_shape_hw=[3, 4],
        reference_filled=np.zeros(labels.shape, bool), source_fill_status="NOT_RECORDED")
    assert np.array_equal(result["labels"], [[0, 10, 20, 255], [30, 30, 40, 50], [70, 70, 70, 80]])
    assert np.array_equal(result["geometry_filled"],
        [[False, False, False, False], [True, False, False, False], [True, True, False, False]])


def test_original_source_fill_and_support_survive_reference_shift_and_carry():
    labels = np.array([[0, 2, 255, 4], [5, 6, 7, 8], [9, 10, 11, 12]], np.uint8)
    source_filled = np.zeros(labels.shape, bool)
    source_filled[1, 1] = True
    source_support = np.ones(labels.shape, bool)
    source_support[2, 1] = False
    reference = sources.carried_label_reference(labels, reference_shift=shift(1.0),
        source_fill_status="KNOWN", source_filled=source_filled, source_support=source_support)
    assert reference["geometry_filled"][1, 2]
    assert reference["geometry_filled"][:, 0].all() and not reference["support"][:, 0].any()
    assert not reference["support"][2, 2]
    result = sources.carry_label_reference(reference["labels"], reference_matrix=matrix(),
        target_matrix=matrix(0.0, 1.0), target_shape_hw=[3, 4],
        reference_filled=reference["geometry_filled"], reference_support=reference["support"],
        source_fill_status="KNOWN")
    assert result["geometry_filled"][2, 2] and result["geometry_filled"][0].all()
    assert not result["support"][0].any()
    assert result["labels"][1, 3] == 255


@pytest.mark.parametrize("labels", [np.array([[256]], np.int32), np.array([[-1]], np.int32),
    np.array([[1.0]], np.float64), np.array([[[1]]], np.uint8)])
def test_unrepresentable_or_noninteger_inputs_fail_without_aliasing(labels):
    with pytest.raises(ValueError, match="lossless integer"):
        sources.carried_label_reference(labels, reference_shift=shift(), source_fill_status="NOT_RECORDED")


@pytest.mark.parametrize("status,filled", [("KNOWN", None), ("NOT_RECORDED", np.zeros((2, 3), bool))])
def test_unknown_source_fill_is_not_silently_declared_measured(status, filled):
    with pytest.raises(ValueError, match="actual mask"):
        sources.carried_label_reference(np.ones((2, 3), np.uint8), reference_shift=shift(),
            source_fill_status=status, source_filled=filled)


@pytest.mark.parametrize("bad", [[[True, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
def test_matrix_types_and_invertibility_are_explicit(bad):
    with pytest.raises(ValueError, match="homography"):
        sources.carry_label_reference(np.ones((2, 3), np.uint8), reference_matrix=matrix(),
            target_matrix=bad, target_shape_hw=[2, 3], reference_filled=np.zeros((2, 3), bool),
            source_fill_status="NOT_RECORDED")


def carried_draft(data, photo_record_fields=3):
    original_contract = deepcopy(data.config["imported_rasters"])
    photo = sources.descriptor(data.config["baseline_path"])
    parent_photo = deepcopy(photo)
    if photo_record_fields == 6:
        pixels = geometry._load_array(Path(photo["path"]))
        parent_photo.update(mtime_ns=Path(photo["path"]).stat().st_mtime_ns,
                            shape=list(pixels.shape), dtype=pixels.dtype.str)
    data.config["frame_binding"] = {"runs": {
        run: {"fields": {"shape_corrected_baseline": deepcopy(parent_photo)}} for run in ("reference", "one")}}
    data.config = sources.bind_imported_rasters(data.config,
        sources=original_contract["sources"], diagnosis=original_contract["diagnosis"])
    parent = save(data)
    draft = sources.prepare_canonical_edit(data.project, actor="synthetic owner", reason="explicit carried fixture")
    payload = sources.canonical_parent_payload(draft)
    root = data.project.root / "carried-staging"
    root.mkdir()
    metadata = geometry._darsia_metadata(Path(payload["actual"]["labels"]["path"]))
    authority = write_json(root / "authority.json", {"decision": {"synthetic": True},
        "source_fill": {"status": "NOT_RECORDED"}})
    proof = write_json(root / "chart-proof.json", {"accepted": True})
    evaluator_path = root / "synthetic-evaluator-identity.py"
    evaluator_path.write_bytes(b"# Synthetic gate identity only; no N12 result.\n")
    evaluator = sources.descriptor(evaluator_path)
    geometries, charts = {}, {}
    for run, transform in (("reference", matrix(2.0)), ("one", matrix(3.0))):
        doc = write_json(root / (run + "-geometry.json"),
            {"acquisition": {"name": run, "shape": list(payload["labels"].shape)}, "transform": transform})
        geometries[run] = dict(run_id=run, matrix=ref(doc, "/transform"),
            run_identity=ref(doc, "/acquisition/name"), shape=ref(doc, "/acquisition/shape"))
        chart = write_json(root / (run + "-chart.json"), dict(format="darsia-native-raster-chart-v1",
            run_id=run, source_image=photo, geometry_document=doc, native_shape_hw=list(payload["labels"].shape),
            offset_xy=[0, 0], status="PASS", proof=ref(proof, "/accepted")))
        charts[run] = ref(chart)
    fill = dict(status="NOT_RECORDED", mask=None, support=None, provenance=ref(authority, "/source_fill"))
    recipe = dict(format="darsia-carried-reference-recipe-v1", reference_run="reference",
        reference_shift=shift(), authority=ref(authority, "/decision"),
        gate_limits=dict(median_mm=2.0, maximum_mm=5.0), chart_proof=ref(proof, "/accepted"),
        evaluators=[evaluator], source_fill=fill)
    recipe_ref = ref(write_json(root / "recipe.json", recipe))
    reference = sources.carried_label_reference(payload["labels"], reference_shift=recipe["reference_shift"],
        source_fill_status=fill["status"])
    reference_labels = archive(root / "reference-labels.npz", reference["labels"], metadata)
    reference_filled = archive(root / "reference-filled.npz", reference["geometry_filled"], metadata)
    target = sources.carry_label_reference(reference["labels"], reference_matrix=matrix(2.0),
        target_matrix=matrix(3.0), target_shape_hw=list(payload["labels"].shape),
        reference_filled=reference["geometry_filled"], source_fill_status=fill["status"])
    labels = archive(root / "one-labels.npz", target["labels"], metadata)
    filled = archive(root / "one-filled.npz", target["geometry_filled"], metadata)
    gate = dict(format="darsia-carried-boundary-gate-v1", status="PASS", run_id="one", reference_run="reference",
        labels=labels, recipe=recipe_ref, target_geometry=geometries["one"], reference_geometry=geometries["reference"],
        target_chart=charts["one"], sharp_boundary_count=3, median_mm=0.0, maximum_mm=0.0,
        missing_prior_witnesses=[], ambiguous_boundaries=[], evaluators=[evaluator])
    gate_ref = ref(write_json(root / "synthetic-gate.json", gate))
    evidence = dict(format="darsia-carried-registration-evidence-v1", run_id="one", reference_run="reference",
        recipe=recipe_ref, reference_geometry=geometries["reference"], target_geometry=geometries["one"],
        reference_chart=charts["reference"], target_chart=charts["one"], source_fill=fill,
        reference_labels=reference_labels, reference_filled=reference_filled, reference_support=None,
        filled=filled, support=None, reference_metrics=reference["metrics"], metrics=target["metrics"], gate=gate_ref)
    evidence_ref = ref(write_json(root / "one-evidence.json", evidence))
    record = dict(kind=sources.CARRIED, parent_revision_id=parent["revision_id"],
        parent_labels=payload["actual"]["labels"], status="carried", authority=authority,
        measurement=gate_ref["document"], labels=labels, source_image=photo, carried_evidence=evidence_ref)
    draft["run_geometry"]["runs"] = [dict(run_id="one", geometry_choice="run_override", enabled=True,
        acquisition_state="unknown", labels_path=labels["path"], source_image_path=photo["path"], registration=record)]
    return dict(draft=draft, parent=parent, labels=target["labels"], root=root, record=record,
        evidence=evidence, recipe=recipe, gate=gate, evaluator=evaluator_path)


def altered_evidence(data, *, evidence=None, gate=None, recipe=None):
    value = deepcopy(data["evidence"] if evidence is None else evidence)
    if recipe is not None:
        value["recipe"] = ref(write_json(data["root"] / "altered-recipe.json", recipe))
    if gate is not None:
        changed = write_json(data["root"] / "altered-gate.json", gate)
        value["gate"] = ref(changed)
        data["record"]["measurement"] = changed
    data["record"]["carried_evidence"] = ref(write_json(data["root"] / "altered-evidence.json", value))


@pytest.mark.parametrize("photo_record_fields", [3, 6])
def test_genuine_store_resolver_and_source_CAS_keep_carried_artifacts_and_full_csv(imported, photo_record_fields):
    original = snapshot(imported.original)
    data = carried_draft(imported, photo_record_fields=photo_record_fields)
    frame_binding_before = deepcopy(data["draft"]["frame_binding"])
    assert all(len(row["fields"]["shape_corrected_baseline"]) == photo_record_fields
               for row in frame_binding_before["runs"].values())
    assert sources._same_grid_metadata(data["parent"]["configuration"]["frame_binding"], frame_binding_before)
    parent_before = snapshot(Path(data["parent"]["artifacts"]["facies_path"]).parent)
    revision = save(imported, data["draft"])
    assert sources._same_grid_metadata(data["draft"]["frame_binding"], frame_binding_before)
    assert sources._same_grid_metadata(revision["configuration"]["frame_binding"], frame_binding_before)
    assert sources._same_grid_metadata(data["parent"]["configuration"]["frame_binding"], frame_binding_before)
    paths = geometry.resolve_run_geometry_artifacts(revision, "one")
    assert paths["scope"] == "run_override"
    assert np.array_equal(geometry._load_array(Path(paths["labels_path"])), data["labels"])
    assert geometry._load_array(Path(paths["labels_path"])).dtype == np.dtype(np.int32)
    cfg = deepcopy(data["draft"])
    cfg["labels_path"] = data["record"]["labels"]["path"]
    canonical, _ = geometry._canonical_facies(cfg)
    assert np.array_equal(geometry._load_array(Path(paths["facies_path"])), canonical)
    assert Path(paths["facies_properties_path"]).read_bytes() == Path(imported.config["facies_properties_source_path"]).read_bytes()
    bound = sources.registered_run_sources(imported.project.root, revision, "one")
    assert "admitted carried labels" in bound["label_serialization"]
    assert data["record"]["carried_evidence"]["document"] in bound["guarded_sources"]
    assert snapshot(imported.original) == original
    assert snapshot(Path(data["parent"]["artifacts"]["facies_path"]).parent) == parent_before
    data["evaluator"].write_bytes(b"# changed source after admission\n")
    with pytest.raises(ValueError, match="source changed"):
        sources.verify_registered_sources(bound)


@pytest.mark.parametrize("mutation", ["FAIL", "HELD", "missing", "ambiguous", "median", "maximum", "typed-count", "foreign-labels", "evaluator"])
def test_failed_held_borrowed_or_incomplete_gate_never_enters_store(imported, mutation):
    data = carried_draft(imported)
    gate = deepcopy(data["gate"])
    if mutation in {"FAIL", "HELD"}:
        gate["status"] = mutation
    elif mutation == "missing":
        gate["missing_prior_witnesses"] = ["sand-top"]
    elif mutation == "ambiguous":
        gate["ambiguous_boundaries"] = ["1_2"]
    elif mutation == "median":
        gate["median_mm"] = 2.01
    elif mutation == "maximum":
        gate["maximum_mm"] = 5.01
    elif mutation == "typed-count":
        gate["sharp_boundary_count"] = True
    elif mutation == "foreign-labels":
        gate["labels"] = data["evidence"]["reference_labels"]
    elif mutation == "evaluator":
        gate["evaluators"] = [data["record"]["authority"]]
    altered_evidence(data, gate=gate)
    before = snapshot(imported.project.root)
    with pytest.raises(ValueError, match="Carried"):
        sources.canonical_parent_payload(data["draft"])
    assert snapshot(imported.project.root) == before


@pytest.mark.parametrize("mutation", ["geometry", "source-fill", "metrics", "reference", "unknown-key"])
def test_exact_source_composition_reference_and_fill_bindings_cannot_be_relabelled(imported, mutation):
    data = carried_draft(imported)
    evidence = deepcopy(data["evidence"])
    if mutation == "geometry":
        evidence["target_geometry"]["matrix"] = evidence["reference_geometry"]["matrix"]
    elif mutation == "source-fill":
        evidence["source_fill"]["status"] = "KNOWN"
    elif mutation == "metrics":
        evidence["metrics"]["composition"][0][2] = 1.0 + .01
    elif mutation == "reference":
        evidence["reference_run"] = "other"
    elif mutation == "unknown-key":
        evidence["fallback"] = True
    altered_evidence(data, evidence=evidence)
    with pytest.raises(ValueError, match="Carried"):
        sources.canonical_parent_payload(data["draft"])


def test_reference_recipe_type_change_is_not_true_equals_one(imported):
    data = carried_draft(imported)
    recipe = deepcopy(data["recipe"])
    recipe["reference_shift"]["column_shift"] = False
    altered_evidence(data, recipe=recipe)
    with pytest.raises(ValueError, match="reference shift"):
        sources.canonical_parent_payload(data["draft"])


def test_foreign_photo_or_grid_cannot_be_borrowed(imported):
    data = carried_draft(imported)
    metadata = geometry._darsia_metadata(Path(data["record"]["labels"]["path"]))
    foreign = archive(data["root"] / "foreign-image.npz", imported.arrays["baseline"], metadata)
    data["record"]["source_image"] = foreign
    data["draft"]["run_geometry"]["runs"][0]["source_image_path"] = foreign["path"]
    with pytest.raises(ValueError, match="bound corrected baseline"):
        sources.canonical_parent_payload(data["draft"])


def test_carried_evidence_cannot_claim_an_unknown_json_schema(imported):
    data = carried_draft(imported)
    data["record"]["carried_evidence"]["client_path"] = str(data["root"])
    with pytest.raises(ValueError, match="JSON reference"):
        sources.canonical_parent_payload(data["draft"])
