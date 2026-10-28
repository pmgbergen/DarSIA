"""Lossless optional score product/commit/index contracts on tiny local bytes."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower.phase import probability_storage as storage
from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json
from darsia.fluidflower.pipeline import products_publication as publication
from darsia.fluidflower.pipeline import staged_data
from darsia.fluidflower import storage_indexed_products as indexed
from .test_staged_data import phase1_data, _closure, _rewrite_frame


def _models():
    bound = dict(path="//synthetic-unit-only/share/head.json", sha256="a" * 64, size_bytes=17)
    return dict(native=dict(documents=dict(g3=deepcopy(bound), regional_refit=deepcopy(bound))),
                edge=dict(documents=dict(g3=deepcopy(bound), regional_refit=deepcopy(bound))),
                placement=dict(pixel_model=deepcopy(bound), regional_model=deepcopy(bound)))


def _capture():
    p = np.array([[np.float32(.70001), .2], [.9999, .4]], np.float32)
    r = np.array([[.60001, .8], [.79999, .3]], np.float32)
    support = np.ones(p.shape, bool)
    placement = {}
    storage.pair(placement, p, r, support, support, (p >= .7) & (r >= .6),
                 branch="placement", pixel_threshold=.7, regional_threshold=.6)
    capture = dict(edge={}, placement=placement, postprocessing={})
    classes = dict(modelled=support.copy(), filled_observed=np.zeros(p.shape, bool),
                   filled_unobserved=np.zeros(p.shape, bool))
    classes["modelled"][1, 1] = False
    classes["filled_unobserved"][1, 1] = True
    labels = np.array([[2, 1], [0, 0]], np.uint8)
    storage.finish(capture, labels, classes, (0, 0, 2, 2))
    return capture, labels


def _product(tmp_path):
    tmp_path.mkdir(parents=True)
    capture, labels = _capture()
    label_path = tmp_path / "phase_labels.png"
    with label_path.open("xb") as stream:
        Image.fromarray(labels).save(stream, format="PNG")
    configuration = _write_json(tmp_path / "phase_config.json", dict(model_bundles=_models()))
    source = dict(run="synthetic", source_sha256="b" * 64)
    spec = dict(case_id="case-one", source=source)
    original = {p: p.read_bytes() for p in tmp_path.iterdir()}
    roles = storage.write_product(tmp_path, spec, capture, model_bindings=_models(),
        phase_configuration=configuration, labels=labels, label_descriptor=asset(label_path))
    assert all(path.read_bytes() == raw for path, raw in original.items())
    return roles, source, labels, capture


def test_lossless_archive_reader_preserves_exact_float32_and_unscored_nan(tmp_path):
    roles, source, labels, capture = _product(tmp_path / "case-one")
    actual = storage.read_product(roles, reader=DataReader(), source=source, case_id="case-one")
    for name, expected in capture["product"]["arrays"].items():
        assert actual["arrays"][name].dtype == expected.dtype
        assert actual["arrays"][name].tobytes() == expected.tobytes()
    receipt = actual["receipt"]
    assert receipt["labels"] == asset(tmp_path / "case-one" / "phase_labels.png")
    assert receipt["logical_bytes"] == labels.size * 11
    assert receipt["compressed_bytes"] == Path(roles["archive"]["path"]).stat().st_size
    assert receipt["projected_3090_compressed_bytes"] == receipt["compressed_bytes"] * 3090
    assert "not an admitted job budget" in receipt["projection_basis"]
    assert receipt["head_bindings"]["native"]["g3"]["sha256"] == "a" * 64
    assert receipt["evidence"]["score_feedback"] is False


@pytest.mark.parametrize("bad", ["pixel_half", "regional_half", "wrong_shape", "float_codes", "unknown_code",
                                 "invented_unscored", "missing_scored", "outside_nonzero", "unknown_rule_bit", "unscored_rule_bit", "float_rule_bits"])
def test_saved_array_contract_refuses_lossy_or_fabricated_scores(bad):
    capture, labels = _capture()
    arrays = deepcopy(capture["product"]["arrays"])
    if bad == "pixel_half": arrays["pixel_probability"] = arrays["pixel_probability"].astype(np.float16)
    elif bad == "regional_half": arrays["regional_probability"] = arrays["regional_probability"].astype(np.float16)
    elif bad == "wrong_shape": arrays["regional_probability"] = arrays["regional_probability"][:, :1]
    elif bad == "float_codes": arrays["branch_support"] = arrays["branch_support"].astype(np.float32)
    elif bad == "unknown_code": arrays["branch_support"][0, 0] = 4
    elif bad == "invented_unscored": arrays["pixel_probability"][1, 1] = 0
    elif bad == "missing_scored": arrays["regional_probability"][0, 0] = np.nan
    elif bad == "outside_nonzero": arrays["pixel_probability"][1, 0] = .8
    elif bad == "unknown_rule_bit": arrays["later_rule_changes"][0, 0] = 4096
    elif bad == "unscored_rule_bit": arrays["later_rule_changes"][1, 1] = 1
    else: arrays["later_rule_changes"] = arrays["later_rule_changes"].astype(np.float32)
    with pytest.raises(ValueError): storage.validate_arrays(arrays, labels=labels)


@pytest.mark.parametrize("field", ["source", "encoding", "head_bindings", "compressed_bytes", "producer_sha256"])
def test_receipt_retains_exact_source_encoding_heads_and_accounting(tmp_path, field):
    roles, source, labels, _ = _product(tmp_path / "case-one")
    value = json.loads(Path(roles["receipt"]["path"]).read_bytes())
    if field == "source": value[field]["source_sha256"] = "c" * 64
    elif field == "encoding": value[field] = "float16"
    elif field == "head_bindings": value[field]["native"]["g3"].pop("sha256")
    elif field == "compressed_bytes": value[field] += 1
    else: value[field] = "missing"
    with pytest.raises(ValueError):
        storage.validate_receipt(value, roles, source=source, case_id="case-one", shape=labels.shape)


def test_corrupt_payload_is_rejected_by_original_reader(tmp_path):
    roles, source, _, _ = _product(tmp_path / "case-one")
    with Path(roles["archive"]["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        storage.read_product(roles, reader=DataReader(), source=source, case_id="case-one")


def test_actual_phase1_commit_and_sealer_admit_only_requested_owned_pair(phase1_data, tmp_path, monkeypatch):
    config = deepcopy(phase1_data)
    config["gas_probability_storage"] = dict(format=storage.REQUEST_FORMAT, encoding=storage.ENCODING)
    config["phase"]["configuration"] = _write_json(tmp_path / "actual_phase.json", dict(model_bundles=_models()))
    closure = _closure(tmp_path / "producer", config)
    record = deepcopy(closure["record"])
    capture, labels = _capture()
    stage = Path(closure["frame"]["path"]).parent
    roles = storage.write_product(stage, config["frames"][0], capture, model_bindings=_models(),
        phase_configuration=config["phase"]["configuration"], labels=labels,
        label_descriptor=record["phase"]["labels"])
    record["probabilities"] = roles
    for bound in roles.values(): record["files"][Path(bound["path"]).name] = bound
    completion = _rewrite_frame(closure, record)

    def no_decode(*args, **kwargs):
        raise AssertionError("The metadata sealer must not decode scientific arrays")

    monkeypatch.setattr(np, "load", no_decode)
    result = staged_data.verify_phase1_completion(completion, closure["configuration"],
        expected_commit="1" * 40, reader=DataReader())
    assert result["scientific_arrays_decoded"] is False
    assert result["records"][0]["record"]["probabilities"] == roles
    missing = deepcopy(record)
    missing.pop("probabilities")
    completion = _rewrite_frame(closure, missing)
    with pytest.raises(ValueError, match="request/actual product"):
        staged_data.verify_phase1_completion(completion, closure["configuration"],
            expected_commit="1" * 40, reader=DataReader())


def test_probability_role_cannot_escape_owned_inventory(phase1_data, tmp_path):
    closure = _closure(tmp_path, phase1_data)
    record = deepcopy(closure["record"])
    record["probabilities"] = dict(archive=record["phase"]["classes"], receipt=record["before_rules"]["receipt"])
    with pytest.raises(ValueError, match="exact owned output leaves"):
        staged_data._frame_record(record, phase1_data["frames"][0], closure["configuration"],
                                  "float32", closure["frame"])


@pytest.mark.parametrize("bad", [None, "format", "run", "phase_source", "label_role"])
def test_indexed_probability_reader_requires_exact_producer_and_explicit_collection(tmp_path, bad):
    roles, source, _, _ = _product(tmp_path / "producer" / "case-one")
    root = Path(roles["archive"]["path"]).parent
    label = asset(root / "phase_labels.png")
    record = dict(format=staged_data.FRAME_FORMAT, complete=True, run="synthetic",
                  case_id="case-one", source=source, probabilities=roles,
                  phase=dict(complete=True, source_sha256=source["source_sha256"], labels=label),
                  files={Path(v["path"]).name: v for v in (*roles.values(), label)})
    if bad == "format": record["format"] = "untyped"
    elif bad == "run": record["run"] = "other-run"
    elif bad == "phase_source": record["phase"]["source_sha256"] = "f" * 64
    elif bad == "label_role": record["phase"]["labels"] = roles["archive"]
    commit = _write_json(root / "phase1_commit.json", record)
    completed = _write_json(root / "completed.json", dict(complete=True, worker_references_closed=True, run="synthetic"))
    job = _write_json(tmp_path / "job.json", dict(synthetic=True))
    rows, small = [], []
    for bound, kind in ((commit, "commit_receipt"), (roles["receipt"], "operational_receipt"),
                        (roles["archive"], "map"), (label, "map")):
        logical = "phase1/case-one/" + Path(bound["path"]).name
        rows.append(publication.product(logical, machine_id="Main", local_path=bound["path"],
                    sha256=bound["sha256"], bytes=bound["size_bytes"], kind=kind))
        if kind != "map": small.append(logical)
    index = publication.publish_run_metadata(tmp_path / "shared", job=job, run="synthetic",
        attempt_id="a" * 32, machine_id="Main", completed=completed, products=rows,
        metadata_product_ids=small, file_writers_closed=True, all_final_products_listed=True)
    if bad is not None:
        with pytest.raises(ValueError, match="actual phase-one commit|final labels"):
            indexed.read_indexed_phase_probabilities(index, "case-one", current_machine_id="Main")
        return
    actual = indexed.read_indexed_phase_probabilities(index, "case-one", current_machine_id="Main")
    assert actual["arrays"]["pixel_probability"].dtype == np.float32
    from darsia.fluidflower.storage_saved_products import StoredProductCollectionRequired
    with pytest.raises(StoredProductCollectionRequired):
        indexed.read_indexed_phase_probabilities(index, "case-one", current_machine_id="Second")
    maps = [row["product_id"] for row in rows if row["kind"] == "map"]
    collected = publication.collect_products(index, maps, tmp_path / "collected",
        current_machine_id="Main", file_writers_closed=True)
    replay = indexed.read_indexed_phase_probabilities(index, "case-one", current_machine_id="Second",
                                                     collections=[collected])
    assert replay["arrays"]["pixel_probability"].tobytes() == actual["arrays"]["pixel_probability"].tobytes()


@pytest.mark.parametrize("bad", ["old_profile", "wrong_request", "missing_writer"])
def test_optin_capacity_refuses_old_or_uncovered_writer_profile(monkeypatch, tmp_path, bad):
    from darsia.fluidflower.pipeline import capacity
    monkeypatch.setattr("darsia_cache.folders.folder_mode", lambda *_args: False)
    request = dict(format=storage.REQUEST_FORMAT, encoding=storage.ENCODING)
    profile = dict(format="synthetic-no-measurement", pipeline="native-phase-statistics",
                   precision="float32", measured_commit="a" * 40, measured_case_ids=["case-one"],
                   measurement=None, completion=None, writers=[], orchestration=[], layout_receipt=None)
    if bad != "old_profile":
        profile["gas_probability_storage"] = deepcopy(request)
    if bad == "wrong_request": profile["gas_probability_storage"]["encoding"] = "float16"
    monkeypatch.setattr(capacity, "settings", lambda *_: (None, None, {"synthetic": True}))
    monkeypatch.setattr(capacity, "read_bound", lambda *_: (profile, b"synthetic-test-only"))
    expected = "actual applicable output writer" if bad == "missing_writer" else "own actually measured"
    with pytest.raises(ValueError, match=expected):
        capacity.load_estimate(None, "Synthetic", dict(pipeline="native-phase-statistics"),
            dict(gas_probability_storage=request), repository=tmp_path)


@pytest.mark.parametrize("bad", ["wrong_spatial_mask", "wrong_branch_bit", "different_membership_receipt"])
def test_saved_reader_independently_replays_spatial_rule_membership(tmp_path, bad):
    roles, source, labels, capture = _product(tmp_path / "case-one")
    receipt = json.loads(Path(roles["receipt"]["path"]).read_bytes())
    arrays = deepcopy(capture["product"]["arrays"])
    if bad == "different_membership_receipt":
        receipt["evidence"]["spatial_proof"]["rules"]["post_top"]["applicable_pixels"] = 1
    else:
        # Deliberately rebind payload hashes to exercise semantic spatial validation,
        # rather than only the outer exact-byte DataReader guard.
        arrays["pixel_probability"][0, 1] = np.float32(.8)
        if bad == "wrong_spatial_mask":
            arrays["later_rule_changes"][0, 0] = storage.RULE_BITS["placement_minimum"]
        else:
            arrays["later_rule_changes"][0, 1] = storage.RULE_BITS["centre_minimum"]
        from darsia.fluidflower.mass.n1_pipeline import _write_npz
        archive_path = tmp_path / "rebound.npz"
        roles["archive"] = _write_npz(archive_path, arrays)
        receipt["archive"] = roles["archive"]
        receipt["arrays"] = {name: storage.identity(value) for name, value in arrays.items()}
        receipt["compressed_bytes"] = roles["archive"]["size_bytes"]
        receipt["projected_3090_compressed_bytes"] = roles["archive"]["size_bytes"] * 3090
    roles["receipt"] = _write_json(tmp_path / "rebound_receipt.json", receipt)
    with pytest.raises(ValueError):
        storage.read_product(roles, reader=DataReader(), source=source, case_id="case-one", labels=labels)


def test_write_refuses_post_finish_pair_mutation_before_any_archive(tmp_path):
    capture, labels = _capture()
    capture["product"]["arrays"]["pixel_probability"][0, 1] = np.float32(.8)
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(ValueError, match="unexplained spatial disagreement"):
        storage.write_product(stage, dict(source={}, case_id="x"), capture,
                              model_bindings={}, phase_configuration={}, labels=labels, label_descriptor={})
    assert list(stage.iterdir()) == []
