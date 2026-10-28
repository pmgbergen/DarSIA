"""Source-only authored guard tests; no scientific qualification is claimed.

These use the existing explicitly synthetic metadata/byte fixture to test
refusals, plus real DataReader SHA checks and gzip bytes. They never forge a
successful phase-one RunIOBinding or replace an original scientific executor.
The full RAW/P1/N1 replay fixture belongs to the separately scheduled QA run.
"""

from copy import deepcopy
import gzip
from pathlib import Path

import pytest

from darsia.fluidflower.mass.n1_pipeline import DataReader
from darsia.fluidflower.pipeline import staged_data
from darsia.fluidflower.pipeline.store_runtime import ResolvedInput
from darsia.fluidflower.published_dataset import staged_replay as coupling
from tests.unit.test_staged_data import phase1_data, _closure, _write


def _row(bound, use="run.configuration"):
    return ResolvedInput(dict(root_alias="series", relative_path=Path(bound["path"]).name,
        role="configuration", sha256=bound["sha256"], bytes=bound["size_bytes"]),
        Path(bound["path"]), (use,))


@pytest.fixture
def sealed_metadata(tmp_path, phase1_data):
    """Original DATA sealer with synthetic bytes, never a producer substitute."""
    original = _closure(tmp_path / "original", phase1_data)
    fresh = _closure(tmp_path / "fresh", phase1_data)
    path = tmp_path / "sealed-phase2.json"
    sealed = staged_data.seal_phase2_configuration(path, original["completion"], original["configuration"],
        expected_phase1_commit="1" * 40, fitted_calibration=original["calibration"],
        fitted_anchor=dict(elapsed_h=2.25, affected_pore_volume_m3=.0001), reader=DataReader())
    config = DataReader().json(sealed["configuration"])
    return original, fresh, config, [_row(sealed["configuration"])]


def _call(fixture, *, original=None, config=None, rows=None):
    old, new, sealed, inputs = fixture
    return coupling.verify_coupling(sealed if config is None else config,
        old["completion"] if original is None else original, new["completion"],
        DataReader(), DataReader(), inputs if rows is None else rows)


def test_metadata_only_reader_cannot_claim_coupled_reproduction():
    class MetadataOnly:
        def json(self, value):
            raise AssertionError("Unverified metadata reader was called")
    with pytest.raises(TypeError, match="content-verifying"):
        coupling.verify_coupling({}, {}, {}, MetadataOnly(), MetadataOnly(), [])


def test_synthetic_complete_records_do_not_stand_in_for_a_genuine_replayed_stage(sealed_metadata):
    with pytest.raises(ValueError, match="actual admitted RunIOBinding"):
        _call(sealed_metadata)


def test_missing_original_phase1_binding_is_refused_before_replay(sealed_metadata):
    old, _fresh, _config, _rows = sealed_metadata
    changed = dict(old["completion"], sha256="0" * 64)
    with pytest.raises(ValueError, match="original phase-one completion binding"):
        _call(sealed_metadata, original=changed)


@pytest.mark.parametrize("field", ["run", "precision", "anchor", "frames"])
def test_arbitrary_in_memory_phase2_metadata_is_not_a_sealed_configuration(sealed_metadata, field):
    _old, _fresh, original, _rows = sealed_metadata
    config = deepcopy(original)
    if field == "frames":
        config[field][0]["source_binding"]["source_sha256"] = "e" * 64
    elif field == "anchor":
        config[field]["affected_pore_volume_m3"] *= 2
    else:
        config[field] = "different"
    with pytest.raises(ValueError, match="actual sealed phase-two configuration bytes"):
        _call(sealed_metadata, config=config)


def test_phase2_configuration_tampering_is_checked_from_actual_bytes(sealed_metadata):
    _old, _fresh, _config, rows = sealed_metadata
    target = rows[0].path
    target.write_bytes(target.read_bytes().replace(b"float32", b"float64"))
    with pytest.raises(ValueError, match="SHA|size"):
        _call(sealed_metadata)


def test_ambiguous_sealed_configuration_consumer_is_refused(sealed_metadata):
    rows = sealed_metadata[3]
    with pytest.raises(ValueError, match="Exactly one"):
        _call(sealed_metadata, rows=rows + rows)


def test_actual_original_phase1_payload_tampering_is_refused(sealed_metadata):
    old = sealed_metadata[0]
    Path(old["record"]["phase"]["labels"]["path"]).write_bytes(b"changed actual original labels")
    with pytest.raises(ValueError, match="SHA|size"):
        _call(sealed_metadata)


def test_only_byte_proven_descriptor_paths_are_normalized():
    old = dict(path="original/labels.png", sha256="a" * 64, size_bytes=17)
    fresh = dict(old, path="fresh/labels.png")
    source = dict(labels=fresh, geometry=dict(extent=[0., 1.]), role="unchanged")
    original = deepcopy(source)
    result = coupling._relocated(source, {coupling._identity(fresh): old})
    assert result == dict(source, labels=old)
    assert source == original
    changed = dict(fresh, sha256="b" * 64)
    assert coupling._relocated(changed, {coupling._identity(fresh): old}) == changed


def test_exact_semantic_comparison_keeps_geometry_type_and_source_identity():
    with pytest.raises(ValueError, match="geometry"):
        coupling._same(dict(shape=[2, 2], measured=1.), dict(shape=[2, 2], measured=True), "geometry")
    with pytest.raises(ValueError, match="source"):
        coupling._same(dict(source="a" * 64), dict(source="b" * 64), "source")


def test_statistics_comparison_ignores_only_gzip_container_time(tmp_path):
    raw = b'{"precision":"float32","value":1.125,"source_binding":{"source_sha256":"original"}}'
    original = _write(tmp_path / "original.json.gz", raw=gzip.compress(raw, mtime=1))
    replayed = _write(tmp_path / "replayed.json.gz", raw=gzip.compress(raw, mtime=2))
    assert original["sha256"] != replayed["sha256"]
    assert coupling._statistics_bytes(DataReader(), original) == raw
    assert coupling._statistics_bytes(DataReader(), replayed) == raw
    assert Path(original["path"]).read_bytes() != Path(replayed["path"]).read_bytes()


@pytest.mark.parametrize("raw", [b'{"value":NaN}', b'{"value":1,"value":2}', b'[]'])
def test_statistics_require_original_strict_json_even_inside_valid_gzip(tmp_path, raw):
    bound = _write(tmp_path / "statistics.json.gz", raw=gzip.compress(raw, mtime=1))
    with pytest.raises(ValueError):
        coupling._statistics_bytes(DataReader(), bound)


def test_statistics_container_tampering_still_fails_original_sha_admission(tmp_path):
    bound = _write(tmp_path / "statistics.json.gz", raw=gzip.compress(b'{"value":1.125}', mtime=1))
    path = Path(bound["path"])
    path.write_bytes(gzip.compress(b'{"value":1.125}', mtime=2))
    with pytest.raises(ValueError, match="SHA"):
        coupling._statistics_bytes(DataReader(), bound)


def test_statistics_comparison_preserves_exact_decoded_serialization(tmp_path):
    one = _write(tmp_path / "one.json.gz", raw=gzip.compress(b'{"value":1.125}', mtime=1))
    two = _write(tmp_path / "two.json.gz", raw=gzip.compress(b'{"value": 1.125}', mtime=1))
    assert coupling._statistics_bytes(DataReader(), one) != coupling._statistics_bytes(DataReader(), two)


def test_statistics_have_no_plain_json_fallback(tmp_path):
    bound = _write(tmp_path / "statistics.json.gz", raw=b'{"value":1.125}')
    with pytest.raises(ValueError, match="gzip container"):
        coupling._statistics_bytes(DataReader(), bound)


def _observation(pid):
    return dict(policy="darsia-scientific-opencv-ipp-off-v1", stage="synthetic-metadata-boundary",
        pid=pid, thread_id=pid + 1, native_thread_id=pid + 2, opencv_version="synthetic-version",
        use_ipp=False, use_ipp_not_exact=False, process_environment_opencv_ipp="disabled",
        observation_scope="executing-thread-at-boundary")


def _metadata_inventory(root, *, pid, change=None):
    """Only metadata and arbitrary bytes: never a fake scientific completion."""
    observation = _observation(pid)
    if change == "enabled_ipp":
        observation["use_ipp"] = True
    if change == "changed_boundary":
        observation["stage"] = "different-boundary"
    audit_value = dict(geometry=[[1., 0.], [0., 1.]], opencv_ipp=observation)
    if change == "unknown_field":
        audit_value["unexplained"] = "new"
    if change == "unadmitted_input":
        audit_value["source"] = dict(path=str(root / "different-input"), sha256="1" * 64)
    audit = _write(root / "native_audit.json", audit_value)
    pixels = _write(root / "native_log.npy", raw=b"synthetic-payload" if change != "changed_pixels" else b"changed-payload")
    statistics = _write(root / "native_statistics.json.gz", raw=gzip.compress(b'{"exact":1.125}', mtime=pid))
    native = _write(root / "native.json", dict(audit=audit, log_linear_rgb=pixels))
    history = _write(root / "history_input.json", dict(native=native, statistics=statistics))
    # Deliberately list parents before their metadata dependencies.
    return dict(files={"history_input.json": history, "native.json": native,
        "native_audit.json": audit, "native_log.npy": pixels, "native_statistics.json.gz": statistics},
        statistics=statistics)


def test_metadata_graph_keeps_actual_receipts_and_normalizes_only_verified_dependencies(tmp_path):
    before = _metadata_inventory(tmp_path / "old", pid=10)
    after = _metadata_inventory(tmp_path / "new", pid=20)
    stats = dict(original=before["statistics"], replayed=after["statistics"])
    assert coupling._statistics_bytes(DataReader(), before["statistics"]) == coupling._statistics_bytes(DataReader(), after["statistics"])
    replacements, receipt = coupling._frame_metadata(before, after, DataReader(), DataReader(), stats)
    original = DataReader().json(before["files"]["history_input.json"])
    regenerated = DataReader().json(after["files"]["history_input.json"])
    assert coupling._metadata_relocated(regenerated, replacements) == original
    assert receipt["original_metadata_retained"] is True
    assert receipt["rewritten_metadata_supplied_to_science"] is False
    assert receipt["opencv_observations"][0]["original"]["pid"] == 10
    assert receipt["opencv_observations"][0]["replayed"]["pid"] == 20
    assert before["files"]["native_audit.json"]["sha256"] != after["files"]["native_audit.json"]["sha256"]


@pytest.mark.parametrize("change", ["enabled_ipp", "changed_boundary", "unknown_field", "changed_pixels"])
def test_metadata_graph_refuses_every_unexplained_runtime_or_scientific_change(tmp_path, change):
    before = _metadata_inventory(tmp_path / "old", pid=10)
    after = _metadata_inventory(tmp_path / "new", pid=20, change=change)
    with pytest.raises(ValueError):
        coupling._frame_metadata(before, after, DataReader(), DataReader(),
            dict(original=before["statistics"], replayed=after["statistics"]))


def test_metadata_graph_needs_genuine_input_admission_for_changed_external_paths(tmp_path):
    before = _metadata_inventory(tmp_path / "old", pid=10, change="unadmitted_input")
    after = _metadata_inventory(tmp_path / "new", pid=20, change="unadmitted_input")
    with pytest.raises(ValueError, match="genuine P1 input admission"):
        coupling._frame_metadata(before, after, DataReader(), DataReader(),
            dict(original=before["statistics"], replayed=after["statistics"]))


def test_metadata_graph_rejects_missing_generated_inventory_leaf(tmp_path):
    before = _metadata_inventory(tmp_path / "old", pid=10)
    after = _metadata_inventory(tmp_path / "new", pid=20)
    after["files"].pop("native_log.npy")
    with pytest.raises(ValueError, match="complete generated frame inventory"):
        coupling._frame_metadata(before, after, DataReader(), DataReader(),
            dict(original=before["statistics"], replayed=after["statistics"]))


def _phase_qa_value(scale, *, backend="cpu"):
    """Producer-shaped metadata only, with no scientific execution claim.

    Top/floor keys follow phase.rules.apply_top_rule/apply_floor_round;
    phase.infer retains those dictionaries directly in its final QA document.
    """
    top = dict(
        removed_top_components=1, excluded_columns=0, acting_columns=4, top_median_row=0.,
        all_pass=True, after_a_co2_union_equals_base=True, after_a_gas_equals_p1=True,
        changed_pixels_exactly_top_flags=True, removed_components_closed=True,
        removed_components_within_fixed_band=True, labels_outside_masks_identical=True,
        rig_domain_unchanged=True, removed_floor_pixels=0, removed_top_pixels=1,
        removed_floor_cm2=0., removed_top_aqueous_cm2=.125, removed_top_gas_cm2=0.,
        nominal_removed_top_aqueous_cm2=.1, nominal_removed_top_gas_cm2=0.,
        remaining_co2_60mm_cm2=0., remaining_co2_90mm_cm2=.25, retained_60mm_reason="none",
        backend=backend, cpu_seconds=.01 * scale, wall_seconds=.02 * scale,
        gpu_seconds=.005 * scale if backend == "cuda" else 0., training=False, inference=False)
    floor = dict(
        floor_median_row=3., acting_columns=4, excluded_columns=0, co2_connectivity_groups=1,
        floor_connectivity_removed_groups=0,
        floor_connectivity_removed_group_min_distance_to_kept_co2_mm=[],
        floor_connectivity_distance_status="not applicable: no removed group",
        floor_connectivity_distance_median_mm=None, floor_connectivity_distance_max_mm=None,
        floor_connectivity_removed_gas_pixels=0, fill_reason_a_columns=0, fill_reason_b_columns=0,
        carryover_mask_known=False, filled_under_gas_pixels=0, fill_invalid_trigger_columns=0,
        fill_unsupported_pixels=0, preinjection_fill_pixels=0, backend=backend,
        requested_backend=backend, cuda_fallback_reason=None,
        step_pixels={name: 0 for name in "ABCD"},
        step_area_cm2={name: 0. for name in "ABCD"},
        step_previous_aqueous_cm2={name: 0. for name in "ABCD"},
        step_previous_gas_cm2={name: 0. for name in "ABCD"},
        fill_reason_a_cm2=0., fill_reason_b_cm2=0.,
        mask_overlaps_pixels={name: 0 for name in ("AB", "AC", "AD", "BC", "BD", "CD")},
        mask_overlaps_on_changed_pixels={name: 0 for name in ("AB", "AC", "AD", "BC", "BD", "CD")},
        unchanged_stage_restoration_pixels=0, changed_pixels=0, flag_not_final_changed_pixels=0,
        changed_unflagged_pixels=0, labels_outside_masks_identical=True,
        gas_changed_outside_a_d_pixels=0, co2_union_changed_outside_a_b_c_pixels=0,
        domain_changed_outside_a_pixels=0, floor_boundary_matches=True,
        fill_flags_are_new_aqueous=True, gas_flags_are_gas_to_aqueous=True,
        size_filter_reapplied=False, training=False, inference=False, all_pass=True,
        cpu_seconds=.03 * scale, wall_seconds=.04 * scale,
        gpu_seconds=.006 * scale if backend == "cuda" else 0.)
    return dict(run="run-one", elapsed_h=1., source_sha256="c" * 64,
        p1=dict(changed_pixels=0), top=top, floor=floor,
        gas_envelopes=dict(u1_elapsed_h=1., u2_elapsed_h=2.), envelope_ordinal=0,
        parent_u1_u2_camera_times_preserved=True,
        native_inference=dict(source_sha256="c" * 64),
        edge_inference=dict(inference_performed=True),
        source_history=dict(previous_source_sha256="b" * 64, selected=True,
                            studio_anchor=False, policy="original selected acquisition"),
        base_seam=dict(rule_b=[]), barrier_geometry=dict(unchanged=True),
        physical_scale_revision=dict(area_factor=1.),
        phase_area=dict(co2_pixels=2, gas_pixels=1, aqueous_pixels=1, co2_cm2=.25,
                        gas_cm2=.125, aqueous_cm2=.125, physical_scale_approved=True,
                        mass_computed=False))


def _phase_qa_inventory(root, value, *, pid, declared=True, name="phase_qa.json", change=None):
    record = _metadata_inventory(root, pid=pid, change=change)
    qa = _write(root / name, value)
    # A parent descriptor exercises dependency-first comparison even when the
    # changed QA container is encountered through another generated document.
    parent = _write(root / "phase_metadata.json", dict(qa=qa))
    record["files"] = {"phase_metadata.json": parent, **record["files"], name: qa}
    if declared:
        record["phase"] = dict(qa=qa)
    return record


def _compare_metadata(before, after):
    return coupling._frame_metadata(before, after, DataReader(), DataReader(),
        dict(original=before["statistics"], replayed=after["statistics"]))


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_phase_qa_preserves_both_producer_timings_and_exact_historical_metadata(tmp_path, backend):
    original = _phase_qa_value(1., backend=backend)
    replayed = _phase_qa_value(2., backend=backend)
    before = _phase_qa_inventory(tmp_path / "old", original, pid=10)
    after = _phase_qa_inventory(tmp_path / "new", replayed, pid=20)
    actual_bytes = {side: Path(record["phase"]["qa"]["path"]).read_bytes()
                    for side, record in (("old", before), ("new", after))}

    replacements, receipt = _compare_metadata(before, after)

    timings = receipt["phase_rule_timing_observations"]
    assert [row["where"] for row in timings] == ["phase_qa.json.top", "phase_qa.json.floor"]
    assert [row["producer"] for row in timings] == [
        "phase.rules.apply_top_rule", "phase.rules.apply_floor_round"]
    assert receipt["differing_runtime_fields"] == ["pid", "thread_id", "native_thread_id"] + [
        "phase_qa.json." + rule + "." + field for rule in ("top", "floor")
        for field in ("cpu_seconds", "wall_seconds", "gpu_seconds")]
    for rule, row in zip(("top", "floor"), timings):
        fields = ("cpu_seconds", "wall_seconds", "gpu_seconds")
        assert row["original"] == {key: original[rule][key] for key in fields}
        assert row["replayed"] == {key: replayed[rule][key] for key in fields}
        assert row["original_file"] == before["phase"]["qa"]
        assert row["replayed_file"] == after["phase"]["qa"]
    assert before["phase"]["qa"]["sha256"] != after["phase"]["qa"]["sha256"]
    assert replacements[coupling._identity(after["phase"]["qa"])] == before["phase"]["qa"]
    assert receipt["original_metadata_retained"] is True
    assert receipt["rewritten_metadata_supplied_to_science"] is False
    for side, record in (("old", before), ("new", after)):
        assert Path(record["phase"]["qa"]["path"]).read_bytes() == actual_bytes[side]


@pytest.mark.parametrize("path,value", [
    (("top", "removed_top_pixels"), 2),
    (("top", "all_pass"), False),
    (("floor", "step_area_cm2", "A"), .125),
    (("floor", "changed_pixels"), False),  # Preserve int/bool distinctions.
    (("phase_area", "gas_cm2"), .25),
    (("top", "backend"), "cuda"),
    (("floor", "requested_backend"), "cuda"),
    (("floor", "cuda_fallback_reason"), "a different execution path"),
    (("source_sha256",), "d" * 64),
    (("source_history", "previous_source_sha256"), "e" * 64),
    (("elapsed_h",), 2.),
])
def test_phase_qa_timing_exception_preserves_scientific_source_and_backend_fields(tmp_path, path, value):
    original, replayed = _phase_qa_value(1.), _phase_qa_value(2.)
    target = replayed
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    before = _phase_qa_inventory(tmp_path / "old", original, pid=10)
    after = _phase_qa_inventory(tmp_path / "new", replayed, pid=20)
    with pytest.raises(ValueError, match="Staged reproduction differs"):
        _compare_metadata(before, after)


@pytest.mark.parametrize("declared,name", [(False, "phase_qa.json"), (True, "other_qa.json")])
def test_phase_rule_timings_require_the_exact_declared_phase_qa_file(tmp_path, declared, name):
    before = _phase_qa_inventory(tmp_path / "old", _phase_qa_value(1.), pid=10,
                                 declared=declared, name=name)
    after = _phase_qa_inventory(tmp_path / "new", _phase_qa_value(2.), pid=20,
                                declared=declared, name=name)
    with pytest.raises(ValueError, match="Staged reproduction differs"):
        _compare_metadata(before, after)


@pytest.mark.parametrize("path", [("cpu_seconds",), ("p1", "wall_seconds"),
                                  ("native_inference", "gpu_seconds")])
def test_phase_qa_timing_like_names_outside_direct_top_floor_remain_exact(tmp_path, path):
    values = [_phase_qa_value(1.), _phase_qa_value(2.)]
    for scale, value in enumerate(values, 1):
        target = value
        for field in path[:-1]:
            target = target[field]
        target[path[-1]] = float(scale)
    before = _phase_qa_inventory(tmp_path / "old", values[0], pid=10)
    after = _phase_qa_inventory(tmp_path / "new", values[1], pid=20)
    with pytest.raises(ValueError, match="Staged reproduction differs"):
        _compare_metadata(before, after)


@pytest.mark.parametrize("rule", ["top", "floor"])
@pytest.mark.parametrize("change", ["missing_timing", "unknown_field", "missing_rule"])
def test_phase_qa_requires_the_known_complete_producer_schema(tmp_path, rule, change):
    values = [_phase_qa_value(1.), _phase_qa_value(2.)]
    for value in values:
        if change == "missing_timing":
            value[rule].pop("gpu_seconds")
        elif change == "unknown_field":
            value[rule]["unrecognized_seconds"] = 1.
        else:
            value.pop(rule)
    before = _phase_qa_inventory(tmp_path / "old", values[0], pid=10)
    after = _phase_qa_inventory(tmp_path / "new", values[1], pid=20)
    with pytest.raises(ValueError, match="producer|top/floor"):
        _compare_metadata(before, after)


@pytest.mark.parametrize("value", [None, True, 0, -1., "0.0"])
def test_phase_qa_invalid_timing_types_and_values_refuse(tmp_path, value):
    original, replayed = _phase_qa_value(1.), _phase_qa_value(2.)
    replayed["floor"]["wall_seconds"] = value
    before = _phase_qa_inventory(tmp_path / "old", original, pid=10)
    after = _phase_qa_inventory(tmp_path / "new", replayed, pid=20)
    with pytest.raises(ValueError, match="execution timing observation"):
        _compare_metadata(before, after)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_phase_rule_timing_policy_rejects_nonfinite_observations(value):
    # Real metadata readers also reject nonfinite JSON before this policy.
    qa = _phase_qa_value(1.)["top"]
    qa["cpu_seconds"] = value
    with pytest.raises(ValueError, match="execution timing observation"):
        coupling._phase_rule_timing_values(qa, "top")


def test_phase_qa_timing_exception_still_requires_exact_binary_scientific_payloads(tmp_path):
    before = _phase_qa_inventory(tmp_path / "old", _phase_qa_value(1.), pid=10)
    after = _phase_qa_inventory(tmp_path / "new", _phase_qa_value(2.), pid=20, change="changed_pixels")
    with pytest.raises(ValueError, match="complete file bytes"):
        _compare_metadata(before, after)


def test_phase_qa_timing_observations_still_require_their_actual_file_sha(tmp_path):
    before = _phase_qa_inventory(tmp_path / "old", _phase_qa_value(1.), pid=10)
    after = _phase_qa_inventory(tmp_path / "new", _phase_qa_value(2.), pid=20)
    path = Path(after["phase"]["qa"]["path"])
    raw = path.read_bytes()
    changed = raw.replace(b'"cpu_seconds": 0.02', b'"cpu_seconds": 0.03', 1)
    assert changed != raw and len(changed) == len(raw)
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="SHA"):
        _compare_metadata(before, after)


def test_phase_qa_source_descriptor_relocation_still_requires_genuine_admission(tmp_path):
    values = [_phase_qa_value(1.), _phase_qa_value(2.)]
    for side, value in zip(("old", "new"), values):
        value["native_inference"]["source"] = dict(path=str(tmp_path / side / "source.npy"),
                                                  sha256="a" * 64, size_bytes=17)
    before = _phase_qa_inventory(tmp_path / "old", values[0], pid=10)
    after = _phase_qa_inventory(tmp_path / "new", values[1], pid=20)
    with pytest.raises(ValueError, match="genuine P1 input admission"):
        _compare_metadata(before, after)


@pytest.mark.parametrize("change", [None, "unknown_join", "changed_original_provenance"])
def test_n1_storage_accepts_only_the_existing_typed_common_join_metadata(tmp_path, phase1_data, change):
    """Storage metadata only; the original catalogue still verifies the C join."""
    from darsia.fluidflower.pipeline.common_calibration import JOIN_FORMAT
    from darsia.fluidflower.storage_n1_outputs import Phase2Storage
    phase1_data["statistics"]["provenance"] = {"original_method": "unchanged"}
    original = _closure(tmp_path / "original", phase1_data)
    target = tmp_path / "phase2.json"
    sealed = staged_data.seal_phase2_configuration(target, original["completion"], original["configuration"],
        expected_phase1_commit="1" * 40, fitted_calibration=original["calibration"],
        fitted_anchor=dict(elapsed_h=2.25, affected_pore_volume_m3=.0001), reader=DataReader())
    config = DataReader().json(sealed["configuration"])
    config["provenance"]["common_frozen_refit"] = dict(format=JOIN_FORMAT)
    if change == "unknown_join":
        config["provenance"]["common_frozen_refit"]["format"] = "not-a-recognized-join"
    elif change == "changed_original_provenance":
        config["provenance"]["original_method"] = "changed"
    if change is None:
        context = Phase2Storage(config, DataReader(), dict(resolution_mm=1., precision="float32"))
        assert context.phase1["statistics"]["provenance"] == {"original_method": "unchanged"}
        assert context.config is config  # No input mutation, new calibration or execution admission.
    else:
        with pytest.raises(ValueError, match="provenance|scientific DATA"):
            Phase2Storage(config, DataReader(), dict(resolution_mm=1., precision="float32"))
