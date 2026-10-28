"""Saved-only reader gates using the Q34 v5 output-only producer on tiny synthetic data."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import storage_outputs as output
from darsia.fluidflower import storage_saved_products as saved


def _json(path, value):
    path.write_bytes((json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode())


def _sha_size(path):
    return {k: v for k, v in output._asset(path).items() if k != "path"}


def _fixture(tmp_path, *, precision="float32", pending=False, case="ac52_f1",
             storage_resolution_mm=1., grid_origin=(0.,0.), extensive_inputs=None):
    root = tmp_path / "attempt"
    dtype = np.dtype(precision)
    stage = root / "preparation" / case
    stage.mkdir(parents=True)
    labels = np.array([[0, 1], [2, 255]], np.uint8)
    path = stage / "phase_labels.png"
    Image.fromarray(labels).save(path)
    phase = dict(labels=output._asset(path), source_sha256="d" * 64)
    eligible = labels != 255
    products = dict(mass_total=np.where(eligible, 3., 0.).astype(dtype),
                    mass_aqueous=np.where(eligible, 1., 0.).astype(dtype),
                    mass_free_gas=np.where(eligible, 1., 0.).astype(dtype),
                    phase_labels=labels.copy(), concentration_aqueous=np.ones(labels.shape, dtype),
                    saturation_free_gas=np.full(labels.shape, .5, dtype))
    frame = SimpleNamespace(phase_labels=labels.copy(), domain=eligible.copy(),
        pixel_area_m2=np.full(labels.shape, 1e-6), depth_m=np.full(labels.shape, .02, dtype),
        porosity=np.full(labels.shape, .3, dtype), facies=np.ones(labels.shape, np.uint8), excluded_masks={})
    result = dict(total_mass_g=9., aqueous_mass_g=3., gas_mass_g=3., precision=precision,
                  fields={"main_mass_eligible_mask": eligible.copy()})
    evaluated = dict(result=result, export={"products": products})
    capture = {} if pending else dict(g3_pixel=np.array([[0., 1.], [.5, np.nan]], np.float32),
        flatfield_valid=np.array([[True, True], [True, False]]), source_box_xywh=[0, 0, 2, 2])
    mass = dict(frame=frame, volume_m3=np.full(labels.shape, .001),
        x_m=grid_origin[0]+storage_resolution_mm*np.array([[.0005, .0015], [.0005, .0015]]),
        y_m=grid_origin[1]+storage_resolution_mm*np.array([[.0005, .0005], [.0015, .0015]]))
    config = dict(path="config.json", sha256="a" * 64, size_bytes=1)
    task = dict(job_id="synthetic", run="ac52", configuration=config,
                commit="b" * 40, calibration_sha256="c" * 64, precision=precision)
    source = dict(source_sha256="d" * 64, source_binding={"source_sha256": "d" * 64})
    spec = dict(case_id=case, source=source)
    request = output.parse_request({"storage": dict(format=output.REQUEST_FORMAT,
        resolution_mm=storage_resolution_mm, gas_score_encoding=output.LA_ENCODING,
        extent_m=[grid_origin[0],grid_origin[0]+.003*storage_resolution_mm,
                  grid_origin[1],grid_origin[1]+.003*storage_resolution_mm])}, precision)
    entry = output.write_frame(root, task, spec, phase,
        dict(labels=labels, classes={"rig_domain": eligible}), capture, mass, evaluated, request,
        **({} if extensive_inputs is None else {"extensive_inputs":
            _synthetic_extensive_inputs(labels, frame.pixel_area_m2,
                products["saturation_free_gas"]) if extensive_inputs is True else extensive_inputs}))
    binding = dict(format="darsia-series-frame-v1", complete=True, case_id=case,
        configuration_sha256=config["sha256"], source=source,
        native={"source_sha256": "d" * 64}, phase=phase,
        storage=entry, files={Path(row["path"]).name: row for row in [entry["receipt"], *entry["files"]]})
    native_result = dict(complete=True, run="ac52", case_id=case, precision=precision,
        working_pixel_dtype=precision, source_binding=source["source_binding"],
        bindings=dict(source_sha256="d" * 64, phase_sha256=phase["labels"]["sha256"],
            calibration_sha256="c" * 64, code_sha256=hashlib.sha256(("b" * 40).encode()).hexdigest()),
        total_mass_g=9., aqueous_mass_g=3., gas_mass_g=3., prediction={"pipeline": binding})
    manifest = dict(format="darsia-series-run-manifest-v1", job_id="synthetic", run="ac52",
        configuration=config, settings={"precision": precision}, code={"commit": "b" * 40}, frames={case: spec})
    manifest_path = root / "manifest.json"
    _json(manifest_path, manifest)
    directory = root / "frames" / case
    directory.mkdir(parents=True)
    record = dict(frame_id=case, nas_directory=str(directory), manifest_sha256=output._asset(manifest_path)["sha256"], files={})
    document = json.loads(Path(entry["receipt"]["path"]).read_bytes())
    state = dict(root=root, record=record, result=native_result, entry=entry, document=document,
                 manifest=manifest, manifest_path=manifest_path, directory=directory, calls=[])
    _reseal(state)
    return state


def _reseal(state):
    entry, doc = state["entry"], state["document"]
    doc["files"] = deepcopy(entry["files"])
    receipt = Path(entry["receipt"]["path"])
    _json(receipt, doc)
    entry["receipt"] = dict(output._asset(receipt), kind="operational_receipt")
    binding = state["result"]["prediction"]["pipeline"]
    binding["storage"] = deepcopy(entry)
    binding["files"] = {Path(v["path"]).name: v for v in [entry["receipt"], *entry["files"]]}
    directory = state["directory"]
    _json(directory / "result.json", state["result"])
    # This nonexistent native archive is deliberately never resolved/read.
    files = {"result.json": _sha_size(directory / "result.json"),
             "products.npz": dict(sha256="e" * 64, size_bytes=999)}
    commit = dict(complete=True, frame_id=state["record"]["frame_id"], job_id="synthetic",
        manifest_sha256=state["record"]["manifest_sha256"], products={"precision": state["result"]["precision"]},
        settings={"precision": state["result"]["precision"]}, files=files)
    _json(directory / "commit.json", commit)
    state["record"]["files"] = dict(files, **{"commit.json": _sha_size(directory / "commit.json")})


def _resolver(state):
    def resolve(bound):
        state["calls"].append(bound["path"])
        name = Path(bound["path"]).name
        assert name in {"manifest.json", "commit.json", "result.json", "storage_completed.json",
                         "storage_stored_frame.npz", "storage_gas_probability_before_rules.png",
                         *({"storage_extensive_frame.npz", "storage_extensive_frame.json"}
                           if state["document"].get("extensive") else set())}
        return Path(bound["path"])
    return resolve


def _synthetic_extensive_inputs(labels, area, saturation, *, hidden_last=False):
    """Explicit tiny synthetic physical inputs, never a dataset mask producer."""
    from darsia.fluidflower import storage_extensive as extensive
    from darsia.fluidflower import storage_extensive_io as payload
    shape = labels.shape
    physical = np.ones(shape, bool)
    hidden = np.zeros(shape, bool)
    if hidden_last:
        hidden.flat[-1] = True
    pv = np.asarray(area, np.float64) * .5
    return dict(format=payload.INPUT_FORMAT, qualification=payload.QUALIFICATION,
        physical_producer_admitted=False, physical_inputs=dict(
            depth_m=np.ones(shape), porosity=np.full(shape, .5), pixel_pore_volume_m3=pv,
            facies=np.ones(shape, np.uint8), physical_mask=physical,
            physical_properties_valid_mask=physical.copy(), mass_support_mask=physical.copy(),
            hidden_mask=hidden, coverage_masks=dict(measured=~hidden, estimated=np.zeros(shape, bool), hidden=hidden.copy()),
            intensive_validity_masks={name: ~hidden for name in extensive.INTENSIVE},
            saturation_valid_mask=np.ones(shape, bool),
            native_extensive_totals=dict(physical_area_m2=float(area.sum()), bulk_volume_m3=float(area.sum()),
                pore_volume_m3=float(pv.sum()), gas_pore_volume_m3=float((pv[labels == 2] * saturation[labels == 2]).sum())),
            channels=dict.fromkeys(extensive.CHANNELS), provenance=dict(synthetic_fixture=True,
                actual_physical_fields="OUTSTANDING", camera_clock=None)))


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_optional_extensive_native_writer_commit_reader_retains_independent_masses(tmp_path, precision):
    from darsia.fluidflower import storage_extensive_io as payload
    state = _fixture(tmp_path, precision=precision, extensive_inputs=True)
    coarse = saved.read_stored_frame(state["record"], resolve_asset=_resolver(state))
    assert not any(Path(p).name.startswith("storage_extensive") for p in state["calls"])
    value = payload.read_committed_payload(state["record"], producer_kind="native-phase-n1", resolve_asset=_resolver(state))
    assert value["state"] == "synthetic_io_only"
    assert value["physical_binding_admitted"] is value["producer_identity_admitted"] is False
    assert value["metadata"]["precision"] == precision
    for name, scalar in (("total", 9.), ("aqueous", 3.), ("gas", 3.)):
        assert value["arrays"]["cell_mass_" + name + "_g"].dtype == np.dtype(precision)
        assert value["arrays"]["cell_mass_" + name + "_g"].sum(dtype=np.float64) == scalar
    assert coarse["mass_totals_g"] == dict(total=9., aqueous=3., gas=3.)
    assert not any("cell_" + channel + "_g" in value["arrays"] for channel in ("M_maalt", "M_kant", "U"))


def test_optional_extensive_keeps_old_coarse_bytes_and_positive_hidden_PV_nan(tmp_path):
    from darsia.fluidflower import storage_extensive_io as payload
    labels = np.array([[0, 1], [2, 255]], np.uint8)
    explicit = _synthetic_extensive_inputs(labels, np.full((2, 2), 1e-6), np.full((2, 2), .5), hidden_last=True)
    old = _fixture(tmp_path / "old")
    state = _fixture(tmp_path / "new", extensive_inputs=explicit)
    for name in ("storage_stored_frame.npz", "storage_gas_probability_before_rules.png"):
        assert (old["root"] / "preparation/ac52_f1" / name).read_bytes() == (state["root"] / "preparation/ac52_f1" / name).read_bytes()
    value = payload.read_committed_payload(state["record"], producer_kind="native-phase-n1", resolve_asset=_resolver(state))
    arrays = value["arrays"]
    assert arrays["pore_volume_m3"][1, 1] > 0
    assert arrays["coverage_hidden_pore_volume_fraction"][1, 1] == 1
    assert np.isnan(arrays["concentration_aqueous"][1, 1])
    assert arrays["cell_mass_total_g"][1, 1] == 0


@pytest.mark.parametrize("bad", ["format", "qualification", "admission", "extra", "missing", "mass_override"])
def test_optional_extensive_input_is_closed_and_never_admits_a_physical_producer(tmp_path, bad):
    value = _synthetic_extensive_inputs(np.array([[0, 1], [2, 255]], np.uint8), np.full((2, 2), 1e-6), np.full((2, 2), .5))
    if bad == "format": value["format"] = "unknown"
    elif bad == "qualification": value["qualification"] = "production"
    elif bad == "admission": value["physical_producer_admitted"] = True
    elif bad == "extra": value["extra"] = None
    elif bad == "missing": del value["physical_inputs"]["hidden_mask"]
    else: value["physical_inputs"]["native_mass_totals_g"] = dict(total=1., aqueous=1., gas=0.)
    with pytest.raises(ValueError, match="closed synthetic-only"):
        _fixture(tmp_path, extensive_inputs=value)
    assert not list(tmp_path.rglob("storage_extensive*"))


@pytest.mark.parametrize("bad", ["source", "configuration", "precision", "admission", "archive", "grid", "totals", "metadata"])
def test_resealed_extensive_envelope_cannot_disagree_with_committed_native_identity(tmp_path, bad):
    from darsia.fluidflower import storage_extensive_io as payload
    state = _fixture(tmp_path, extensive_inputs=True)
    section = state["document"]["extensive"]
    path = Path(section["receipt"]["path"])
    envelope = json.loads(path.read_bytes())
    if bad == "admission": envelope["physical_producer_admitted"] = True
    elif bad == "archive": envelope["archive"]["sha256"] = "0" * 64
    elif bad == "grid": envelope["api_metadata"]["grid"]["extent_m"][0] = -1.
    elif bad == "totals": envelope["api_metadata"]["native_mass_totals_g"]["total"] = 6.
    elif bad == "metadata": envelope["api_metadata"] = None
    else: envelope["frame_identity"][bad] = "changed"
    _json(path, envelope)
    updated = dict(output._asset(path), kind="operational_receipt")
    section["receipt"] = updated
    state["entry"]["extensive"] = deepcopy(section)
    state["entry"]["files"] = [updated if Path(row["path"]).name == payload.RECEIPT else row for row in state["entry"]["files"]]
    _reseal(state)
    with pytest.raises(ValueError):
        payload.read_committed_payload(state["record"], producer_kind="native-phase-n1", resolve_asset=_resolver(state))


@pytest.mark.parametrize("bad", ["archive", "missing", "kind", "extra"])
def test_optional_extensive_cannot_escape_actual_native_inventory(tmp_path, bad):
    from darsia.fluidflower import storage_extensive_io as payload
    state = _fixture(tmp_path, extensive_inputs=True)
    if bad == "archive":
        path = Path(state["document"]["extensive"]["archive"]["path"])
        path.write_bytes(path.read_bytes() + b"changed")
    elif bad == "missing":
        state["entry"]["files"] = [r for r in state["entry"]["files"] if Path(r["path"]).name != payload.ARCHIVE]
        _reseal(state)
    elif bad == "kind":
        state["entry"]["extensive"]["archive"]["kind"] = "frame_scalar"
        state["document"]["extensive"] = deepcopy(state["entry"]["extensive"])
        _reseal(state)
    else:
        extra = dict(state["entry"]["files"][-1], path=str(state["root"] / "preparation/ac52_f1/storage_unknown.npz"))
        state["entry"]["files"].append(extra)
        _reseal(state)
    with pytest.raises(ValueError):
        payload.read_committed_payload(state["record"], producer_kind="native-phase-n1", resolve_asset=_resolver(state))


def test_native_resolution_cannot_activate_synthetic_extensive_payload(tmp_path):
    value = _synthetic_extensive_inputs(np.zeros((1, 1), np.uint8), np.ones((1, 1)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="coarse grid"):
        output.write_frame(None, None, None, None, None, None, None, None,
                           {"resolution_mm": "native"}, extensive_inputs=value)
    assert not list(tmp_path.iterdir())


def _reseal_extensive_array_values(section, mutate):
    """Re-anchor an internally valid synthetic archive, never its original coarse map."""
    from darsia.fluidflower import storage_extensive as extensive
    archive_path, receipt_path = Path(section["archive"]["path"]), Path(section["receipt"]["path"])
    envelope = json.loads(receipt_path.read_bytes())
    metadata = envelope["api_metadata"]
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    mutate(arrays)
    metadata["arrays"] = {key: extensive._identity(value) for key, value in arrays.items()}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    # The attack must pass the pure API's original schema and scalar checks;
    # only its disagreement with the independently committed cell map is new.
    extensive.read_extensive_archive(buffer.getvalue(), metadata,
        expected_input_identities=metadata["input_identities"],
        native_mass_totals_g=metadata["native_mass_totals_g"],
        native_extensive_totals=metadata["native_extensive_totals"],
        expected_channels=metadata["channels"], expected_provenance=metadata["provenance"],
        expected_grid_metadata=metadata["grid"])
    archive_path.write_bytes(buffer.getvalue())
    section["archive"] = dict(output._asset(archive_path), kind="map")
    envelope["archive"] = deepcopy(section["archive"])
    _json(receipt_path, envelope)
    section["receipt"] = dict(output._asset(receipt_path), kind="operational_receipt")
    return {Path(row["path"]).name: row for row in (section["archive"], section["receipt"])}


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("component", ["total", "aqueous", "gas"])
def test_resealed_extensive_spatial_mass_permutation_cannot_replace_native_coarse_map(tmp_path, precision, component):
    from darsia.fluidflower import storage_extensive_io as payload
    state = _fixture(tmp_path, precision=precision, extensive_inputs=True)
    def permute(arrays):
        key = "cell_mass_" + component + "_g"
        before = arrays[key].copy()
        arrays[key][0, 0], arrays[key][1, 1] = before[1, 1], before[0, 0]
        assert not np.array_equal(arrays[key], before)
        assert arrays[key].sum(dtype=np.float64) == before.sum(dtype=np.float64)
        if component == "aqueous":
            mean = arrays["concentration_aqueous"].copy()
            arrays["concentration_aqueous"][0, 0], arrays["concentration_aqueous"][1, 1] = mean[1, 1], mean[0, 0]
    section = state["document"]["extensive"]
    updated = _reseal_extensive_array_values(section, permute)
    state["entry"]["extensive"] = deepcopy(section)
    state["entry"]["files"] = [updated.get(Path(row["path"]).name, row) for row in state["entry"]["files"]]
    _reseal(state)
    with pytest.raises(ValueError, match="Extensive " + component + " cell masses differ from committed coarse arrays"):
        payload.read_committed_payload(state["record"], producer_kind="native-phase-n1", resolve_asset=_resolver(state))


def _alter_archive(state, mutate):
    row = next(v for v in state["entry"]["files"] if Path(v["path"]).name == "storage_stored_frame.npz")
    path = Path(row["path"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    mutate(arrays)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.write_bytes(buffer.getvalue())
    row.update(output._asset(path))
    state["document"]["coarse"]["arrays"] = {key: output._array_identity(value) for key, value in arrays.items()}
    _reseal(state)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_real_producer_roundtrip_preserves_nan_precision_orientation_and_independent_mass(tmp_path, precision):
    state = _fixture(tmp_path, precision=precision)
    result = saved.read_stored_frame(state["record"], resolve_asset=_resolver(state))
    assert result["precision"] == precision and result["imshow_origin"] == "lower"
    assert result["grid"]["extent_m"] == [0., .003, 0., .003]
    assert result["mass_totals_g"] == dict(total=9., aqueous=3., gas=3.)
    assert result["mass_totals_g"]["total"] != result["mass_totals_g"]["aqueous"] + result["mass_totals_g"]["gas"]
    arrays = result["arrays"]
    assert arrays["concentration_aqueous"].dtype == np.dtype(precision)
    assert np.isnan(arrays["concentration_aqueous"][2]).all()
    assert arrays["gas_probability_after_rules"][1, 1] == 0  #255 outside N1 still contributes a non-gas vote.
    assert not arrays["static_cell_has_area"][1, 1]
    assert result["before_native"] is None
    assert not any(p.endswith(".png") for p in state["calls"])
    assert all(not a.flags.writeable for a in arrays.values())


def test_before_png_is_explicit_and_keeps_valid_zero_one_and_missing(tmp_path):
    state = _fixture(tmp_path)
    result = saved.read_stored_frame(state["record"], resolve_asset=_resolver(state), load_before=True)
    before = result["before_native"]
    assert before[0, 0] == 0 and before[0, 1] == 1 and before[1, 0] == np.float32(128 / 255)
    assert np.isnan(before[1, 1]) and not before.flags.writeable
    assert sum(p.endswith("storage_gas_probability_before_rules.png") for p in state["calls"]) == 1


def test_pending_before_does_not_disable_final_mass_or_invent_probability(tmp_path):
    state = _fixture(tmp_path, pending=True)
    result = saved.read_stored_frame(state["record"], resolve_asset=_resolver(state), load_before=True)
    assert result["before_rules"]["state"] == "pending" and result["before_native"] is None
    assert "gas_probability_before_rules" not in result["arrays"]
    assert result["mass_totals_g"]["total"] == 9 and not any(p.endswith(".png") for p in state["calls"])


@pytest.mark.parametrize("which", ["sha", "size"])
def test_changed_native_commit_descriptor_is_rejected(tmp_path, which):
    state = _fixture(tmp_path)
    row = state["record"]["files"]["commit.json"]
    row["sha256" if which == "sha" else "size_bytes"] = "0" * 64 if which == "sha" else row["size_bytes"] + 1
    with pytest.raises(ValueError, match="Stored product"):
        saved.read_stored_frame(state["record"], resolve_asset=_resolver(state))


@pytest.mark.parametrize("field", ["run", "source", "configuration", "calibration_sha256", "precision"])
def test_resealed_storage_identity_cannot_disagree_with_original_native_result(tmp_path, field):
    state = _fixture(tmp_path)
    if field == "source": state["document"][field]["source_sha256"] = "f" * 64
    elif field == "configuration": state["document"][field]["sha256"] = "f" * 64
    elif field == "calibration_sha256": state["document"][field] = "f" * 64
    elif field == "precision": state["document"][field] = "float64"
    else: state["document"][field] = "ac19"
    _reseal(state)
    with pytest.raises(ValueError, match="identity differs|source/configuration"):
        saved.read_stored_frame(state["record"], resolve_asset=_resolver(state))


@pytest.mark.parametrize("change", ["dtype", "shape", "unknown", "mass", "support"])
def test_resealed_coarse_arrays_still_need_exact_schema_precision_and_native_closure(tmp_path, change):
    state = _fixture(tmp_path)
    def mutate(arrays):
        if change == "dtype": arrays["concentration_aqueous"] = arrays["concentration_aqueous"].astype(np.float64)
        elif change == "shape": arrays["concentration_aqueous"] = arrays["concentration_aqueous"][:2]
        elif change == "unknown": arrays["unbound_extra"] = np.zeros((3, 3), np.float32)
        elif change == "mass": arrays["cell_mass_total_g"][0, 0] += 1
        else: arrays["concentration_aqueous"][0, 0] = np.nan
    _alter_archive(state, mutate)
    with pytest.raises(ValueError, match="precision|identity|Unknown coarse|mass closure|finite-area support"):
        saved.read_stored_frame(state["record"], resolve_asset=_resolver(state))


def test_changed_archive_bytes_are_not_accepted_even_if_numpy_can_open_it(tmp_path):
    state = _fixture(tmp_path)
    row = next(v for v in state["entry"]["files"] if v["path"].endswith(".npz"))
    p = Path(row["path"])
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Stored product"):
        saved.read_stored_frame(state["record"], resolve_asset=_resolver(state))


def test_resealed_extent_mismatch_is_not_a_display_regrid(tmp_path):
    state = _fixture(tmp_path)
    state["document"]["coarse"]["grid"]["extent_m"][0] = -.001
    _reseal(state)
    with pytest.raises(ValueError, match="grid/extent"):
        saved.read_stored_frame(state["record"], resolve_asset=_resolver(state))


def test_missing_local_coarse_asset_identifies_collect_without_native_fallback(tmp_path):
    state = _fixture(tmp_path)
    resolve = _resolver(state)
    def unavailable(bound):
        if bound["path"].endswith("storage_stored_frame.npz"):
            raise FileNotFoundError(bound["path"])
        return resolve(bound)
    with pytest.raises(saved.StoredProductCollectionRequired) as failure:
        saved.read_stored_frame(state["record"], resolve_asset=unavailable)
    assert failure.value.descriptor["path"].endswith("storage_stored_frame.npz")
    assert len(failure.value.descriptor["sha256"]) == 64


def test_run_uses_original_manifest_identity_and_never_sums_acquisition_times(tmp_path):
    state = _fixture(tmp_path)
    completion = dict(complete=True, pipeline="native-phase-n1", run="ac52", precision="float32", frames=1,
        case_count=1, ordered_case_ids=["ac52_f1"], records=[state["record"]],
        manifest=output._asset(state["manifest_path"]), configuration=state["manifest"]["configuration"],
        calibration={"sha256": "c" * 64})
    result = saved.read_stored_run(completion, resolve_asset=_resolver(state))
    assert result["ordered_case_ids"] == ["ac52_f1"] and len(result["frames"]) == 1
    assert "mass_totals_g" not in result
    completion.update(frames=2, case_count=2, records=completion["records"] * 2, ordered_case_ids=["ac52_f1"] * 2)
    with pytest.raises(ValueError, match="run/order"):
        saved.read_stored_run(completion, resolve_asset=_resolver(state))


@pytest.mark.parametrize('change',['dtype','negative','full_sum','included_exceeds_full','missing'])
def test_q34_resealed_per_cell_count_contract_cannot_be_weakened(tmp_path,change):
    state=_fixture(tmp_path)
    def mutate(arrays):
        included='static_included_area_native_pixel_count'
        canvas='static_full_canvas_probability_pixel_count'
        if change=='dtype': arrays[canvas]=arrays[canvas].astype(np.float32)
        elif change=='negative': arrays[included][0,0]=-1
        elif change=='full_sum': arrays[canvas][2,2]+=1
        elif change=='included_exceeds_full': arrays[included][1,1]=arrays[canvas][1,1]+1
        elif change=='missing': del arrays[canvas]
    _alter_archive(state,mutate)
    with pytest.raises(ValueError):saved.read_stored_frame(state['record'],resolve_asset=_resolver(state))


@pytest.mark.parametrize('field',['cell_count_fields','probability_coordinate_counts','assignment'])
def test_q34_resealed_count_or_half_open_semantics_are_exact(tmp_path,field):
    state=_fixture(tmp_path)
    state['document']['coarse']['grid'][field]={} if field!='assignment' else 'clamp outside pixels'
    _reseal(state)
    with pytest.raises(ValueError,match='grid/extent/orientation/precision'):
        saved.read_stored_frame(state['record'],resolve_asset=_resolver(state))


def test_q34_saved_counts_include_excluded_canvas_and_empty_cells(tmp_path):
    state=_fixture(tmp_path)
    result=saved.read_stored_frame(state['record'],resolve_asset=_resolver(state)); arrays=result['arrays']
    np.testing.assert_array_equal(arrays['static_included_area_native_pixel_count'],
        np.array([[1,1,0],[1,0,0],[0,0,0]],np.int64))
    np.testing.assert_array_equal(arrays['static_full_canvas_probability_pixel_count'],
        np.array([[1,1,0],[1,1,0],[0,0,0]],np.int64))
    assert arrays['static_pixel_area_m2_sum'][1,1]==0.


@pytest.mark.parametrize('extent',[[.0001,.0031,0.,.003],[0.,.0025,0.,.003]])
def test_q34_resealed_request_and_grid_cannot_shift_or_fractionally_size_lattice(tmp_path,extent):
    state=_fixture(tmp_path)
    state['document']['request']['extent_m']=extent
    state['document']['coarse']['grid']['extent_m']=extent
    _reseal(state)
    with pytest.raises(ValueError,match='exact origin-zero metric lattice'):
        saved.read_stored_frame(state['record'],resolve_asset=_resolver(state))


def test_q34_saved_negative_origin_fractional_h_real_writer_roundtrip(tmp_path):
    state=_fixture(tmp_path,storage_resolution_mm=.25,grid_origin=(-.001,.00025))
    result=saved.read_stored_frame(state['record'],resolve_asset=_resolver(state))
    assert result['grid']['shape']==[3,3]
    assert result['grid']['extent_m']==[-.001,-.00025,.00025,.001]
    assert result['grid']['resolution_mm']==.25
    assert result['mass_totals_g']==dict(total=9.,aqueous=3.,gas=3.)


def test_q34_saved_decimal_integer_edges_do_not_add_a_float_ceil_column(tmp_path):
    state=_fixture(tmp_path,grid_origin=(.3,0.))
    result=saved.read_stored_frame(state['record'],resolve_asset=_resolver(state))
    assert result['grid']['shape']==[3,3]
    assert result['grid']['extent_m']==[.3,.303,0.,.003]
