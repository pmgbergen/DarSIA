"""TESTONLY: a phase-two n1 task admitted by rule under a qualified native-phase-n1 profile.

n1 computes mass from stored phase-one products with the qualified evaluation and
publisher. A full n1 task on the qualified head is admitted as another run: the
measured mass-frame bytes per frame plus upper bounds for its declared options.
The cache side admits it when the n1 boundary modules are in the qualified
closure. Synthetic metadata only.
"""
import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import storage_extensive, storage_outputs, storage_resolution
from darsia.fluidflower.pipeline import capacity as cap
from darsia.fluidflower.pipeline import n1_capacity as n1
from darsia.fluidflower.pipeline import option_capacity as oc
from darsia.fluidflower.phase import probability_storage as ps
from darsia_cache import control
from darsia_cache.policy import CachePolicyError
from .test_capacity import bound_output_profile_TESTONLY, native_layout
from .test_bound_profile_other_runs import bound_cache_TESTONLY, other_cache_task_TESTONLY

POLICY = dict(contract="darsia-pixel-domain-v1", adopted_for_execution=True, fixture_scope_complete=True,
              fixture_names=["TESTONLY"], injector_hardware_excluded=False,
              authorized_decisions={"TESTONLY": "decision"}, max_column_distance_m=.02, raster_scale=256,
              raster_shift=8, geometry_chunk_pixels=100, below_floor_phase_unknown_policy="unknown-no-capacity",
              include_estimated_mass=False, baseline_coverage_tolerance=1e-6, heldout_fit_run=False,
              table_fields=["M_measured_g", "M_edge_g", "U_g", "phase_unknown_no_capacity_area_cm2"])
STORAGE = dict(format=storage_outputs.REQUEST_FORMAT, resolution_mm=1,
               gas_score_encoding=storage_outputs.LA_ENCODING, extent_m=[0, 0.006, 0, 0.003])


def descriptor_TESTONLY(size):
    return dict(path="TESTONLY", sha256="0" * 64, size_bytes=size)


def n1_task_TESTONLY(*, run="ac17", count=41):
    """The qualified 39-frame profile, and an n1 task of another run on the qualified canvas."""
    qualified, _, wrapped = bound_output_profile_TESTONLY()
    task = dict(qualified, pipeline=n1.PIPELINE, run=run, attempt_id="2" * 32,
                ordered_case_ids=[run + "_f" + str(index) for index in range(1, count + 1)],
                configuration=dict(qualified["configuration"], sha256="e" * 64))
    canvas = native_layout()["phase_shape_hw"]
    frames = [dict(case_id=case, current=dict(phase_shape_hw=list(canvas))) for case in task["ordered_case_ids"]]
    phase1 = dict(completion=descriptor_TESTONLY(900), frames={
        case: dict(commit=descriptor_TESTONLY(70), before_rules=dict(
            state="complete", receipt=descriptor_TESTONLY(30), png=descriptor_TESTONLY(500)))
        for case in task["ordered_case_ids"]})
    config = dict(format="darsia-n1-run-v1", run=run, precision="float32", frames=frames, phase1=phase1,
                  shared=dict(porosity_by_facies={"1": 0.3, "2": 0.4}))
    return task, config, wrapped


def estimate_TESTONLY(task, config, wrapped, monkeypatch, *, pins=None):
    """The bound-profile estimate of an n1 task, its writers synthetic."""
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: pytest.fail("n1 has no native layout"))
    monkeypatch.setattr(cap.subprocess, "check_output", lambda args: ("TESTONLY " + args[-1]).encode())
    monkeypatch.setattr(cap, "_writer_identity",
                        lambda repository, commit, row: (pins if pins is not None else []).append(row))
    return cap._bound_result_estimate(None, "TESTONLY", task, config, wrapped, {},
                                      repository="TESTONLY", metadata_rows=None)


def test_the_requests_are_the_n1_options_checked_by_their_writers():
    assert n1.validate_requests({}) == ()
    assert n1.validate_requests(dict(precision="float32", pixel_domain=POLICY, storage=STORAGE)) == (
        "pixel_domain", "storage")
    assert n1.validate_requests(dict(precision="float32", storage=dict(STORAGE, resolution_mm="native"))) == ()
    with pytest.raises(ValueError, match="Pixel-domain decisions"):
        n1.validate_requests(dict(pixel_domain=dict(POLICY, contract="TESTONLY")))
    for key in n1.REFUSED:
        with pytest.raises(ValueError, match="Optional output " + key):
            n1.validate_requests({key: dict(format=ps.REQUEST_FORMAT, encoding=ps.ENCODING)})


def test_every_frame_must_declare_the_qualified_canvas():
    measured = native_layout()
    frames = [dict(current=dict(phase_shape_hw=list(measured["phase_shape_hw"])))] * 2
    assert n1.canvas(dict(frames=frames), measured) == tuple(measured["phase_shape_hw"])
    with pytest.raises(ValueError, match="qualified phase canvas"):
        n1.canvas(dict(frames=frames + [dict(current=dict(phase_shape_hw=[3, 7]))]), measured)


def test_the_mass_frame_bytes_leave_out_the_stage_and_the_rewritten_archives():
    _, _, wrapped = bound_output_profile_TESTONLY()
    actual = wrapped["result_profile"]
    inventory, ids = actual["measurement"]["inventory"], actual["measured_case_ids"]
    assert n1.mass_frame_bytes(inventory, ids) == (53 + 1, 53)  # manifest.json spread over the frames
    with pytest.raises(ValueError, match="no mass-frame files"):
        n1.mass_frame_bytes(inventory, ids, excluded=n1.ARCHIVES)


def test_the_pixel_domain_bound_counts_both_archives_at_exact_dtypes():
    value = n1.pixel_domain_bytes(dict(pixel_domain=POLICY), 100, "float32")
    products = sum(n1._itemsize(kind, "float32") for kind in n1.PIXEL_PRODUCTS.values())
    assert products == 18 + 1 + 10 * 4 + 2 * 4
    bounded_products = dict(n1.PIXEL_PRODUCTS, approved_analysis_gap_mask=1)
    assert value == (oc.npz_bound(dict(n1._payloads(bounded_products, 100, "float32"),
                                       **{name: n1.SCALAR_BYTES for name in n1.PRODUCT_SCALARS}))
                     + oc.npz_bound(n1._payloads(n1.PIXEL_GEOMETRY, 100, "float32")))
    hidden = n1.pixel_domain_bytes(dict(pixel_domain=dict(POLICY, n16={})), 100, "float32")
    assert hidden - value > 100 * (6 + 4 + 5 * 4 + 6 * 8)


def test_the_hidden_mass_tables_are_what_the_rule_adds():
    from darsia.fluidflower.mass import pixel_pipeline, pixels
    shape = (2, 2)
    ones, zeros, field = np.ones(shape, bool), np.zeros(shape, bool), np.full(shape, 0.1)
    pixel = dict(masks=dict(measured=ones), per_pixel_mass_g=dict(measured=np.full(shape, 0.5)),
                 maps=dict(mass_aqueous=np.full(shape, 0.25, np.float32), mass_free_gas=np.zeros(shape, np.float32)))
    masses = pixels.measured_component_maps(pixel, np.ones(shape, np.uint8), np.ones(shape))
    returned = [row.value for node in ast.walk(ast.parse(inspect.getsource(pixel_pipeline._prepare_n16)))
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
                for row in node.value.keywords if row.arg == "geometry"]
    names = [key.arg for key in returned[0].keywords]
    n16 = dict(hidden=zeros, masks=dict(physical=ones, valid=ones),
               unknown=dict(U_per_pixel_g=np.zeros(shape), U_g=0.0, U_layer=np.zeros(shape), receipt={}),
               bottom=dict(estimated_mask=zeros, zero_mask=zeros, unknown_mask=zeros,
                           donor_rows=np.full(shape, -1, np.int32), donor_distance_m=np.zeros(shape)),
               bottom_mass=np.zeros(shape), edge=np.zeros(shape), M_edge_g=0.0, measured_masses=masses,
               capacity_inputs=dict(solubility_g_per_l=field), geometry={name: field for name in names},
               policy=dict(contract="TESTONLY", authority="TESTONLY", capacity_selector="TESTONLY"),
               authority={}, observation={}, precision={})
    extra = {name: zeros.copy() for name in ("concentration_unknown_mask", "unknown_without_capacity_mask",
                                             "concentration_estimated_mask", "mass_contribution_mask")}
    extra.update(unknown_capacity_g=np.zeros(shape, np.float32), edge_mass_g=np.zeros(shape, np.float32))
    geometry = dict(n1.PIXEL_GEOMETRY)
    evaluated = dict(result=dict(M_measured_g=1.0, domains=dict(full_rig={})), prediction=dict(injected_mass_g=None),
                     extra_products=dict(extra), geometry_fields=dict.fromkeys(geometry, field),
                     pixel_domain={}, statistics={})
    pixel_pipeline._apply_n16(evaluated, n16, SimpleNamespace(math=SimpleNamespace(dtype=np.float32)),
                              {}, dict(full_rig=ones))
    added = {name: value for name, value in evaluated["extra_products"].items() if name not in extra}
    assert set(added) == set(n1.N16_PRODUCTS)
    for name, value in added.items():
        assert np.asarray(value).dtype.itemsize <= n1._itemsize(n1.N16_PRODUCTS[name], "float32"), name
    assert set(evaluated["geometry_fields"]) - set(geometry) == set(n1.N16_GEOMETRY)


def test_the_coarse_tables_cover_the_real_reducers():
    shape = (2, 2)
    x = np.array([[0.0005, 0.0015], [0.0005, 0.0015]])
    y = np.array([[0.0005, 0.0005], [0.0015, 0.0015]])
    grid = storage_resolution.make_grid(x, y, np.full(shape, 1e-6), include=np.ones(shape, bool),
                                        extent_m=[0, 0.002, 0, 0.002], storage_resolution_mm=1, precision="float32")
    labels = np.array([[0, 1], [2, 255]], np.uint8)
    masses = {name: np.full(shape, 0.25) for name in storage_resolution.MASS_FIELDS}
    masks = {"m" + str(index): np.ones(shape, bool) for index in range(n1.COARSE_MASKS)}
    frame = storage_resolution.aggregate_frame(
        grid, concentration_aqueous=np.ones(shape), saturation_free_gas=np.zeros(shape), phase_labels=labels,
        masks=masks, probabilities={storage_resolution.PROBABILITY_FIELDS[0]: np.full(shape, 0.5),
                                    storage_resolution.PROBABILITY_FIELDS[1]: labels == 2},
        native_pixel_masses_g=masses, native_mass_totals_g={name: 1.0 for name in masses})["arrays"]
    static = storage_resolution.aggregate_static(grid, depth_m=np.ones(shape), porosity=np.full(shape, 0.3),
                                                 facies=np.ones(shape, np.int64))
    assert len(frame) == n1.COARSE_FRAME_FIELDS + n1.COARSE_MASKS
    assert max(len((name + ".npy").encode()) for name in [*frame, *("static_" + key for key in static)]) <= n1.NAME_BYTES
    assert all(value.dtype.itemsize <= 4 for value in frame.values())
    sizes = sorted(np.asarray(value).dtype.itemsize for value in static.values())
    assert sizes == sorted([4] * n1.COARSE_STATIC["precision"] + [8] * n1.COARSE_STATIC["int64"]
                           + [1] * n1.COARSE_STATIC["boolean"])


def test_the_extensive_table_covers_the_real_reducer_with_every_channel():
    shape = (2, 2)
    x = np.array([[0.0005, 0.0015], [0.0005, 0.0015]])
    y = np.array([[0.0005, 0.0005], [0.0015, 0.0015]])
    area = np.full(shape, 1e-6)
    grid = storage_resolution.make_grid(x, y, area, include=np.ones(shape, bool),
                                        extent_m=[0, 0.002, 0, 0.002], storage_resolution_mm=1, precision="float32")
    ones, zeros = np.ones(shape, bool), np.zeros(shape, bool)
    depth, phi = np.full(shape, 0.02), np.full(shape, 0.4)
    pv = area * depth * phi
    labels = np.array([[1, 1], [2, 2]], np.uint8)
    saturation = np.where(labels == 2, 0.5, 0.0)
    facies = np.array([[1, 2], [1, 2]], np.int64)
    masses = dict(total=np.full(shape, 0.5), aqueous=np.full(shape, 0.25), gas=np.full(shape, 0.25))
    channel = lambda: dict(pixel_values_g=np.full(shape, 0.1), support_mask=ones, native_total_g=0.4,
                           producer_binding=dict(TESTONLY=True))
    product = storage_extensive.aggregate_extensive(
        grid, depth_m=depth, porosity=phi, pixel_pore_volume_m3=pv, facies=facies, physical_mask=ones,
        physical_properties_valid_mask=ones, mass_support_mask=ones, hidden_mask=zeros,
        coverage_masks=dict(measured=ones, estimated=zeros, hidden=zeros),
        intensive_validity_masks=dict.fromkeys(storage_extensive.INTENSIVE, ones), phase_labels=labels,
        saturation_free_gas=saturation, saturation_valid_mask=ones, native_pixel_masses_g=masses,
        native_mass_totals_g={name: float(value.sum()) for name, value in masses.items()},
        native_extensive_totals=dict(physical_area_m2=float(area.sum()), bulk_volume_m3=float((area * depth).sum()),
                                     pore_volume_m3=float(pv.sum()),
                                     gas_pore_volume_m3=float((saturation * pv).sum())),
        channels={name: channel() for name in storage_extensive.CHANNELS}, provenance={})
    arrays, facies_count = product["arrays"], 2
    assert max(len((name + ".npy").encode()) for name in arrays) <= n1.NAME_BYTES
    real = sorted(value.nbytes for value in arrays.values())
    assert real == sorted(n1.extensive_sizes(4, "float32", facies_count)[:-1] + [4 * 8])
    kinds = [value.dtype for value in arrays.values()]
    assert sum(kind == np.float32 for kind in kinds) == n1.EXTENSIVE["precision"] + 2 * facies_count
    assert sum(kind == np.int64 for kind in kinds) == n1.EXTENSIVE["int64"] + 1  # The int64 facies majority.
    assert sum(kind == bool for kind in kinds) == n1.EXTENSIVE["boolean"]
    assert len(arrays) == (n1.EXTENSIVE["precision"] + 2 * facies_count + n1.EXTENSIVE["int64"]
                           + n1.EXTENSIVE["boolean"] + 1)


def test_the_storage_bound_counts_the_copies_the_coarse_frame_and_the_extensive_payload():
    task, config, _ = n1_task_TESTONLY()
    cfg = dict(config, storage=STORAGE)
    plain = n1.storage_bytes(cfg, canvas_hw=(3, 6), precision="float32", extensive=False)
    cells = 3 * 6  # 0.006 x 0.003 m at 1 mm.
    assert n1.coarse_sizes(cells, "float32") == [cells * 4] * (14 + 8 + 6) + [cells * 8] * 3 + [cells]
    assert plain == 900 + (70 + 30 + 500) + n1.anonymous_npz_bound(n1.coarse_sizes(cells, "float32")) + n1.RECEIPT_BYTES
    extensive = n1.storage_bytes(cfg, canvas_hw=(3, 6), precision="float32", extensive=True)
    assert extensive - plain == n1.anonymous_npz_bound(n1.extensive_sizes(cells, "float32", 2)) + n1.RECEIPT_BYTES
    broken = dict(cfg, phase1=dict(config["phase1"], completion=dict(path="TESTONLY")))
    with pytest.raises(ValueError, match="exact descriptor sizes"):
        n1.storage_bytes(broken, canvas_hw=(3, 6), precision="float32", extensive=False)


def test_an_n1_task_is_admitted_with_the_mass_frame_bytes(monkeypatch):
    task, config, wrapped = n1_task_TESTONLY()
    pins = []
    value = estimate_TESTONLY(task, config, wrapped, monkeypatch, pins=pins)
    assert type(value) is cap.OptionSpaceEstimate and value.case_count == 41
    assert value.measured_native_bytes_per_frame == 54
    assert value.products_per_frame_estimated_bytes == 54 + n1.RECORD_BYTES
    run_bytes = oc.run_file_bytes(wrapped["result_profile"]["measurement"]["inventory"],
                                 wrapped["result_profile"]["measured_case_ids"])
    retained = 3 * (41 * (53 + n1.RECORD_BYTES) + run_bytes)
    assert value.retained_serialization_estimated_bytes == retained
    assert value.transient_estimated_bytes == 53 + n1.RECORD_BYTES + retained
    assert value.options["route"] == "n1" and value.options["declared"] == []
    assert [row["relative_path"] for row in pins] == list(n1.WRITERS)


def test_the_pixel_domain_replaces_the_measured_archives(monkeypatch):
    task, config, wrapped = n1_task_TESTONLY()
    rows = wrapped["result_profile"]["measurement"]["inventory"]
    extra = [dict(relative_path="frames/" + case + "/result.json", bytes=5) for case in wrapped["scope"]["ordered_case_ids"]]
    wrapped["result_profile"]["measurement"].update(inventory=rows + extra,
        total_logical_bytes=wrapped["result_profile"]["measurement"]["total_logical_bytes"] + 5 * len(extra))
    pins = []
    value = estimate_TESTONLY(task, dict(config, pixel_domain=dict(POLICY, n16={})), wrapped, monkeypatch, pins=pins)
    height, width = native_layout()["phase_shape_hw"]
    assert value.measured_native_bytes_per_frame == 5 + 1  # products.npz left out; result.json and manifest kept.
    assert value.options["parts"]["pixel_domain"]["frame_bytes"] == n1.pixel_domain_bytes(
        dict(pixel_domain=dict(POLICY, n16={})), height * width, "float32")
    assert [row["relative_path"] for row in pins] == [*n1.WRITERS, *n1.PIXEL_WRITERS]


def test_storage_and_n16_add_the_extensive_payload(monkeypatch):
    task, config, wrapped = n1_task_TESTONLY()
    measurement = wrapped["result_profile"]["measurement"]
    extra = [dict(relative_path="frames/" + case + "/result.json", bytes=5) for case in wrapped["scope"]["ordered_case_ids"]]
    measurement.update(inventory=measurement["inventory"] + extra,
                       total_logical_bytes=measurement["total_logical_bytes"] + 5 * len(extra))
    pins = []
    value = estimate_TESTONLY(task, dict(config, pixel_domain=dict(POLICY, n16={}), storage=STORAGE), wrapped,
                              monkeypatch, pins=pins)
    assert value.options["declared"] == ["pixel_domain", "storage"]
    assert value.options["parts"]["storage"]["extensive"] is True
    assert [row["relative_path"] for row in pins][-len(n1.EXTENSIVE_WRITERS):] == list(n1.EXTENSIVE_WRITERS)


@pytest.mark.parametrize("key", sorted(n1.REFUSED))
def test_phase_one_outputs_are_refused_on_n1_before_any_read(monkeypatch, key):
    task, config, wrapped = n1_task_TESTONLY()
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: pytest.fail("No writer read"))
    with pytest.raises(ValueError, match="qualified head and route: Optional output " + key):
        cap._bound_result_estimate(None, "TESTONLY", task,
                                   dict(config, **{key: dict(format=ps.REQUEST_FORMAT, encoding=ps.ENCODING)}),
                                   wrapped, {}, repository="TESTONLY", metadata_rows=None)


def test_an_n1_frame_off_the_qualified_canvas_is_refused(monkeypatch):
    task, config, wrapped = n1_task_TESTONLY()
    config["frames"][0] = dict(config["frames"][0], current=dict(phase_shape_hw=[3, 7]))
    with pytest.raises(ValueError, match="qualified head and route: Every n1 frame must declare"):
        estimate_TESTONLY(task, config, wrapped, monkeypatch)


@pytest.mark.parametrize("key,value", [("backend", "cuda"), ("precision", "float64"), ("commit", "f" * 40)])
def test_n1_never_opens_another_head(monkeypatch, key, value):
    task, config, wrapped = n1_task_TESTONLY()
    task[key] = value
    with pytest.raises(ValueError, match="qualified head and route"):
        estimate_TESTONLY(task, config, wrapped, monkeypatch)


def test_the_series_route_is_unchanged_by_the_n1_rule(monkeypatch):
    qualified, config, wrapped = bound_output_profile_TESTONLY()
    monkeypatch.setattr(n1, "n1_estimate", lambda *args, **kwargs: pytest.fail("Not for the series route"))
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: native_layout())
    value = cap._bound_result_estimate(None, "TESTONLY", qualified, config, wrapped, {},
                                       repository="TESTONLY", metadata_rows=None)
    assert type(value) is cap.SpaceEstimate


def n1_cache_TESTONLY(f):
    """An n1 task on the qualified head, with its boundary modules in the qualified closure."""
    root = f.adapter.runtime.repository / "src" / "darsia" / "fluidflower"
    names = [root / "pipeline" / name for name in ("staged_data.py", "job_inputs.py", "n1_capacity.py")]
    names += [root / "mass" / "pixel_pipeline.py", root / "mass" / "pixels.py", root / "geometry" / "pixels.py"]
    names += [root / name for name in ("storage_extensive.py", "storage_extensive_io.py", "storage_n1_outputs.py",
                                       "storage_n1_saved_products.py", "storage_outputs.py", "storage_resolution.py")]
    pins = [dict(path=str(path), sha256="1" * 64, size_bytes=1) for path in names]
    return dict(other_cache_task_TESTONLY(f), pipeline="n1"), pins


def test_the_cache_side_admits_a_full_n1_task_with_its_boundary_in_the_closure(bound_cache_TESTONLY):
    f = bound_cache_TESTONLY
    task, pins = n1_cache_TESTONLY(f)
    with pytest.raises(CachePolicyError, match="startup writer/source/runtime closure"):
        control.applicable_job_cache_profile(f.policy, f.adapter, task, None)
    f.profile["producer_source_closure"] = f.profile["producer_source_closure"] + pins
    f.adapter.lease.profile["module_closure"] = f.adapter.lease.profile["module_closure"] + pins
    assert control.applicable_job_cache_profile(f.policy, f.adapter, task, None)[0] == 0


@pytest.mark.parametrize("change", [dict(storage_profile="lean", format_version=2), dict(backend="cuda"),
                                    dict(commit="4" * 40)])
def test_the_cache_side_refuses_a_lean_cuda_or_other_head_n1_task(bound_cache_TESTONLY, change):
    f = bound_cache_TESTONLY
    task, pins = n1_cache_TESTONLY(f)
    f.profile["producer_source_closure"] = f.profile["producer_source_closure"] + pins
    f.adapter.lease.profile["module_closure"] = f.adapter.lease.profile["module_closure"] + pins
    with pytest.raises(CachePolicyError, match="qualified head and route"):
        control.applicable_job_cache_profile(f.policy, f.adapter, dict(task, **change), None)
    assert f.adapter.profiles == {}
