"""Closed synthetic photo/model/frame inputs for the new placement policy."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import photo_edge_continuation as policy
from darsia.fluidflower import map_placement as placement
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower.series_geometry import RigFrame


def _bound(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _frame(run, photo, shape):
    height, width = shape
    return RigFrame(corners=dict(top_left=(0., 0.), top_right=(0., width-1.),
        bottom_right=(height-1., width-1.), bottom_left=(height-1., 0.)),
        width_m=(width-1)*.001, height_m=(height-1)*.001, run_id=run,
        source="private saved frame", image=str(photo), image_size=(width, height))


@pytest.fixture
def closed_case(tmp_path):
    series, root = tmp_path / "series", tmp_path / "project"
    (series / "config").mkdir(parents=True)
    root.mkdir()
    models = root / "models"
    models.mkdir()
    (series / "series.toml").write_text('schema_version=1\n[runs]\nids=["source", "target"]\n'
        '[runs.config_files]\nsource="config/source.toml"\ntarget="config/target.toml"\n', encoding="utf-8")
    (series / "config/common.toml").write_text("# private common configuration\n", encoding="utf-8")
    photos, frames, entries = {}, {}, {}
    shapes = dict(source=(8, 10), target=(10, 16))
    for run, shape in shapes.items():
        height, width = shape
        photo = series / (run + ".png")
        Image.new("RGB", (width, height), (50, 80, 110)).save(photo)
        photos[run] = photo
        frame = _frame(run, photo, shape)
        frames[run] = frame.to_dict()
        (series / "config" / (run + ".toml")).write_text(
            f'[data]\nfolder="{series.as_posix()}"\nbaseline="{run}.png"\n'
            f'[rig]\nwidth={frame.width_m}\nheight={frame.height_m}\n'
            '[corrections.curvature.crop]\n' + ''.join(f'{key}={list(value)}\n' for key, value in frame.corners.items()), encoding="utf-8")
        h = [[1., 0., 0. if run == "source" else 3.], [0., 1., 0. if run == "source" else 1.], [0., 0., 1.]]
        model = models / (run + ".json")
        model.write_text(json.dumps(dict(format=placement.GEOMETRY_FORMAT, run_id=run,
            reference_image=_bound(photo), shape_hw=list(shape), H_metric_mm_to_camera_u=h,
            baseline_native_grid=dict(shape_hw=list(shape), H_baseline_camera_u_to_rig_corrected_native=np.eye(3).tolist()))), encoding="utf-8")
        entries[run] = dict(reference_model=_bound(model))
    index = models / "index.json"
    index.write_text(json.dumps(dict(format=placement.GEOMETRY_FORMAT, runs=entries)), encoding="utf-8")
    manifest = dict(project_id="private-project", tools=dict(native_series=dict(
        series_root=str(series), series_id="private-series", rig_frames=frames)))
    project = SimpleNamespace(root=root, manifest_path=root / "series.json", project_id="private-project", manifest=manifest)
    project.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source = np.full(shapes["source"], 2, dtype=np.uint8)
    return SimpleNamespace(project=project, series=series, index=index, photos=photos,
        source=source, models=models, shapes=shapes, entries=entries)


def _contract(case, *, grid="camera_u"):
    return policy.build_contracts(case.project, case.index, "source", ["target"], [2, 3], target_grid=grid)["target"]


def test_authorized_continuation_changes_no_labels_or_original_provenance(closed_case):
    case = closed_case
    contract = _contract(case)
    models = placement.read_run_geometry(case.index)
    original, filled = placement.place_layer_map(case.source, models["source"], models["target"], case.shapes["target"])
    source_bytes, original_bytes, filled_bytes = case.source.tobytes(), original.tobytes(), filled.tobytes()
    value = policy.apply_policy(case.source, contract, labels=original, filled=filled)
    assert value["counts"]["continued_pixels"] == 80
    assert value["counts"]["inside_gap_pixels"] == 0
    assert value["continued"][0, 0] and value["continued"][9, 15]
    assert not value["continued"][4, 7]
    assert not value["support"][0, 0] and value["effective_coverage"][0, 0]
    assert value["support"][4, 7] and value["effective_coverage"][4, 7]
    assert not np.any(value["support"] & value["continued"])
    assert np.array_equal(value["inside"], value["effective_coverage"] | value["gaps"])
    assert case.source.tobytes() == source_bytes and original.tobytes() == original_bytes and filled.tobytes() == filled_bytes
    assert value["labels"].dtype == original.dtype and value["labels"].tobytes() == original_bytes
    assert value["filled"].tobytes() == filled_bytes


@pytest.mark.parametrize("donor", [0, 7])
def test_zero_or_unknown_photo_edge_donor_stays_a_gap(closed_case, donor):
    case = closed_case
    case.source[:, 0] = donor
    value = policy.apply_policy(case.source, _contract(case))
    assert value["labels"][4, 0] == donor
    assert value["filled"][4, 0] and not value["continued"][4, 0]
    assert value["gaps"][4, 0] and not value["effective_coverage"][4, 0]
    assert value["continued"][4, 15], "The other edge's known donor still continues"
    assert not value["gaps"][4, 7]


def test_inside_photo_hole_is_not_filled_or_classified_as_continuation(closed_case):
    case = closed_case
    case.source[3, 4] = 0
    value = policy.apply_policy(case.source, _contract(case))
    assert value["labels"][4, 7] == 0 and value["gaps"][4, 7]
    assert not value["filled"][4, 7] and not value["continued"][4, 7]
    assert value["counts"]["inside_gap_pixels"] == 1


@pytest.mark.parametrize('hole', [None, 'interior', 'boundary'])
def test_projective_native_frame_centres_sample_boundary_without_filling_real_holes(closed_case, tmp_path, hole):
    from darsia.fluidflower import layer_painting as lp
    from darsia.fluidflower.painting_preview_frame_domain import _project
    case = closed_case
    corners = np.asarray([[2.5, 0.], [8.5, 0.], [8.5, 7.], [1.1, 7.]], float)
    source_frame = _frame('source', case.photos['source'], case.shapes['source'])
    source_frame.corners = {name: tuple(point[::-1]) for name, point in zip(source_frame.corners, corners)}
    inside_source = lp.frame_mask(source_frame, case.source.shape, (1., 1.))
    source = np.where(inside_source, 2, 0).astype(np.uint8)
    # Destination centre (5,2) maps to source (2.45,.49), physically INSIDE
    # the sloping left edge. Ordinary nearest selects (row0,col2), OUTSIDE
    # the frozen source raster. Its nearest boundary centre is (row0,col3).
    q = np.asarray([2.45, .49])
    composition = np.asarray([[1.15, .04, 0.], [-.02, 1.05, 0.], [.004, -.002, 1.]])
    denominator = composition[2, :2] @ q + 1.
    composition[:2, 2] = np.asarray([5., 2.]) * denominator - composition[:2, :2] @ q
    native = np.asarray([[.8, .03, .4], [.01, .85, .2], [.0005, .0008, 1.]])
    camera = np.linalg.inv(native) @ composition
    target_frame = _frame('target', case.photos['target'], case.shapes['target'])
    target_corners = _project(corners, camera)
    target_frame.corners = {name: tuple(point[::-1]) for name, point in zip(target_frame.corners, target_corners)}
    frames = case.project.manifest['tools']['native_series']['rig_frames']
    frames.update(source=source_frame.to_dict(), target=target_frame.to_dict())
    case.project.manifest_path.write_text(json.dumps(case.project.manifest), encoding='utf8')
    path = case.models / 'target.json'
    model = json.loads(path.read_bytes())
    model['H_metric_mm_to_camera_u'] = camera.tolist()
    model['baseline_native_grid'] = dict(shape_hw=[9, 13], H_baseline_camera_u_to_rig_corrected_native=native.tolist())
    path.write_text(json.dumps(model), encoding='utf8')
    index = json.loads(case.index.read_bytes())
    index['runs']['target']['reference_model'] = _bound(path)
    case.index.write_text(json.dumps(index), encoding='utf8')
    if hole == 'interior':
        source[3:5, 5:7] = 0
    elif hole == 'boundary':
        source[0, 3] = 0  # Never search past an unknown nearest boundary donor.
    contract = _contract(case, grid='native')
    legacy = deepcopy(contract)
    legacy.pop('frame_raster'); legacy.pop('source_frame')
    original_contract = json.dumps(legacy, sort_keys=True)
    old = policy.apply_policy(source, legacy)
    assert old['gaps'][2, 5]
    result = policy.apply_policy(source, contract)
    assert result['inside'][2, 5]
    assert result['counts']['frame_boundary_sampled_pixels'] > 0
    if hole is None:
        assert not result['gaps'].any()
        assert result['labels'][2, 5] == 2 and result['support'][2, 5]
    else:
        assert result['gaps'].any()
        if hole == 'boundary':
            assert result['gaps'][2, 5] and result['labels'][2, 5] == 0
    path = tmp_path / 'new-native-product.npz'
    np.savez_compressed(path, array=result['labels'].astype(np.int32), metadata={},
        **{key: result[key] for key in ('filled', 'support', 'inside', 'continued', 'effective_coverage', 'gaps')})
    policy.verify_product(path, source, contract, counts=result['counts'])
    assert json.dumps(legacy, sort_keys=True) == original_contract
    replay = policy.apply_policy(source, legacy)
    for field in ('labels', 'filled', 'support', 'inside', 'gaps'):
        np.testing.assert_array_equal(replay[field], old[field])


def test_new_physical_frame_raster_marker_is_explicit_and_unknown_rule_refuses(closed_case):
    contract = _contract(closed_case)
    assert contract['frame_raster'] == policy.FRAME_RASTER
    assert contract['source_frame'] == closed_case.project.manifest['tools']['native_series']['rig_frames']['source']
    contract['frame_raster'] = 'unsupported alternative domain'
    with pytest.raises(ValueError, match='physical-frame raster rule'):
        policy.apply_policy(closed_case.source, contract)


def test_continuation_stops_at_target_own_frame_not_the_projected_reference_frame(closed_case):
    case = closed_case
    frame = case.project.manifest["tools"]["native_series"]["rig_frames"]["target"]
    frame["corners"] = dict(top_left=[2., 1.], top_right=[2., 14.], bottom_right=[8., 14.], bottom_left=[8., 1.])
    case.project.manifest_path.write_text(json.dumps(case.project.manifest), encoding="utf-8")
    value = policy.apply_policy(case.source, _contract(case))
    assert value["filled"][4, 0] and not value["inside"][4, 0] and not value["continued"][4, 0]
    assert value["inside"][4, 1] and value["continued"][4, 1]
    assert not value["inside"][0, 7] and not value["gaps"][0, 7]


def test_native_policy_uses_exact_existing_native_transport(closed_case):
    case = closed_case
    path = case.models / "target.json"
    model = json.loads(path.read_bytes())
    model["baseline_native_grid"] = dict(shape_hw=[5, 8],
        H_baseline_camera_u_to_rig_corrected_native=[[.5, 0., 0.], [0., .5, 0.], [0., 0., 1.]])
    path.write_text(json.dumps(model), encoding="utf-8")
    index = json.loads(case.index.read_bytes())
    index["runs"]["target"]["reference_model"] = _bound(path)
    case.index.write_text(json.dumps(index), encoding="utf-8")
    models = placement.read_run_geometry(case.index)
    old, filled, support, h = adoption.native_labels(case.source, models["source"], models["target"])
    value = policy.apply_policy(case.source, _contract(case, grid="native"), labels=old, filled=filled, support=support)
    assert value["labels"].shape == (5, 8) and np.array_equal(value["labels"], old)
    assert np.array_equal(value["support"], support) and np.array_equal(value["filled"], filled)
    assert value["continued"][2, 0] and not value["support"][2, 0] and value["effective_coverage"][2, 0]
    assert h == [[.5, 0., 1.5], [0., .5, .5], [0., 0., 1.]]


def test_configured_own_frame_is_replayed_from_exact_frozen_toml(closed_case):
    case = closed_case
    del case.project.manifest["tools"]["native_series"]["rig_frames"]["target"]
    case.project.manifest_path.write_text(json.dumps(case.project.manifest), encoding="utf-8")
    contract = _contract(case)
    assert contract["frame_authority"]["kind"] == "series configuration"
    assert contract["own_frame"]["image_size"] is None
    assert policy.apply_policy(case.source, contract)["counts"]["continued_pixels"] == 80


@pytest.mark.parametrize("field", ["labels", "filled", "support"])
def test_carrier_provenance_mismatch_is_refused_without_input_mutation(closed_case, field):
    case = closed_case
    contract = _contract(case)
    original = policy.apply_policy(case.source, contract)
    changed = original[field].copy()
    changed[0, 0] = (3 if field == "labels" else not changed[0, 0])
    before = changed.tobytes()
    with pytest.raises(ValueError, match="original categorical replay"):
        policy.apply_policy(case.source, contract, **{field: changed})
    assert changed.tobytes() == before


@pytest.mark.parametrize("failure", ["frame", "frame_sha", "source_grid", "model", "index", "authority_bytes"])
def test_source_or_frame_drift_fails_closed(closed_case, failure):
    case = closed_case
    contract = _contract(case)
    if failure == "frame":
        contract["own_frame"]["corners"]["top_left"][0] += 1.
    elif failure == "frame_sha":
        contract["frame_sha256"] = "0" * 64
    elif failure == "source_grid":
        contract["source_model"]["shape_hw"] = [4, 5]
    elif failure == "model":
        (case.models / "target.json").write_text("{}", encoding="utf-8")
    elif failure == "index":
        case.index.write_text("{}", encoding="utf-8")
    else:
        contract["frame_authority"]["project"]["text"] += " "
    with pytest.raises((ValueError, KeyError)):
        policy.apply_policy(case.source, contract)


def test_a_freshly_bound_model_declaring_another_run_is_refused(closed_case):
    case = closed_case
    path = case.models / "target.json"
    value = json.loads(path.read_bytes())
    value["run"] = "another-run"
    path.write_text(json.dumps(value), encoding="utf-8")
    index = json.loads(case.index.read_bytes())
    index["runs"]["target"]["reference_model"] = _bound(path)
    case.index.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="declares another run"):
        _contract(case)


def test_capture_drift_blocks_new_writer_but_frozen_historical_authority_survives_unrelated_project_metadata(closed_case, tmp_path):
    case = closed_case
    contract = _contract(case)
    output = tmp_path / "continuation.json"
    policy.save_contracts(output, {"target": contract})
    original = output.read_bytes()
    case.project.manifest["unrelated_note"] = "private later metadata"
    case.project.manifest_path.write_text(json.dumps(case.project.manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source metadata changed"):
        policy.save_contracts(tmp_path / "must-not-exist.json", {"target": contract})
    assert not (tmp_path / "must-not-exist.json").exists()
    restored = policy.load_contract(output, "target")
    assert policy.apply_policy(case.source, restored)["counts"]["continued_pixels"] == 80
    assert output.read_bytes() == original


def test_pooled_sidecar_preserves_closed_contracts_without_mutating_inputs(closed_case, tmp_path):
    """New serialization shares metadata and reloads the original admitted ABI."""
    contracts = {"target": _contract(closed_case)}
    original = deepcopy(contracts)
    path = tmp_path / "pooled.json"
    policy.save_contracts(path, contracts)
    encoded = json.loads(path.read_bytes())
    assert encoded["format"] == policy.SIDECAR_FORMAT
    assert len(encoded["pool"]) == 3  # Two distinct models and one project snapshot.
    expected = dict(format=policy.FORMAT, policy=policy.POLICY, runs=original, adopted=False)
    assert policy.read_contracts(path) == expected
    assert policy.load_contract(path, "target") == original["target"]
    assert contracts == original


def test_moved_continuation_and_models_keep_original_closed_contract(closed_case, tmp_path, monkeypatch):
    import toml
    from darsia.fluidflower import storage_paths
    case = closed_case
    contract = _contract(case)
    old_folder = case.models
    sidecar = old_folder / "continuation.json"
    policy.save_contracts(sidecar, {"target": contract})
    original_bytes = sidecar.read_bytes()
    original_contract = deepcopy(contract)
    moved = tmp_path / "retained-models"
    old_folder.rename(moved)
    registry = tmp_path / "registry.toml"
    registry.write_text(toml.dumps(dict(schema_version=1,
        storage=dict(path_aliases={str(old_folder): str(moved)}))), encoding="utf-8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    storage_paths.refresh_storage_paths()

    assert policy.load_contract(sidecar, "target") == original_contract
    assert (moved / sidecar.name).read_bytes() == original_bytes
    assert not old_folder.exists()
    # The original model SHA is still authoritative after relocation.
    model = moved / "target.json"
    model.write_bytes(model.read_bytes() + b" ")
    with pytest.raises(ValueError, match="reference model|measured model changed"):
        policy.load_contract(sidecar, "target")


def test_legacy_full_sidecar_remains_readable(closed_case, tmp_path):
    """Existing closed placements keep the historical full-contract format."""
    contract = _contract(closed_case)
    path = tmp_path / "legacy.json"
    legacy = dict(format=policy.FORMAT, policy=policy.POLICY,
                  runs={"target": contract}, adopted=False)
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert policy.read_contracts(path) == legacy
    assert policy.load_contract(path, "target") == contract


def test_39_run_metadata_pool_fits_original_bound_and_expands_exactly(closed_case, tmp_path):
    """Serialization-only size fixture; no simulated authority is admitted.

    Repeating 260 kB model metadata and 270 kB project metadata approximates the
    actual production closure sizes. The pure serializer must retain all bytes
    in one bounded sidecar; the separate original guards still admit each run.
    """
    contract = _contract(closed_case)
    runs = {}
    for index in range(39):
        run = "case" + str(index)
        item = deepcopy(contract)
        item["target_run"] = run
        item["source_model"]["document"]["serialization_size_fixture"] = "x" * 260000
        item["target_model"]["document"]["serialization_size_fixture"] = "x" * 260000
        item["target_model"]["document"]["run_id"] = run
        item["frame_authority"]["project"]["text"] = "p" * 270000
        runs[run] = item
    legacy = dict(format=policy.FORMAT, policy=policy.POLICY, runs=runs, adopted=False)
    assert len(json.dumps(legacy, sort_keys=True, indent=2).encode()) > policy.MAX_BYTES
    path = tmp_path / "size-fixture.json"
    packed = policy._pooled_document(runs)
    raw = (json.dumps(packed, sort_keys=True, indent=2) + "\n").encode()
    assert len(raw) < policy.MAX_BYTES
    assert len(packed["pool"]) == 41  # Source, 39 target documents, one project.
    path.write_bytes(raw)
    assert policy.read_contracts(path) == legacy


@pytest.mark.parametrize("damage", ["pool-bytes", "missing-value", "extra-ref-key", "unconsumed-value"])
def test_pooled_sidecar_rejects_changed_or_ambiguous_metadata(closed_case, tmp_path, damage):
    """A pool cannot alter closed values, redirect unknown refs or add extras."""
    path = tmp_path / "pool.json"
    policy.save_contracts(path, {"target": _contract(closed_case)})
    value = json.loads(path.read_bytes())
    reference = value["runs"]["target"]["source_model"]
    digest = reference["pool_sha256"]
    if damage == "pool-bytes":
        value["pool"][digest]["shape_hw"] = [1, 1]
    elif damage == "missing-value":
        del value["pool"][digest]
    elif damage == "extra-ref-key":
        reference["fallback"] = "unbound"
    else:
        item = {"unused": True}
        value["pool"][policy._json_sha(item)] = item
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="pool"):
        policy.read_contracts(path)


def test_pooled_sidecar_keeps_original_read_bound(tmp_path, monkeypatch):
    """The new representation does not increase or bypass the metadata ceiling."""
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * 65)
    monkeypatch.setattr(policy, "MAX_BYTES", 64)
    with pytest.raises(ValueError, match="metadata size"):
        policy.read_contracts(path)
