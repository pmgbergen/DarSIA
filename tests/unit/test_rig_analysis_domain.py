"""Rig adapter boundary tests with synthetic adopted-source metadata.

Only adoption/scientific replay is supplied at its explicit boundary. RAW request
pins, native pixel/grid/H joins, parsed property tables, image serialization and
the adapter's save/restore checks remain real. No test claims a genuine adoption,
RAW decode, complete native build or externally authenticated certificate.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import darsia
import numpy as np
import pytest

from darsia.fluidflower import rig_analysis_domain as domain
from darsia.fluidflower import rig_raw_build as raw
from darsia.presets.workflows.facies_props import FaciesProps
from darsia.presets.workflows.rig import Rig


def _document(path, value):
    """Write explicit synthetic metadata and return its original byte descriptor."""
    path.write_bytes((json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode())
    return raw._descriptor(path)


class _OwnedTransport:
    """Exclusive unit writer, not a claim about a production owned-build operation."""
    def __init__(self, *, fail=False):
        """Select a deterministic transport failure without deleting partial files."""
        self.fail = fail

    @contextmanager
    def open(self, path):
        """Keep actual CreateNew bytes even if the transport refuses completion."""
        with Path(path).open("xb") as stream:
            if self.fail:
                stream.write(b"partial")
                raise OSError("synthetic bounded transport failure")
            yield stream


@pytest.fixture
def case(tmp_path, monkeypatch):
    """Supply actual images/CSV/request files and a named synthetic replay boundary."""
    root = tmp_path / "inputs"
    root.mkdir()
    export = root / "export"
    export.mkdir()
    raw_path = root / "baseline.arw"
    raw_path.write_bytes(b"synthetic sensor bytes; never decoded")
    profile = _document(root / "profile.json", {"synthetic_profile": True})
    configuration = root / "run.toml"
    configuration.write_text("[synthetic]\nvalue = 1\n", encoding="utf8")
    series = root / "series.toml"
    series.write_text("[synthetic]\nvalue = 2\n", encoding="utf8")
    baseline = darsia.OpticalImage(np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 32,
                                  dimensions=[0.49, 0.9])
    labels = darsia.ScalarImage(np.array([[0, 1, 2], [1, 2, 0]], np.uint16), dimensions=[0.49, 0.9])
    labels_path, facies_path = export / "native_labels.npz", export / "native_facies.npz"
    labels.save(labels_path)
    labels.save(facies_path)
    stored_labels = root / "store_labels.npz"
    stored_labels.write_bytes(labels_path.read_bytes())
    properties = export / "facies_properties.csv"
    properties.write_text("id,porosity,permeability\n0,-0.0,-0.0\n1,0.4,1e-12\n", encoding="utf8")
    camera = dict(configuration=profile["path"], sha256=profile["sha256"], profile_id="synthetic", run_id="sample")
    h = [[1., 0., 2.], [0., 1., 3.], [0., 0., 1.]]
    inverse = [[1., 0., -2.], [0., 1., -3.], [0., 0., 1.]]
    pixels, grid = raw._array_identity(baseline.img), raw._grid(baseline)
    certificate = dict(format=raw.FORMAT, artifact_kind="raw_shape_basis", complete_rig_model=False, run_id="sample",
        inputs=dict(raw=raw._descriptor(raw_path), profile=profile, camera_binding=camera),
        corrected_baseline=dict(pixels=pixels, grid=grid),
        H_rig_corrected_native_to_baseline_camera_u=h,
        renderer=dict(profile_id="synthetic", raw_acquisition=raw._descriptor(raw_path),
            native_shape_hw=[2, 3], source_pixels=pixels, source_grid=grid,
            native_to_camera_u_h=h, camera_u_to_native_h=inverse))
    basis = _document(root / "basis.json", certificate)
    native_grid = dict(renderer=basis, shape_hw=[2, 3],
        H_rig_corrected_native_to_baseline_camera_u=h,
        H_baseline_camera_u_to_rig_corrected_native=inverse,
        legacy_dimensions_m=[0.49, 0.9], physical_metric_is_separate=True)
    source = _document(root / "source.json", {"synthetic_metadata_boundary": True})
    gap = np.array([[True, False, False], [False, False, False]], bool)
    mask_path = root / "approved_mask.npz"
    with mask_path.open("xb") as stream:
        np.savez(stream, approved_analysis_gap=gap)
    gap_contract = dict(format="darsia-approved-analysis-gap-native-v1", run_id="sample",
        meaning="outside-analysis-unquantified", preparation=source, approval=source,
        mask=raw._descriptor(mask_path), field="approved_analysis_gap", identity=raw._array_identity(gap),
        source_labels=raw._descriptor(stored_labels), exported_labels=raw._descriptor(stored_labels),
        source_model=source, native_grid=native_grid, source_to_export_native_h=np.eye(3).tolist(),
        provenance={"physical_frame_unchanged": True, "physical_qualification_claimed": False})
    product = dict(format="darsia-painted-native-product-v1", run_id="sample", adopted_revision=source,
        preparation=source, source_labels=raw._descriptor(stored_labels), exported_labels=raw._descriptor(stored_labels),
        source_model=source, native_grid=native_grid, source_to_export_native_h=np.eye(3).tolist(),
        analysis_exclusions=gap_contract, target_corrections=[])
    rig = Rig()
    rig.baseline_path = raw_path
    rig.baseline = baseline.copy()
    rig.shape_corrected_baseline = baseline.copy()
    rig.labels, rig.facies = labels.copy(), labels.copy()
    rig.depth = darsia.ScalarImage(np.full((2, 3), 0.01), dimensions=[0.49, 0.9])
    rig.raw_camera, rig.raw_baseline_sha256 = deepcopy(camera), raw._descriptor(raw_path)["sha256"]
    rig._raw_build_id = "1" * 32
    value = SimpleNamespace(root=root, export=export, rig=rig, gap=gap, contract=gap_contract,
        product=product, native_grid=native_grid, properties=properties, source=source,
        labels_path=labels_path, facies_path=facies_path, basis=basis, certificate=certificate,
        camera=camera, profile=profile, raw_path=raw_path, configuration=configuration,
        series=series, calls=[])

    def replay(contract, native_product, *, labels, run):
        """Supply only scientific/adoption boundary DATA; never assert real approval."""
        value.calls.append(run)
        assert np.array_equal(labels, rig.labels.img)
        selected = native_product if native_product is not None else contract
        return dict(mask=None if contract is None else value.gap.copy(),
            provenance=None if contract is None else deepcopy(contract["provenance"]),
            contract=deepcopy(selected), native_grid=deepcopy(selected["native_grid"]), dependencies=[source])

    monkeypatch.setattr(domain, "_read_native", replay)
    monkeypatch.setattr(domain, "_export_receipt", lambda paths: deepcopy(value.receipt))
    from darsia.fluidflower import painted_analysis_exclusions as shared
    monkeypatch.setattr(shared, "native_analysis_exclusion_dependencies", lambda contract, reader=None: [source])
    _synchronize(value)
    return value


def _synchronize(case):
    """Bind changed synthetic input DATA before the adapter admits it."""
    rig = case.rig
    receipt = dict(binding={}, files={
        "native_labels.npz": raw._descriptor(case.labels_path),
        "native_facies.npz": raw._descriptor(case.facies_path),
        "facies_properties.csv": raw._descriptor(case.properties)})
    if case.contract is not None:
        receipt["binding"]["analysis_exclusions"] = deepcopy(case.contract)
    if case.product is not None:
        receipt["binding"]["native_product"] = deepcopy(case.product)
    case.receipt = receipt
    _document(case.export / "export.json", receipt)
    request = dict(format=raw.REQUEST_FORMAT, run_id="sample",
        configuration_files=[raw._descriptor(case.configuration)],
        series_context=dict(root=str(case.root), manifest=raw._descriptor(case.series)),
        baseline=raw._descriptor(case.raw_path), profile=case.profile, operation_overrides={}, protocol_inputs=[],
        output=dict(destination=str(case.root / "unused_output")), shape_basis_certificate=case.basis)
    if case.contract is not None:
        request["analysis_exclusions"] = deepcopy(case.contract)
    if case.product is not None:
        request["native_product"] = deepcopy(case.product)
    request_binding = _document(case.root / "request.json", request)
    rig._raw_build = raw.RawRigBuild(operation_id=rig._raw_build_id,
        started_utc="2026-01-01T00:00:00+00:00", destination=case.root / "unused_output",
        camera=deepcopy(case.camera), raw_source=raw._descriptor(case.raw_path), profile=case.profile,
        configurations=tuple(request["configuration_files"]), implementation=(), registration_policies={},
        artifact_kind="raw_rig_baseline", request_binding=request_binding, request=request,
        request_inputs=(), output=SimpleNamespace())
    case.props = FaciesProps.load(rig.facies, case.properties)
    rig.porosity, rig.permeability = case.props.porosity, case.props.permeability


def _prepare(case):
    """Exercise actual adapter admission and property support on the unit case."""
    domain.prepare_analysis_domain(case.rig, case.contract, labels_path=case.labels_path,
                                  facies_path=case.facies_path, native_product=case.product)
    domain.apply_property_support(case.rig, case.props)
    return case.rig._analysis_domain


def _save(case, folder, *, inventory=False, omit=None):
    """Use real original image serializers in the documented helper hook order."""
    folder.mkdir()
    metadata = dict(baseline_path=str(case.rig.baseline_path), raw_camera=case.rig.raw_camera,
        raw_baseline_sha256=case.rig.raw_baseline_sha256, raw_build_id=case.rig._raw_build_id)
    domain.save_analysis_state(case.rig, folder, _OwnedTransport(), metadata)
    _document(folder / "meta_data.json", metadata)
    for name in domain._IMAGES:
        if name != omit:
            getattr(case.rig, name).save(folder / (name + ".npz"))
    if omit is None:
        domain.verify_saved_analysis_state(case.rig, folder, metadata)
    if inventory:
        document = deepcopy(case.certificate)
        document.update(artifact_kind="raw_rig_baseline",
            output_ownership=dict(operation_id=case.rig._raw_build_id),
            outputs=[raw._descriptor(path) for path in sorted(folder.iterdir())])
        document["inputs"]["operation_request"] = case.rig._raw_build.request_binding
        _document(folder / raw.CERTIFICATE_NAME, document)
    return metadata


def _loaded(folder, metadata):
    """Load actual serialized images without pretending to execute original Build."""
    rig = Rig()
    rig.baseline_path = Path(metadata["baseline_path"])
    rig.raw_camera = deepcopy(metadata["raw_camera"])
    rig.raw_baseline_sha256 = metadata["raw_baseline_sha256"]
    for name in domain._IMAGES:
        setattr(rig, name, darsia.imread(folder / (name + ".npz")))
    return rig


def test_absent_declaration_keeps_legacy_calls_and_metadata_untouched(tmp_path):
    """Do not add source reads or optional state to an undeclared Rig."""
    rig, geometry, metadata = SimpleNamespace(), object(), {"legacy": True}
    assert domain.prepare_analysis_domain(rig, None, labels_path=None, facies_path=None) is None
    domain.apply_property_support(rig, None)
    assert domain.wrap_analysis_geometry(rig, geometry) is geometry
    domain.save_analysis_state(rig, tmp_path, None, metadata)
    domain.verify_saved_analysis_state(rig, tmp_path, metadata)
    domain.restore_analysis_state(rig, tmp_path, metadata)
    assert metadata == {"legacy": True} and not vars(rig)


def test_relocated_raw_keeps_admission_save_and_restore_pixel_identity(case, tmp_path, monkeypatch):
    from darsia.fluidflower import storage_paths
    original = case.raw_path
    moved = tmp_path / "relocated-sensor.arw"
    original.rename(moved)
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: [
        (str(original).replace("\\", "/").casefold(), str(moved).replace("\\", "/"))])
    # The real operation has resolved its input; the historical certificate and
    # immutable RAW request still retain the original locator and exact bytes.
    case.rig._raw_build = replace(case.rig._raw_build, raw_source=raw._input_descriptor(original))
    state = _prepare(case)
    assert json.loads(state.proof_json)["raw"]["path"] == str(moved)
    folder = tmp_path / "saved-relocated"
    metadata = _save(case, folder)
    restored = _loaded(folder, metadata)
    domain.restore_analysis_state(restored, folder, metadata)
    np.testing.assert_array_equal(restored.shape_corrected_baseline.img, case.rig.shape_corrected_baseline.img)
    assert not original.exists() and restored.baseline_path == original


@pytest.mark.parametrize("damage,field", [("pixels", "baseline pixels"), ("inverse", "inverse H")])
def test_source_alias_does_not_hide_numeric_mismatch(case, tmp_path, monkeypatch, damage, field):
    from darsia.fluidflower import storage_paths
    original = case.raw_path
    moved = tmp_path / "relocated-sensor.arw"
    original.rename(moved)
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: [
        (str(original).replace("\\", "/").casefold(), str(moved).replace("\\", "/"))])
    case.rig._raw_build = replace(case.rig._raw_build, raw_source=raw._input_descriptor(original))
    if damage == "pixels":
        case.rig.shape_corrected_baseline.img[0, 0, 0] += 0.125
    else:
        # This is the consumer's grid, not a changed source-certificate file.
        case.product["native_grid"]["H_baseline_camera_u_to_rig_corrected_native"][0][2] -= 1
        case.contract["native_grid"]["H_baseline_camera_u_to_rig_corrected_native"][0][2] -= 1
        request = deepcopy(case.rig._raw_build.request)
        request["native_product"] = deepcopy(case.product)
        request["analysis_exclusions"] = deepcopy(case.contract)
        request_binding = _document(case.root / "request.json", request)
        case.rig._raw_build = replace(case.rig._raw_build, request=request, request_binding=request_binding)
        case.receipt["binding"]["native_product"] = deepcopy(case.product)
        case.receipt["binding"]["analysis_exclusions"] = deepcopy(case.contract)
        _document(case.export / "export.json", case.receipt)
    with pytest.raises(ValueError, match=field):
        _prepare(case)


@pytest.mark.parametrize("standalone", [False, True])
def test_native_admission_keeps_known_zero_and_gap_values_bits(case, standalone):
    """Gap approval and an omitted property-table row remain independent causes."""
    if standalone:
        case.product = None
        _synchronize(case)
    original = {name: getattr(case.rig, name).img.copy() for name in ("porosity", "permeability")}
    state = _prepare(case)
    known = case.rig.facies.img != 2
    for name in original:
        assert getattr(case.rig, name).img[known].tobytes() == original[name][known].tobytes()
        assert np.isnan(getattr(case.rig, name).img[~known]).all()
    assert case.rig.porosity.img[0, 0] == 0 and state.approved_gap[0, 0]
    assert np.signbit(case.rig.porosity.img[0, 0])
    assert state.porosity_unknown.sum() == 2
    assert not state.porosity_nonfinite.any()
    for mask in (state.approved_gap, state.porosity_unknown, state.permeability_unknown):
        with pytest.raises(ValueError):
            mask.setflags(write=True)
    assert len(case.calls) == 1


def test_fully_painted_complete_product_has_no_fabricated_gap_or_wrapper(case):
    """An admitted complete native product retains the exact original geometry."""
    case.contract = None
    case.product["analysis_exclusions"] = None
    case.properties.write_text("id,porosity,permeability\n0,0.0,0.0\n1,0.4,1e-12\n2,0.3,2e-12\n", encoding="utf8")
    _synchronize(case)
    state = _prepare(case)
    geometry = darsia.ExtrudedPorousGeometry(case.rig.porosity, case.rig.depth,
        space_dim=2, num_voxels=[2, 3], dimensions=[0.49, 0.9])
    assert not state.approved_gap.any() and not state.porosity_unknown.any()
    assert domain.wrap_analysis_geometry(case.rig, geometry) is geometry
    assert json.loads(state.proof_json)["contract"] is None


def test_assigned_nonfinite_values_are_retained_and_accounted_separately(case):
    """Blank/Inf cells do not become zeros, missing IDs, or a complete total."""
    case.properties.write_text("id,porosity,permeability\n0,0.0,0.0\n1,,inf\n", encoding="utf8")
    _synchronize(case)
    original = {name: getattr(case.rig, name).img.copy() for name in ("porosity", "permeability")}
    state = _prepare(case)
    assigned = case.rig.facies.img != 2
    for name in original:
        assert getattr(case.rig, name).img[assigned].tobytes() == original[name][assigned].tobytes()
    assert state.porosity_unknown.sum() == 2 and state.porosity_nonfinite.sum() == 2
    assert state.permeability_nonfinite.sum() == 2
    geometry = darsia.ExtrudedPorousGeometry(case.rig.porosity, case.rig.depth,
        space_dim=2, num_voxels=[2, 3], dimensions=[0.49, 0.9])
    wrapped = domain.wrap_analysis_geometry(case.rig, geometry)
    with pytest.raises(ValueError, match="incomplete"):
        wrapped.integrate(np.ones((2, 3)))
    result = wrapped.integrate_quantified(np.ones((2, 3)))
    assert result.unquantified_pixels == 5 and result.complete is False
    assert result.full_value is None and result.unknown_mass is None and result.unknown_volume is None


@pytest.mark.parametrize("damage", ["pixels", "origin", "inverse", "raw", "profile", "labels", "facies"])
def test_same_shape_cannot_replace_actual_native_identity(case, damage):
    """Exercise native pixels/grid/H/source joins while keeping array dimensions fixed."""
    if damage == "pixels":
        case.rig.shape_corrected_baseline.img[0, 0, 0] += 0.125
    elif damage == "origin":
        case.rig.shape_corrected_baseline.origin[0] += 0.1
    elif damage == "inverse":
        case.native_grid["H_baseline_camera_u_to_rig_corrected_native"][0][2] -= 1.
        _synchronize(case)
    elif damage == "raw":
        case.rig.raw_baseline_sha256 = "0" * 64
    elif damage == "profile":
        case.rig.raw_camera["profile_id"] = "another-profile"
    else:
        path = case.labels_path if damage == "labels" else case.facies_path
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        _prepare(case)
    assert not hasattr(case.rig, "_analysis_domain")


def test_actual_loaded_labels_cannot_be_resized_or_replaced_after_admission(case):
    """The active False-correction path still needs its original admitted pixels."""
    domain.prepare_analysis_domain(case.rig, case.contract, labels_path=case.labels_path,
                                  facies_path=case.facies_path, native_product=case.product)
    case.rig.facies.img[0, 0] = 1
    with pytest.raises(ValueError, match="labels or facies"):
        domain.apply_property_support(case.rig, case.props)


def test_admitted_native_arrays_inherit_verified_baseline_grid_without_rereading(case, tmp_path, monkeypatch):
    """Empty archive metadata cannot resize maps or replace their admitted IDs."""
    from darsia.fluidflower import setup_rig as dispatch
    values = case.rig.labels.img.copy()
    for path in (case.labels_path, case.facies_path):
        with path.open("wb") as stream:
            np.savez(stream, array=values, metadata={})
    stored = Path(case.product["exported_labels"]["path"])
    stored.write_bytes(case.labels_path.read_bytes())
    for selected in (case.contract, case.product):
        selected["source_labels"] = raw._descriptor(stored)
        selected["exported_labels"] = raw._descriptor(stored)
    for image in (case.rig.baseline, case.rig.shape_corrected_baseline):
        image.origin[:] = [1.25, 2.5]
    grid = raw._grid(case.rig.shape_corrected_baseline)
    case.certificate["corrected_baseline"]["grid"] = grid
    case.certificate["renderer"]["source_grid"] = grid
    case.basis = _document(Path(case.basis["path"]), case.certificate)
    case.native_grid["renderer"] = case.basis
    rig = dispatch.corrected_rig_class(Rig)()
    rig.__dict__.update(vars(case.rig))
    case.rig = rig
    _synchronize(case)
    original = domain._source_array
    admitted = {}

    def load(bound):
        array = original(bound)
        admitted[Path(bound["path"]).name] = array
        return array

    monkeypatch.setattr(domain, "_source_array", load)
    domain.prepare_analysis_domain(rig, case.contract, labels_path=case.labels_path,
                                  facies_path=case.facies_path, native_product=case.product)
    before = {path: path.read_bytes() for path in (case.labels_path, case.facies_path, case.raw_path)}

    def repeated(*args, **kwargs):
        pytest.fail("Admitted native setup repeated an image read, array hash or resize")

    with monkeypatch.context() as guarded:
        guarded.setattr(darsia, "imread", repeated)
        guarded.setattr(darsia, "resize", repeated)
        guarded.setattr(raw, "_array_identity", repeated)
        guarded.setattr(dispatch, "_require_exact_grid", repeated)
        rig.setup_labels(case.labels_path)
        rig.setup_facies(case.facies_path)
    assert not hasattr(rig, "_prepared_analysis_fields")
    for role, path in (("labels", case.labels_path), ("facies", case.facies_path)):
        image = getattr(rig, role)
        assert image.img is admitted[path.name]
        assert image.scalar and not image.series and image.img.dtype == values.dtype
        assert np.array_equal(image.img, values)
        for key in ("dimensions", "origin", "voxel_size"):
            assert np.array_equal(getattr(image, key), getattr(rig.shape_corrected_baseline, key))
    rig.setup_facies_props(case.properties)
    folder = tmp_path / "saved_inherited_grid"
    metadata = _save(case, folder)
    restored = _loaded(folder, metadata)
    domain.restore_analysis_state(restored, folder, metadata)
    for role in ("labels", "facies"):
        assert np.array_equal(getattr(restored, role).img, values)
        assert np.array_equal(getattr(restored, role).origin, rig.shape_corrected_baseline.origin)
        assert np.array_equal(getattr(restored, role).dimensions, rig.shape_corrected_baseline.dimensions)
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("field", ["labels", "facies"])
def test_prepared_native_fields_are_consumed_once_only_for_their_exact_path(case, monkeypatch, field):
    from darsia.fluidflower import setup_rig as dispatch
    domain.prepare_analysis_domain(case.rig, case.contract, labels_path=case.labels_path,
                                  facies_path=case.facies_path, native_product=case.product)
    calls = []

    class ExistingRig:
        def setup_labels(self, **kwargs):
            calls.append(kwargs["path"])

        def setup_facies(self, **kwargs):
            calls.append(kwargs["path"])

    rig = dispatch.corrected_rig_class(ExistingRig)()
    rig.__dict__.update(vars(case.rig))
    monkeypatch.setattr(dispatch, "_require_exact_grid", lambda *args: None)
    setup = getattr(rig, "setup_" + field)
    path = getattr(case, field + "_path")
    other = path.with_name("another.npz")
    setup(other)
    setup(path)
    setup(path)
    assert calls == [other, path]
    assert field not in rig._prepared_analysis_fields.images


def test_prepared_fields_cannot_outlive_their_active_native_state(case):
    domain.prepare_analysis_domain(case.rig, case.contract, labels_path=case.labels_path,
                                  facies_path=case.facies_path, native_product=case.product)
    case.rig._analysis_domain = None
    assert domain.setup_prepared_analysis_field(case.rig, "labels", case.labels_path) is False
    assert not hasattr(case.rig, "_prepared_analysis_fields")


def test_active_native_inputs_require_the_original_full_build(case):
    """A shape-only operation cannot grant physical native-input admission."""
    case.rig._raw_build = replace(case.rig._raw_build, artifact_kind="raw_shape_basis")
    with pytest.raises(ValueError, match="original full RAW build"):
        _prepare(case)
    assert not case.calls


@pytest.mark.parametrize("damage", ["role", "completeness"])
def test_identical_basis_pixels_cannot_replace_the_original_basis_role(case, damage):
    """The slot's original role is checked before native labels are admitted."""
    if damage == "role":
        case.certificate["artifact_kind"] = "raw_rig_baseline"
    else:
        case.certificate["complete_rig_model"] = True
    case.basis = _document(Path(case.basis["path"]), case.certificate)
    case.native_grid["renderer"] = case.basis
    _synchronize(case)
    with pytest.raises(ValueError, match="shape-only RAW basis role"):
        _prepare(case)
    assert not hasattr(case.rig, "_analysis_domain")


@pytest.mark.parametrize("inventory", [False, True])
def test_masks_properties_and_negative_scope_round_trip(case, tmp_path, inventory):
    """Restore actual image bits and guards, reporting available inventory coverage."""
    _prepare(case)
    folder = tmp_path / "saved"
    metadata = _save(case, folder, inventory=inventory)
    assert len(case.calls) == 1
    restored = _loaded(folder, metadata)
    report = domain.restore_analysis_state(restored, folder, metadata)
    assert len(case.calls) == 2
    assert not report["uncovered_outputs"] if inventory else report["uncovered_outputs"]
    for name in ("porosity", "permeability"):
        assert getattr(restored, name).img.tobytes() == getattr(case.rig, name).img.tobytes()
    state = restored._analysis_domain
    assert json.loads(state.provenance_json)["physical_qualification_claimed"] is False
    assert np.array_equal(state.approved_gap, case.gap)
    geometry = darsia.ExtrudedPorousGeometry(restored.porosity, restored.depth,
        space_dim=2, num_voxels=[2, 3], dimensions=[0.49, 0.9])
    with pytest.raises(ValueError, match="incomplete"):
        domain.wrap_analysis_geometry(restored, geometry).integrate(np.ones((2, 3)))


def test_later_legitimate_configuration_change_does_not_add_a_restore_gate(case, tmp_path):
    """The completed cache pins its consumed request without chasing current TOML."""
    _prepare(case)
    folder = tmp_path / "saved"
    metadata = _save(case, folder, inventory=True)
    case.configuration.write_text("[synthetic]\nvalue = 3\n", encoding="utf8")
    case.series.write_text("[synthetic]\nvalue = 4\n", encoding="utf8")
    restored = _loaded(folder, metadata)
    report = domain.restore_analysis_state(restored, folder, metadata)
    assert not report["uncovered_outputs"]
    assert restored._analysis_domain.properties_json == case.rig._analysis_domain.properties_json


def test_partial_original_inventory_reports_missing_coverage_without_a_new_gate(case, tmp_path):
    """An older inventory adds only its actual available output-byte checks."""
    _prepare(case)
    folder = tmp_path / "saved"
    metadata = _save(case, folder, inventory=True)
    certificate_path = folder / raw.CERTIFICATE_NAME
    certificate = json.loads(certificate_path.read_bytes())
    certificate["outputs"] = [row for row in certificate["outputs"]
                              if Path(row["path"]).name != "analysis_domain_masks.npz"]
    _document(certificate_path, certificate)
    report = domain.restore_analysis_state(_loaded(folder, metadata), folder, metadata)
    assert report["uncovered_outputs"] == ["analysis_domain_masks.npz"]
    assert "meta_data.json" in report["checked_outputs"]


def test_existing_inventory_detects_even_semantically_identical_metadata_change(case, tmp_path):
    """Original completed output hashes bind bytes without creating a new certificate."""
    _prepare(case)
    folder = tmp_path / "saved"
    metadata = _save(case, folder, inventory=True)
    with (folder / "meta_data.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="output changed"):
        domain.restore_analysis_state(_loaded(folder, metadata), folder, metadata)


@pytest.mark.parametrize("damage", ["missing_mask", "mask_bytes", "provenance", "table"])
def test_declared_saved_support_never_falls_back_to_legacy(case, tmp_path, damage):
    """Refuse a broken declared state rather than silently rebuild full zero geometry."""
    _prepare(case)
    folder = tmp_path / "saved"
    metadata = _save(case, folder)
    if damage == "missing_mask":
        metadata["analysis_domain"]["masks"]["path"] = str(folder / "absent.npz")
    elif damage == "mask_bytes":
        with (folder / "analysis_domain_masks.npz").open("ab") as stream:
            stream.write(b"changed")
    elif damage == "provenance":
        metadata["analysis_domain"]["provenance"]["physical_qualification_claimed"] = True
    else:
        with case.properties.open("a", encoding="utf8") as stream:
            stream.write("2,0.4,1e-12\n")
    restored = _loaded(folder, metadata)
    with pytest.raises((ValueError, FileNotFoundError)):
        domain.restore_analysis_state(restored, folder, metadata)
    assert not hasattr(restored, "_analysis_domain")


def test_swallowed_physical_write_failure_prevents_save_completion(case, tmp_path):
    """A missing physical image remains visible and previously written files remain."""
    _prepare(case)
    folder = tmp_path / "saved"
    metadata = _save(case, folder, omit="porosity")
    with pytest.raises((ValueError, FileNotFoundError, AssertionError)):
        domain.verify_saved_analysis_state(case.rig, folder, metadata)
    assert (folder / "analysis_domain_masks.npz").exists()
    assert (folder / "permeability.npz").exists()
    assert not (folder / "porosity.npz").exists()


def test_owned_mask_writer_failure_retains_partial_and_adds_no_success_metadata(case, tmp_path):
    """Keep failed exclusive sidecar bytes and propagate the original transport error."""
    _prepare(case)
    folder = tmp_path / "saved"
    folder.mkdir()
    metadata = {}
    with pytest.raises(OSError, match="synthetic bounded"):
        domain.save_analysis_state(case.rig, folder, _OwnedTransport(fail=True), metadata)
    assert (folder / "analysis_domain_masks.npz").read_bytes() == b"partial"
    assert "analysis_domain" not in metadata


def _real_metadata_dependencies(case, monkeypatch):
    """Connect shared metadata collection to the synthetic adoption boundary.

    Every selected metadata document and binary descriptor has real private bytes.
    Only the continuation schema parser and external adoption/scientific admission
    remain unit boundaries; this is not a genuine approved preparation fixture.
    """
    from darsia.fluidflower import painted_analysis_exclusions as shared
    from darsia.fluidflower import painted_native_product as native
    from darsia.fluidflower import photo_edge_continuation as continuation

    def relative(binding):
        """Retain actual bytes with the preparation's original relative locator."""
        return dict(binding, path=str(Path(binding["path"]).relative_to(case.root)))

    passive_config, passive_series = raw._descriptor(case.configuration), raw._descriptor(case.series)
    historical_request = _document(case.root / "historical_request.json", dict(
        configuration_files=[passive_config], series_context=dict(manifest=passive_series)))
    baseline_archive = case.root / "original_native_baseline.npz"
    case.rig.shape_corrected_baseline.save(baseline_archive)
    baseline_binding = raw._descriptor(baseline_archive)
    case.certificate["inputs"].update(configuration_files=[passive_config],
        series_context=dict(manifest=passive_series), operation_request=historical_request)
    case.certificate["corrected_baseline"]["archive"] = baseline_binding
    case.certificate["renderer"]["source_npz"] = baseline_binding
    case.basis = _document(Path(case.basis["path"]), case.certificate)
    case.native_grid["renderer"] = case.basis
    model_document = dict(reference_image=raw._descriptor(case.raw_path),
        baseline_native_grid=case.native_grid, baseline_configuration=passive_config,
        inherited_series_context=dict(manifest=passive_series))
    model = _document(case.root / "selected_model.json", model_document)
    selected_contract = dict(
        source_model=dict(descriptor=model, photograph=raw._descriptor(case.raw_path),
                          document=deepcopy(model_document)),
        target_model=dict(descriptor=model, photograph=raw._descriptor(case.raw_path)),
        frame_authority=dict(configuration_files=[passive_config],
                             series_context=dict(manifest=passive_series)))
    sidecar = _document(case.root / "continuation.json", dict(runs={"sample": selected_contract}))
    painting = dict(files={"map.npz": case.product["source_labels"]})
    review = _document(case.root / "selected_review.json", dict(
        source=dict(painting_facies=painting, geometry_index=case.source),
        source_map=case.product["source_labels"],
        runs=[dict(run_id="sample", product=case.contract["mask"],
                   base_product=case.product["source_labels"], reference_model=model,
                   target_model=model, native_photo=raw._descriptor(case.raw_path), contract=selected_contract)]))
    approval = _document(case.root / "selected_approval.json", dict(review=review))
    preparation = _document(case.root / "selected_preparation.json", dict(
        runs=[dict(run_id="sample", labels=relative(case.product["source_labels"]),
                   reference_model=model, target_model=model)],
        source_map=relative(case.product["source_labels"]), painting_facies=painting,
        geometry_index=case.source,
        continuation=dict(file="continuation.json", sha256=sidecar["sha256"], size_bytes=sidecar["size_bytes"]),
        analysis_exclusions=dict(approval=approval, runs={"sample": dict(artifact=case.contract["mask"],
            field="approved_analysis_gap", identity=case.contract["identity"])})))
    case.contract.update(preparation=preparation, approval=approval, source_model=model)
    case.product.update(preparation=preparation, source_model=model)
    _synchronize(case)

    def continuation_metadata(path):
        """Read checked toy metadata without claiming original continuation admission."""
        assert Path(path).resolve() == Path(sidecar["path"]).resolve()
        return json.loads(Path(path).read_bytes())

    monkeypatch.setattr(continuation, "read_contracts", continuation_metadata)
    monkeypatch.setattr(shared, "native_analysis_exclusion_dependencies", native.native_analysis_exclusion_dependencies)
    original_boundary = domain._read_native
    case.real_collections = []

    def replay_with_actual_dependencies(contract, native_product, *, labels, run):
        """Return real metadata dependencies after the named synthetic science boundary."""
        result = original_boundary(contract, native_product, labels=labels, run=run)
        selected = native_product if native_product is not None else contract
        result["dependencies"] = native.native_analysis_exclusion_dependencies(selected)
        case.real_collections.append(deepcopy(result["dependencies"]))
        return result

    monkeypatch.setattr(domain, "_read_native", replay_with_actual_dependencies)


@pytest.mark.parametrize("damage", [None, "source_labels", "mask", "raw", "profile"])
def test_real_collector_restore_keeps_passive_provenance_and_active_byte_guards(case, tmp_path, monkeypatch, damage):
    """Actual selected dependency collection must survive later passive Apply changes."""
    _real_metadata_dependencies(case, monkeypatch)
    _prepare(case)
    folder = tmp_path / "saved_real_dependencies"
    metadata = _save(case, folder, inventory=True)
    assert len(case.real_collections) == 1
    dependencies = {Path(row["path"]).resolve() for row in case.real_collections[0]}
    passive = {case.configuration.resolve(), case.series.resolve()}
    active = dict(source_labels=Path(case.product["source_labels"]["path"]),
                  mask=Path(case.contract["mask"]["path"]), raw=case.raw_path,
                  profile=Path(case.profile["path"]))
    assert all(path.resolve() in dependencies for path in active.values())
    assert dependencies.isdisjoint(passive)
    persisted = {Path(row["path"]).resolve() for row in metadata["analysis_domain"]["proof"]["dependencies"]}
    assert persisted.isdisjoint(passive)
    case.configuration.write_text("[synthetic]\nvalue = 5\n", encoding="utf8")
    case.series.write_text("[synthetic]\nvalue = 6\n", encoding="utf8")
    if damage is not None:
        with active[damage].open("ab") as stream:
            stream.write(b"changed actual consumed source")
    restored = _loaded(folder, metadata)
    original_open, passive_opens = Path.open, []

    def observed_open(path, *args, **kwargs):
        """Fail on a passive file reopen rather than merely inspect a claimed list."""
        if path.resolve() in passive:
            passive_opens.append(str(path))
            pytest.fail("Restore reopened passive historical TOML or series provenance")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    if damage is None:
        report = domain.restore_analysis_state(restored, folder, metadata)
        assert not report["uncovered_outputs"]
        assert len(case.real_collections) == 2
        assert restored._analysis_domain.properties_json == case.rig._analysis_domain.properties_json
    else:
        with pytest.raises(ValueError):
            domain.restore_analysis_state(restored, folder, metadata)
        assert not hasattr(restored, "_analysis_domain")
    assert not passive_opens
