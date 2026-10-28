"""Synthetic coordinate/static DATA tests, not physical or RAW qualification."""
from copy import deepcopy

import cv2
import numpy as np
import pytest
from pathlib import Path

from tests.unit.test_build_geometry_data import case, shift
from darsia.fluidflower.mass.n1_pipeline import DataReader, _write_json, _write_npz, asset
from darsia.fluidflower.pipeline.build_static_components import (
    _warp_components, prepare_static_components_data,
)
from darsia.fluidflower.pipeline.series import RUN_FORMAT
from darsia.fluidflower.native.raw_rig_source import FRAME_CONTRACT


@pytest.fixture
def static_case(case):
    """Bind real tiny archives and an original typed own-native run to the geometry fixture."""
    case.old_config["source"]["homography_min_scale"] = 1e-14
    case.native["source"]["homography_min_scale"] = 1e-14
    case.old_binding["baseline_reference_to_native_h"] = np.eye(3).tolist()
    case.new_binding["baseline_reference_to_native_h"] = np.eye(3).tolist()
    geometry = case.call()["geometry"]
    transport = DataReader().json(geometry)["coordinate_transport"]["inputs"]
    old_phase = deepcopy(case.phase)
    old_phase["parameters"]["geometry"].update(native_shape_hw=[2, 4],
        source_box_xywh=[2, 1, 4, 2], rig_origin_xy=[-2, -1])
    old_phase["parameters"]["scientific"]["native_shape_hw"] = [2, 4]
    old_phase_binding = _write_json(case.root / "static_old_phase.json", old_phase)
    values = dict(open_components=np.array([[3, 0, 9, 9], [3, 0, 9, 9]], np.uint64),
                  barrier_mask=np.array([[False, True, False, False]] * 2),
                  reference_domain=np.ones((2, 4), bool))
    archive = _write_npz(case.root / "old_static.npz", values)
    native_geometry = _write_json(case.root / "old_native_frame.json",
        dict(contract=FRAME_CONTRACT, run="sample", acquisition=case.old["sources"]["camera_arw"],
             reference_to_native_h=case.old["H_ac28_to_frame_native"],
             baseline_to_native_h=case.old["H_v5_B_to_frame_native"]))
    source = dict(run="sample", source_sha256=case.old["sources"]["camera_arw"]["sha256"],
        selected=True, frame_index=0, elapsed_h=0., previous_source_sha256="d" * 64,
        source_binding=dict(source_sha256=case.old["sources"]["camera_arw"]["sha256"]))
    frame = dict(case_id="frame", frame_index=0, elapsed_h=0., source=source,
        geometry=transport["old_geometry"], native_task=dict(kind="raw_rig",
            geometry=native_geometry, processor=dict(raw_binding=transport["old_raw_binding"])),
        phase_roles=dict(area=dict(role="selected", factor=1.),
            static=dict(components=archive, source_ids=[9, 3, 9], unchanged_source=True)), mass_spec={})
    configuration = dict(format=RUN_FORMAT, run="sample", frames=[frame],
        native=dict(configuration=transport["old_native_configuration"]),
        phase=dict(configuration=old_phase_binding,
            initial_previous=dict(source=dict(source_sha256="d" * 64))),
        mass=dict(template=dict(calibration=None), calibration=None),
        table_fields=["case_id"], history_policy="explicit fixture history", selection_index_field="frame_index")

    class StaticOnlyReader(DataReader):
        """Allow original metadata verification but decode only the exact static archive."""
        def __init__(self):
            """Retain actual archive calls for a positive no-other-array assertion."""
            super().__init__()
            self.archives = []
            self.component_binding = deepcopy(archive)

        def array(self, binding):
            """Reject top/area/RAW array reads during static production."""
            raise AssertionError("Only the static component archive may be decoded")

        def archive(self, binding, names):
            """Use the original archive reader after recording the exact allowed binding."""
            assert binding == self.component_binding
            assert tuple(names) == ("open_components", "barrier_mask", "reference_domain")
            self.archives.append(deepcopy(binding))
            return super().archive(binding, names)

        def path(self, binding):
            """Refuse body hashing of the retained top curve or absent RAW acquisition."""
            assert binding.get("path") != case.curve["path"]
            assert not binding.get("path", "").endswith(".ARW")
            return super().path(binding)

    case.static_geometry = geometry
    case.static_configuration = configuration
    case.static_values = values
    case.static_reader = StaticOnlyReader()
    case.static_output = case.output.parent / "static_output"

    def call():
        """Freeze explicit old-run fixture updates, then invoke only the real new writer."""
        binding = _write_json(case.root / ("old_run_" + str(len(list(case.root.glob('old_run_*.json')))) + ".json"),
                              case.static_configuration)
        return prepare_static_components_data(geometry=case.static_geometry,
            old_configuration=binding, case_id="frame", output=case.static_output, reader=case.static_reader)

    case.static_call = call
    return case


def test_shared_index_warp_preserves_large_ids_masks_and_noncontiguous_input():
    """One nearest index map preserves all original uint64 bits and typed outside zeros."""
    large = np.uint64(2**63 + 17)
    storage = np.array([[large, 8, 4, 8], [0, 8, large, 8]], np.uint64)
    ids = storage[:, ::2]
    assert not ids.flags.c_contiguous
    values = dict(open_components=ids, barrier_mask=ids == 0, reference_domain=np.ones(ids.shape, bool))
    actual, observation = _warp_components(values, shift(1, 1), [4, 5])
    for name, original in values.items():
        expected = np.zeros((4, 5), original.dtype)
        expected[1:3, 1:3] = original
        assert actual[name].dtype == original.dtype
        assert actual[name].tobytes(order="C") == expected.tobytes(order="C")
    assert observation["before"]["use_ipp"] is observation["after"]["use_ipp"] is False


@pytest.mark.parametrize("delta", [0., -.5, .5, -2.])
def test_index_transport_uses_original_nearest_convention_on_non_square_crops(delta):
    """Identity, half-pixel ties and cropping use original OpenCV categorical coordinates."""
    ids = np.array([[1, 2, 3], [4, 5, 6]], np.uint8)
    values = dict(open_components=ids, barrier_mask=ids % 2 == 0,
                  reference_domain=np.ones(ids.shape, bool))
    transform = shift(delta, 0)
    actual, _ = _warp_components(values, transform, [2, 4])
    for name, original in values.items():
        expected = cv2.warpPerspective(original.astype(np.uint8), transform, (4, 2),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(original.dtype)
        assert actual[name].dtype == original.dtype
        assert actual[name].tobytes(order="C") == expected.tobytes(order="C")


def test_original_component_and_geometry_bindings_produce_native_bundle_only(static_case):
    """Genuine source joins retain literal IDs while decoding no RAW/top/area body."""
    result = static_case.static_call()
    assert result["static"]["source_ids"] == [9, 3, 9]
    assert result["static"]["unchanged_source"] is True
    arrays = DataReader().archive(result["static"]["components"], tuple(static_case.static_values))
    expected, _ = _warp_components(static_case.static_values, static_case.transform, [3, 5])
    for name in expected:
        assert arrays[name].dtype == static_case.static_values[name].dtype
        assert arrays[name].tobytes(order="C") == expected[name].tobytes(order="C")
    metadata = DataReader().json(result["static_transport"])
    assert metadata["current_native_shape_hw"] == [3, 5]
    assert metadata["current_native_shape_hw"] != static_case.shape
    assert metadata["phase_categories_used"] is metadata["components_relabelled"] is False
    assert metadata["physical_qualification"] is metadata["class_extension_performed"] is False
    assert len(static_case.static_reader.archives) == 1


@pytest.mark.parametrize("acquisition_size", [False, True])
def test_distinct_native_frame_and_historical_component_descriptor_are_preserved(static_case, acquisition_size):
    """Original two-key components and either acquisition descriptor join separate native/rig rows."""
    frame = static_case.static_configuration["frames"][0]
    native_binding = frame["native_task"]["geometry"]
    assert native_binding != frame["geometry"]
    native = DataReader().json(native_binding)
    if not acquisition_size:
        native["acquisition"] = {key: native["acquisition"][key] for key in ("path", "sha256")}
    frame["native_task"]["geometry"] = _write_json(static_case.root / "historical_native_frame.json", native)
    components = frame["phase_roles"]["static"]["components"]
    historical = {key: components[key] for key in ("path", "sha256")}
    frame["phase_roles"]["static"]["components"] = historical
    static_case.static_reader.component_binding = deepcopy(historical)
    result = static_case.static_call()
    audit = DataReader().json(result["static_transport"])
    assert audit["inputs"]["old_components"] == historical
    assert audit["inputs"]["old_native_frame"] == frame["native_task"]["geometry"]
    retained = DataReader().json(audit["inputs"]["old_native_frame"])
    assert retained == native and ("size_bytes" in retained["acquisition"]) is acquisition_size
    arrays = DataReader().archive(result["static"]["components"], tuple(static_case.static_values))
    expected, _ = _warp_components(static_case.static_values, static_case.transform, [3, 5])
    for name in expected:
        assert arrays[name].dtype == expected[name].dtype
        assert arrays[name].tobytes(order="C") == expected[name].tobytes(order="C")


@pytest.mark.parametrize("change", ["contract", "run", "acquisition", "selected_source", "reference", "baseline", "audit", "rig_row"])
def test_original_native_frame_admission_and_audit_joins_refuse_unrelated_rows(static_case, change):
    """A separate native-frame document still retains original contract/source/frozen-H predicates."""
    frame = static_case.static_configuration["frames"][0]
    native = DataReader().json(frame["native_task"]["geometry"])
    if change == "contract": native["contract"] = "unrelated-native-frame"
    elif change == "run": native["run"] = "other"
    elif change == "acquisition": native["acquisition"]["sha256"] = "e" * 64
    elif change == "selected_source": frame["source"]["source_binding"]["source_sha256"] = "e" * 64
    elif change == "reference": native["reference_to_native_h"] = shift(1, 0).tolist()
    elif change == "baseline": native["baseline_to_native_h"] = shift(1, 0).tolist()
    elif change == "audit":
        native["reference_to_native_h"] = native["baseline_to_native_h"] = shift(1, 0).tolist()
    frame["native_task"]["geometry"] = (_write_json(static_case.root / "unrelated_native_frame.json", native)
        if change != "rig_row" else frame["geometry"])
    with pytest.raises(ValueError):
        static_case.static_call()
    assert not static_case.static_output.exists()


def test_all_barrier_bundle_retains_empty_source_ids_without_qualification(static_case):
    """Geometry-only transport accepts zero open IDs without inventing a phase source."""
    values = dict(open_components=np.zeros((2, 4), np.uint64),
                  barrier_mask=np.ones((2, 4), bool), reference_domain=np.ones((2, 4), bool))
    binding = _write_npz(static_case.root / "all_barrier.npz", values)
    roles = static_case.static_configuration["frames"][0]["phase_roles"]["static"]
    roles.update(components=binding, source_ids=[])
    static_case.static_reader.component_binding = binding
    actual = static_case.static_call()
    assert actual["static"]["source_ids"] == []
    arrays = DataReader().archive(actual["static"]["components"], tuple(values))
    expected, _ = _warp_components(values, static_case.transform, [3, 5])
    for name in values:
        assert arrays[name].dtype == values[name].dtype
        assert arrays[name].tobytes(order="C") == expected[name].tobytes(order="C")
    metadata = DataReader().json(actual["static_transport"])
    assert metadata["physical_qualification"] is metadata["phase_categories_used"] is False
    assert metadata["components_relabelled"] is False


@pytest.mark.parametrize("change", ["reference_basis", "run", "source", "native", "geometry", "raw_binding", "duplicate"])
def test_unrelated_or_reference_basis_run_controls_refuse_before_writes(static_case, change):
    """Current array dimensions cannot replace actual original own-native provenance."""
    config = static_case.static_configuration
    frame = config["frames"][0]
    if change == "reference_basis": frame["phase_roles"]["static"]["unchanged_source"] = False
    elif change == "run": config["run"] = frame["source"]["run"] = "other"
    elif change == "source": frame["source"]["source_sha256"] = "e" * 64
    elif change == "native": config["native"]["configuration"] = frame["geometry"]
    elif change == "geometry": frame["geometry"] = config["native"]["configuration"]
    elif change == "raw_binding": frame["native_task"]["processor"]["raw_binding"] = frame["geometry"]
    elif change == "duplicate": config["frames"].append(deepcopy(frame))
    with pytest.raises(ValueError):
        static_case.static_call()
    assert not static_case.static_output.exists()


@pytest.mark.parametrize("change", ["shape", "negative", "barrier", "domain", "old_phase"])
def test_malformed_old_native_arrays_or_phase_grid_refuse(static_case, change):
    """All original component invariants and the old phase-native grid remain required."""
    values = deepcopy(static_case.static_values)
    if change == "shape": values = {key: value[:, :2] for key, value in values.items()}
    elif change == "negative":
        values["open_components"] = values["open_components"].astype(np.int64)
        values["open_components"][0, 0] = -1
    elif change == "barrier": values["barrier_mask"][0, 0] = True
    elif change == "domain": values["reference_domain"][0, 0] = False
    if change == "old_phase":
        phase = DataReader().json(static_case.static_configuration["phase"]["configuration"])
        phase["parameters"]["geometry"]["native_shape_hw"] = [3, 5]
        static_case.static_configuration["phase"]["configuration"] = _write_json(static_case.root / "wrong_old_phase.json", phase)
    else:
        binding = _write_npz(static_case.root / "malformed_components.npz", values)
        static_case.static_configuration["frames"][0]["phase_roles"]["static"]["components"] = binding
        # Keep the genuine archive reader; only its explicitly allowed input changes.
        static_case.static_reader = DataReader()
    with pytest.raises(ValueError):
        static_case.static_call()
    assert not static_case.static_output.exists()


def test_forged_saved_transform_is_rejected_by_original_transport_reader(static_case):
    """A pinned geometry JSON cannot self-authorize an altered matrix."""
    geometry = DataReader().json(static_case.static_geometry)
    geometry["coordinate_transport"]["old_native_to_new_native_h"] = shift(8, 1).tolist()
    static_case.static_geometry = _write_json(static_case.root / "forged_geometry.json", geometry)
    with pytest.raises(ValueError, match="coordinate transport"):
        static_case.static_call()
    assert not static_case.static_output.exists()


def test_existing_output_is_retained_without_overwrite(static_case):
    """The original publishers receive only a new exclusively created output folder."""
    static_case.static_output.mkdir()
    marker = static_case.static_output / "keep"
    marker.write_bytes(b"retained")
    with pytest.raises(ValueError, match="fresh positively local"):
        static_case.static_call()
    assert marker.read_bytes() == b"retained"


def test_output_cannot_overlap_the_actual_aliased_component_folder(static_case):
    """Original DataReader aliases cannot hide a consumed physical source parent."""
    logical = static_case.static_configuration["frames"][0]["phase_roles"]["static"]["components"]
    original = Path(logical["path"]).read_bytes()
    folder = static_case.output.parent / "aliased_input"
    folder.mkdir()
    physical = folder / "components.npz"
    physical.write_bytes(original)
    static_case.static_reader = DataReader(aliases=[dict(logical=logical, physical=asset(physical))])
    static_case.static_output = folder / "new_static"
    with pytest.raises(ValueError, match="overlaps a consumed source"):
        static_case.static_call()
    assert not static_case.static_output.exists()
    assert physical.read_bytes() == original
    assert Path(logical["path"]).read_bytes() == original


def test_changed_original_archive_refuses_before_output(static_case, monkeypatch):
    """An actual input mutation after decoding is caught by the original byte recheck."""
    original = static_case.static_reader.archive

    def changed(binding, names):
        """Decode through the original reader, then change only the synthetic source file."""
        values = original(binding, names)
        with Path(binding["path"]).open("ab") as stream:
            stream.write(b"changed fixture input")
        return values

    monkeypatch.setattr(static_case.static_reader, "archive", changed)
    with pytest.raises(ValueError):
        static_case.static_call()
    assert not static_case.static_output.exists()


def test_source_change_after_native_write_retains_partial_without_sidecar(static_case, monkeypatch):
    """A failed post-write source recheck retains original publisher output and no success metadata."""
    from darsia.fluidflower.mass import n1_pipeline
    original = n1_pipeline._write_npz

    def changed(path, arrays):
        """Write the actual native archive before changing a synthetic consumed source."""
        binding = original(path, arrays)
        old_phase = static_case.static_configuration["phase"]["configuration"]
        with Path(old_phase["path"]).open("ab") as stream:
            stream.write(b"changed fixture metadata")
        return binding

    monkeypatch.setattr(n1_pipeline, "_write_npz", changed)
    with pytest.raises(ValueError):
        static_case.static_call()
    assert (static_case.static_output / "static_components.npz").is_file()
    assert not (static_case.static_output / "static_components.json").exists()
