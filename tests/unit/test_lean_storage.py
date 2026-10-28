"""Explicit full/lean storage profiles for native N1 frame products.

Synthetic frames only. The full profile must stay byte-identical to the base
commit's writer; the lean profile may change storage only. Every array a lean
reader returns must equal the array the full profile stores, bit for bit.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import types
import uuid
import zipfile

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import storage_profile as lean
from darsia.fluidflower.mass import n1_pipeline as n1
from darsia.fluidflower.pipeline import checkpoint
from darsia.fluidflower.pipeline.job_queue import write_immutable

REPOSITORY = Path(__file__).resolve().parents[2]
BASE_COMMIT = "3ee2e7b3866dab6baba748010d16fc68c9ce362a"  # Lock-integration head before this change.
CODE = "6dfca622f5ed8ebe5c3ea6474025dcf5bf28023d"
LEGACY_FIELDS = dict(source_to_native="H_ac28_to_frame_native",
                     original_domain_to_native="H_ac28_to_frame_native",
                     carried_homography_provenance="original_H_ac28_to_native")


def _module(relative, name):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny(tmp_path):
    """The unchanged mandatory tiny saved-model fixture, with explicit field names."""
    module = _module("tests/test_n1_pipeline.py", "_lean_tiny_fixture")
    config, _ = module.tiny_config.__wrapped__(tmp_path)
    config["geometry_fields"] = dict(LEGACY_FIELDS)
    return config


def _bind(path, config):
    path.write_text(json.dumps(config), encoding="utf-8")
    return n1.asset(path)


def _save(root, name, value, kind):
    path = root / name
    if kind == "json":
        path.write_text(json.dumps(value), encoding="utf-8")
    elif kind == "npy":
        np.save(path, value, allow_pickle=False)
    elif kind == "npz":
        np.savez(path, **value)
    else:
        Image.fromarray(value).save(path)
    return n1.asset(path)


def _synthetic(tmp_path):
    """A larger frame pair with a perspective registration and a measured correction.

    The canvas maps inside the physical reference, part of the canvas lies
    outside the measured-depth hull, a facies hole needs the fill rule, the
    floor cuts the bottom rows, and the two frames have different registrations.
    """
    root = tmp_path / "synthetic"
    root.mkdir()
    config = _tiny(root)
    rng = np.random.default_rng(20261019)
    reference_hw, canvas_hw, centre_hw, origin = (40, 56), (32, 48), (30, 45), [-2, -1]
    facies = np.ones(reference_hw, np.uint8)
    facies[:, 30:] = 2
    facies[10:13, 20:26] = 0
    physical = dict(facies=facies, domain=np.ones(reference_hw, bool),
                    porosity=0.30 + 0.10 * rng.random(reference_hw),
                    depth_m=0.009 + 0.002 * rng.random(reference_hw))
    shared = config["shared"]
    shared["physical_reference"] = _save(root, "physical.npz", physical, "npz")
    shared["reference_matrix"] = [[1 / 56, 0., 0.], [0., -1 / 40, 1.], [0., 0., 1.]]
    shared["floor_curve"] = _save(root, "floor.npy", 30. + 0.02 * np.arange(56), "npy")
    shared["floor_origin_xy"] = [0, 0]
    shared["depth_points"] = [[.15, .25, .010], [.75, .25, .011], [.75, .90, .009],
                              [.15, .90, .0105], [.45, .60, .0102]]
    shared["porosity_by_facies"] = {"1": .35, "2": .30}
    shared["swi_by_facies"] = {"1": .2, "2": .25}
    shared["clearwater"]["rgb"] = _save(root, "clear.npy", np.full((*reference_hw, 3), .9, np.float32), "npy")
    shared["clearwater"]["valid"] = _save(root, "clear_valid.npy", np.ones(reference_hw, bool), "npy")
    config["physical_correction"] = _save(root, "correction.json", dict(
        correction_matrix=[[1.01, 0., -.3], [0., 1.01, .2], [0., 0., 1.]]), "json")
    rig = np.zeros(canvas_hw, bool)
    rig[1:31, 2:46] = True
    labels = np.full(canvas_hw, 255, np.uint8)
    labels[rig] = 0
    labels[5:16, 10:31] = 1
    labels[6:10, 15:21] = 2
    phase = _save(root, "labels.png", labels, "png")
    classes = _save(root, "classes.npz", dict(rig_domain=rig, modelled=rig, filled_observed=rig,
                                               filled_unobserved=rig & (np.arange(32)[:, None] >= 29)), "npz")
    columns = np.arange(canvas_hw[1])[None, :]
    area = (0.027 * (1 + 0.01 * columns / 48) * np.ones(canvas_hw)).astype(np.float32)
    top = np.ones(canvas_hw, bool)
    top[0] = False
    valid = _save(root, "centre_valid.npy", np.ones(centre_hw, bool), "npy")
    frames = []
    for number, (index, shift) in enumerate(((1, 0.), (3, .4))):
        inverse = np.array([[1.03, .01, 3.2 + shift], [-.008, 1.02, 2.7 - shift / 2], [1e-4, -5e-5, 1.]])
        geometry = {LEGACY_FIELDS["source_to_native"]: np.linalg.inv(inverse).tolist()}

        def raw(name, rgb):
            logs = np.moveaxis(np.log(rgb * np.float32(65535)), 2, 0).astype(np.float32)
            return dict(phase_shape_hw=list(canvas_hw), native_origin_xy=origin,
                        native_centre=dict(native_shape_hw=list(centre_hw),
                                           log_linear_rgb=_save(root, name, logs, "npy"), valid_mask=valid),
                        ring_windows=[], binding=dict(geometry=geometry))
        current = raw("current_%d.npy" % index, (.8 - .2 * rng.random((*centre_hw, 3))).astype(np.float32))
        baseline = raw("baseline_%d.npy" % index, np.full((*centre_hw, 3), .95, np.float32))
        original = _save(root, "parent_%d.png" % index, np.zeros(canvas_hw, np.uint8), "png")
        frames.append(dict(
            case_id="sample_%d" % index, frame_index=index, elapsed_h=float(index),
            selected_target_h=float(index), source_binding=dict(source_sha256=str(number + 1) * 64),
            current=current, baseline=baseline, labels=phase, classes=classes,
            area=_save(root, "area_%d.npz" % index, dict(pixel_area_mm2=area * (1 + number / 100)), "npz"),
            photo_top=_save(root, "top.npz", dict(keep=top), "npz"),
            original_domain=dict(parent_labels_map=original, native_shape_hw=list(canvas_hw),
                                 H_ac28_to_frame_native=geometry[LEGACY_FIELDS["source_to_native"]],
                                 frame_index=0),
            inner_polygon_native=[[0, 0], [44, 0], [44, 29], [0, 29]],
            atmosphere=dict(pressure_bar=1.01, temperature_c=21.5, pressure_gradient_bar_per_m=0.,
                            temperature_gradient_c_per_m=0.),
            nominal_injected_mass_g=1., light_epoch="measured"))
    config["frames"] = frames
    return config


@pytest.fixture
def synthetic(tmp_path):
    return _synthetic(tmp_path)


def _run(tmp_path, name, config, profile=None, *, module=n1, **task):
    config = deepcopy(config)
    if profile is not None:
        config["storage_profile"] = profile
        if isinstance(profile, dict) and isinstance(profile.get("profile"), str):
            # N55: the job's choice, carried by its task, selects the profile.
            task.setdefault("storage_profile", profile["profile"])
    bound = _bind(tmp_path / (name + "_config.json"), config)
    return module.run_n1(dict(run=config["run"], configuration=bound, commit=CODE, **task),
                         tmp_path / name, "cpu", "float32")


def _frames(outcome):
    for record in outcome["records"]:
        root = Path(record["nas_directory"])
        yield root, json.loads((root / "result.json").read_bytes())


def _archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _same_bits(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return (first.dtype == second.dtype and first.shape == second.shape
            and np.ascontiguousarray(first).tobytes() == np.ascontiguousarray(second).tobytes())


# ------------------------------------------------------------------- setting


def test_default_constant_is_full_and_setting_is_strict():
    assert lean.DEFAULT_STORAGE_PROFILE == "full"
    assert lean.resolve({}) == dict(profile="full", colour_intermediates="keep")
    assert lean.resolve({"storage_profile": {"profile": "lean"}}) == dict(
        profile="lean", colour_intermediates="keep")
    assert lean.resolve({"storage_profile": {"profile": "lean", "colour_intermediates": "drop"}}) == dict(
        profile="lean", colour_intermediates="drop")
    for bad in ({"profile": "small"}, {"colour_intermediates": "keep"}, {"profile": "lean", "extra": 1},
                {"profile": "full", "colour_intermediates": "drop"},
                {"profile": "lean", "colour_intermediates": "delete"}, "lean"):
        with pytest.raises(ValueError):
            lean.resolve({"storage_profile": bad})


def test_default_constant_selects_new_jobs_only(synthetic, tmp_path, monkeypatch):
    # N55: the constant is the default of NEW jobs (prepare_job, Jobs panel).
    # A task without the explicit field is a legacy task and stays full.
    monkeypatch.setattr(lean, "DEFAULT_STORAGE_PROFILE", "lean")
    assert lean.new_job_profile() == "lean"
    legacy = _run(tmp_path, "defaulted", synthetic)
    assert "lean_storage" not in legacy
    assert all("native_geometry.npz" in record["files"] for record in legacy["records"])
    chosen = _run(tmp_path, "chosen", synthetic, storage_profile="lean")
    assert chosen["lean_storage"]["profile"] == "lean"
    assert all("native_geometry.npz" not in record["files"] for record in chosen["records"])


# ------------------------------------------------- full profile = base writer


class _Clock:
    """Deterministic wall/perf clocks and UUIDs, reset per run."""

    def __init__(self, monkeypatch, modules):
        self.monkeypatch, self.modules = monkeypatch, modules

    def reset(self):
        counter, ids = iter(range(1, 10 ** 6)), iter(range(1, 10 ** 6))
        fixed = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)

        class FixedDatetime:
            @staticmethod
            def now(tz=None):
                return fixed

        class FixedTime:
            def __getattr__(self, name):
                return getattr(time, name)

            @staticmethod
            def time():
                return 1792000000.0

        self.monkeypatch.setattr(time, "perf_counter", lambda: float(next(counter)))
        self.monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(ids)))
        self.monkeypatch.setattr(zipfile, "time", FixedTime())
        self.monkeypatch.setattr(gzip, "time", FixedTime())
        for module in self.modules:
            self.monkeypatch.setattr(module, "datetime", FixedDatetime)


def _base_writer():
    """The base commit's n1_pipeline source, in this package's import context."""
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "show", BASE_COMMIT + ":src/darsia/fluidflower/mass/n1_pipeline.py"],
            stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("The base commit is not available in this checkout")
    name = "darsia.fluidflower.mass._lean_base_n1_pipeline"
    module = types.ModuleType(name)
    module.__package__ = "darsia.fluidflower.mass"
    module.__file__ = n1.__file__
    sys.modules[name] = module
    exec(compile(raw, "base:n1_pipeline.py", "exec"), module.__dict__)
    return module


def _tree(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(Path(root).rglob("*")) if path.is_file()}


def _assert_full_matches_base(tmp_path, monkeypatch, config):
    """Same input, same clocks: every output byte of the default must match base."""
    base = _base_writer()
    clock = _Clock(monkeypatch, (n1, base))
    clock.reset()
    expected = _run(tmp_path, "out", config, module=base, job_id="job")
    (tmp_path / "out").rename(tmp_path / "base_out")
    clock.reset()
    actual = _run(tmp_path, "out", config, job_id="job")
    old, new = _tree(tmp_path / "base_out"), _tree(tmp_path / "out")
    assert sorted(old) == sorted(new)
    assert [name for name in old if old[name] != new[name]] == []
    assert json.dumps(expected, sort_keys=True, default=str) == json.dumps(actual, sort_keys=True, default=str)
    return new


def test_full_profile_is_byte_identical_to_base_writer(synthetic, tmp_path, monkeypatch):
    files = _assert_full_matches_base(tmp_path, monkeypatch, synthetic)
    assert any(name.endswith("native_geometry.npz") for name in files)
    assert not any("lean_run_static" in name for name in files)


def test_full_profile_pixel_domain_is_byte_identical_to_base_writer(tmp_path, monkeypatch):
    fixture = _module("tests/fluidflower/test_n1_pixel_domain_integration.py", "_lean_domain_fixture")
    request = types.SimpleNamespace(param={})
    config, _ = fixture.frozen_run.__wrapped__(tmp_path, request)
    _assert_full_matches_base(tmp_path, monkeypatch, config)


# ------------------------------------------------------ lean profile writes


def _full_and_lean(tmp_path, config, colour="keep"):
    full = _run(tmp_path, "full", config)
    lean_run = _run(tmp_path, "lean", config, {"profile": "lean", "colour_intermediates": colour})
    return full, lean_run


def test_lean_drops_exactly_the_contracted_writes(synthetic, tmp_path):
    full, lean_run = _full_and_lean(tmp_path, synthetic)
    store = lean_run["lean_storage"]["store"]
    assert Path(store["index"]["path"]) == tmp_path / "lean" / lean.STORE_INDEX
    assert n1.asset(store["archive"]["path"]) == store["archive"]
    for (old_root, old), (new_root, new) in zip(_frames(full), _frames(lean_run)):
        assert set(p.name for p in old_root.iterdir()) - set(p.name for p in new_root.iterdir()) == {
            "native_geometry.npz"}
        old_products, new_products = _archive(old_root / "products.npz"), _archive(new_root / "products.npz")
        assert set(old_products) - set(new_products) == {"mass_total"}
        assert list(new_products) == [name for name in old_products if name != "mass_total"]
        for name, value in new_products.items():
            assert _same_bits(value, old_products[name]), name
        record = new["lean_storage"]
        assert record["static"]["written"] == []
        assert sorted(record["static"]["regenerated"]) == sorted(old["static"]["arrays"])
        assert record["products"]["omitted"]["mass_total"]["identity"] == n1._array_identity(old_products["mass_total"])
        assert new["static"]["asset"] is None and new["static"]["arrays"] == old["static"]["arrays"]
        # Only storage records differ; every scientific scalar and binding is unchanged.
        drop = {"products", "static", "lean_storage", "compute_seconds", "opencv_ipp"}
        assert {k: v for k, v in old.items() if k not in drop} == {k: v for k, v in new.items() if k not in drop}
        with gzip.open(old_root / "native_statistics.json.gz") as a, gzip.open(new_root / "native_statistics.json.gz") as b:
            assert a.read() == b.read()
    manifest = json.loads(Path(lean_run["manifest"]["path"]).read_bytes())
    assert manifest["storage_profile"]["profile"] == "lean"


def test_lean_readers_return_full_arrays_bit_for_bit(synthetic, tmp_path):
    full, lean_run = _full_and_lean(tmp_path, synthetic)
    descriptors = []
    for (old_root, old), (new_root, new) in zip(_frames(full), _frames(lean_run)):
        geometry = _archive(old_root / "native_geometry.npz")
        regenerated = lean.read_static_fields(new, new_root)
        assert set(regenerated) == set(geometry)
        for name, value in geometry.items():
            assert _same_bits(regenerated[name], value), name
        products = _archive(old_root / "products.npz")
        rebuilt = lean.read_products(new, new_root)
        assert set(rebuilt) == set(products)
        for name, value in products.items():
            assert _same_bits(rebuilt[name], value), name
        # Full frames read through the same API.
        for name, value in lean.read_static_fields(old, old_root).items():
            assert _same_bits(value, geometry[name])
        assert _same_bits(lean.read_products(old, old_root, ["mass_total"])["mass_total"], products["mass_total"])
        descriptors.append(new["static"]["storage"]["regeneration"]["source_to_native_h"])
    assert descriptors[0] != descriptors[1]  # Per-frame registration, one shared store.


def test_lean_store_is_written_once_and_exclusively(synthetic, tmp_path):
    _, lean_run = _full_and_lean(tmp_path, synthetic)
    store = list((tmp_path / "lean").glob("lean_run_static*"))
    assert sorted(p.name for p in store) == [lean.STORE_INDEX, lean.STORE_ARCHIVE]
    before = {p.name: p.read_bytes() for p in store}
    from darsia.fluidflower.mass.production_precision import PixelMath
    config = dict(synthetic)
    pipeline = n1.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"))
    with pytest.raises(FileExistsError):
        lean.LeanRun(tmp_path / "lean", config, pipeline, dict(profile="lean", colour_intermediates="keep"),
                     colour_intermediates_produced=False)
    assert {p.name: p.read_bytes() for p in store} == before
    stored = lean.load_store(lean_run["lean_storage"]["store"]["index"])
    assert stored.document["tables"]["porosity_by_facies"] == [["1", .35], ["2", .30]]
    assert _same_bits(stored.physical["porosity"], np.load(synthetic["shared"]["physical_reference"]["path"])["porosity"])


def test_tampered_store_or_descriptor_is_detected(synthetic, tmp_path):
    _, lean_run = _full_and_lean(tmp_path, synthetic)
    root, result = next(_frames(lean_run))
    store = lean_run["lean_storage"]["store"]
    original = Path(store["archive"]["path"]).read_bytes()
    original_index = Path(store["index"]["path"]).read_bytes()
    # A changed per-run input with an honest new SHA is caught by the field SHA.
    arrays = _archive(store["archive"]["path"])
    arrays["depth_points"][0, 2] *= 1.5
    Path(store["archive"]["path"]).unlink()
    archive = n1._write_npz(Path(store["archive"]["path"]), arrays)
    index = json.loads(Path(store["index"]["path"]).read_bytes())
    index["archive"].update(sha256=archive["sha256"], size_bytes=archive["size_bytes"])
    index["arrays"]["depth_points"] = n1._array_identity(arrays["depth_points"])
    Path(store["index"]["path"]).unlink()
    forged = n1._write_json(Path(store["index"]["path"]), index)
    changed = deepcopy(result)
    changed["static"]["storage"]["regeneration"]["store"] = forged
    with pytest.raises(ValueError, match="recorded SHA-256: depth_m"):
        lean.read_static_fields(changed, root, ["depth_m"])
    # The original descriptor no longer matches the changed store bytes.
    with pytest.raises(ValueError, match="SHA or identity"):
        lean.read_static_fields(result, root, ["depth_m"])
    for name, raw in (("archive", original), ("index", original_index)):
        Path(store[name]["path"]).unlink()
        Path(store[name]["path"]).write_bytes(raw)
    lean.read_static_fields(result, root)  # The restored store regenerates again.
    # A changed frame registration descriptor is caught by the field SHA.
    changed = deepcopy(result)
    changed["static"]["storage"]["regeneration"]["source_to_native_h"][0][2] += 0.5
    with pytest.raises(ValueError, match="recorded SHA-256"):
        lean.read_static_fields(changed, root, ["facies"])
    # A changed retained per-frame input is caught by its descriptor SHA.
    changed = deepcopy(result)
    changed["static"]["storage"]["regeneration"]["inputs"]["area"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA or identity"):
        lean.read_static_fields(changed, root, ["pixel_area_m2"])
    # A wrong rebuilt-mass identity is caught as well.
    changed = deepcopy(result)
    changed["lean_storage"]["products"]["omitted"]["mass_total"]["identity"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mass_total"):
        lean.read_products(changed, root, ["mass_total"])


# ------------------------------------------------------------- fallbacks


def test_mass_total_is_kept_when_the_sum_is_not_bit_identical():
    gas = np.array([0., 1., np.nan, 2.], np.float32)
    aqueous = np.array([0., 2., np.nan, 3.], np.float32)
    frame = lean.LeanFrame(types.SimpleNamespace(), {}, {})
    total = gas + aqueous
    written = frame.products(dict(mass_total=total, mass_aqueous=aqueous, mass_free_gas=gas))
    assert list(written) == ["mass_aqueous", "mass_free_gas"]
    nudged = total.copy()
    nudged[3] = np.nextafter(nudged[3], np.float32(10))
    written = frame.products(dict(mass_total=nudged, mass_aqueous=aqueous, mass_free_gas=gas))
    assert list(written) == ["mass_total", "mass_aqueous", "mass_free_gas"]
    assert frame.products_record["kept_reasons"]["mass_total"] == "rebuilt sum is not bit-identical"
    written = frame.products(dict(mass_total=total.astype(np.float64), mass_aqueous=aqueous, mass_free_gas=gas))
    assert "mass_total" in written


def test_a_field_that_does_not_regenerate_keeps_being_written(synthetic, tmp_path, monkeypatch):
    full = _run(tmp_path, "full", synthetic)
    original = lean._Regeneration.field

    def drifting(self, name):
        value = original(self, name)
        if name == "depth_m":
            value = np.nextafter(np.asarray(value), np.inf)
        return value

    monkeypatch.setattr(lean._Regeneration, "field", drifting)
    lean_run = _run(tmp_path, "lean", synthetic, {"profile": "lean"})
    monkeypatch.undo()
    for (old_root, old), (new_root, new) in zip(_frames(full), _frames(lean_run)):
        storage = new["static"]["storage"]
        assert storage["written"] == ["depth_m"]
        assert storage["kept_reasons"] == {"depth_m": "regenerated field is not bit-identical"}
        assert set(storage["regenerated"]) == set(old["static"]["arrays"]) - {"depth_m"}
        assert set(_archive(new_root / "native_geometry.npz")) == {"depth_m"}
        geometry = _archive(old_root / "native_geometry.npz")
        for name, value in lean.read_static_fields(new, new_root).items():
            assert _same_bits(value, geometry[name]), name


def test_without_a_regeneration_descriptor_every_field_is_written(synthetic, tmp_path, monkeypatch):
    full = _run(tmp_path, "full", synthetic)

    def unavailable(self):
        raise KeyError("no registration descriptor for this route")

    monkeypatch.setattr(lean.LeanFrame, "regeneration", unavailable)
    lean_run = _run(tmp_path, "lean", synthetic, {"profile": "lean"})
    for (old_root, old), (new_root, new) in zip(_frames(full), _frames(lean_run)):
        storage = new["static"]["storage"]
        assert storage["regenerated"] == [] and storage["regeneration"] is None
        assert sorted(storage["written"]) == sorted(old["static"]["arrays"])
        assert _archive(new_root / "native_geometry.npz").keys() == _archive(old_root / "native_geometry.npz").keys()
        for name, value in lean.read_static_fields(new, new_root).items():
            assert _same_bits(value, _archive(old_root / "native_geometry.npz")[name])


def test_a_field_without_a_rule_is_not_regenerable(synthetic, tmp_path):
    _, lean_run = _full_and_lean(tmp_path, synthetic)
    _, new = next(_frames(lean_run))
    engine = lean._Regeneration(lean.load_store(lean_run["lean_storage"]["store"]["index"]),
                                new["static"]["storage"]["regeneration"], n1.DataReader())
    with pytest.raises(KeyError, match="No lean regeneration rule"):
        engine.field("n16_solubility_g_per_l")


# ------------------------------------------------------------ N15/N16 route


def test_n16_route_regenerates_all_static_fields_and_keeps_storage(tmp_path):
    saved = _module("tests/unit/test_storage_n1_saved_products.py", "_lean_saved_products")
    fixture = _module("tests/fixtures/n1_phase2_fixture.py", "_lean_phase2_fixture")
    tiny = fixture.tiny_config.__wrapped__(tmp_path)
    config = saved._genuine_n16_config(tmp_path, tiny, fixture.phase1_data.__wrapped__(), adjacent=True)
    full = _run(tmp_path, "full", config)
    lean_run = _run(tmp_path, "lean", config, {"profile": "lean"})
    for (old_root, old), (new_root, new) in zip(_frames(full), _frames(lean_run)):
        geometry = _archive(old_root / "native_geometry.npz")
        assert len(geometry) == 11
        assert new["static"]["storage"]["written"] == [] and new["static"]["asset"] is None
        regenerated = lean.read_static_fields(new, new_root)
        for name, value in geometry.items():
            assert _same_bits(regenerated[name], value), name
        products = _archive(old_root / "products.npz")
        for name, value in lean.read_products(new, new_root).items():
            assert _same_bits(value, products[name]), name
        # The in-run coarse layer is unchanged by the lean profile.
        for name in [n for n in old_root.iterdir() if n.name.startswith("storage_") and n.suffix == ".npz"]:
            assert _archive(name).keys() == _archive(new_root / name.name).keys()
            for key, value in _archive(name).items():
                assert _same_bits(value, _archive(new_root / name.name)[key]), key
        assert new["n16_extensive_producer"]["native_geometry"] == new["static"]
        assert new["n16_extensive_producer"]["native_products"] == new["products"]
        assert "mass_total" in new["lean_storage"]["products"]["omitted"]
    # Phase-two DATA admits the explicit setting; it has no colour intermediates.
    from darsia.fluidflower.pipeline import staged_data as staged
    staged.validate_phase2_run(dict(config, storage_profile={"profile": "lean"}))
    with pytest.raises(ValueError, match="no colour intermediates"):
        staged.validate_phase2_run(dict(config, storage_profile={"profile": "lean", "colour_intermediates": "drop"}))
    # A lean frame cannot serve as the N16 measured-depth reference.
    spec = config["frames"][0]
    read = lambda bound: json.loads(Path(bound["path"]).read_bytes())
    _, lean_result = next(_frames(lean_run))
    with pytest.raises(ValueError, match="full-profile reference"):
        staged.validate_n16_frame_binding(
            config, spec, authority=read(config["pixel_domain"]["n16"]["authority"]),
            source=read(spec["pixel_domain_geometry"]["source_receipt"]), original=lean_result,
            calibration=read(config["calibration"]))


def test_frozen_basis_must_be_a_full_profile_reference(tmp_path):
    fixture = _module("tests/fluidflower/test_n1_pixel_domain_integration.py", "_lean_domain_fixture2")
    config, _ = fixture.frozen_run.__wrapped__(tmp_path, types.SimpleNamespace(param={}))
    lean_run = _run(tmp_path, "lean_domain", config, {"profile": "lean"})
    root, result = next(_frames(lean_run))
    assert result["static"]["storage"]["written"] == []
    reference = deepcopy(config)
    original = n1.asset(root / "result.json")
    reference["frames"][0]["frozen_response_basis"]["result"] = original
    from darsia.fluidflower.mass import pixel_pipeline
    from darsia.fluidflower.mass.production_precision import PixelMath
    native = n1.N1Pipeline(reference, precision="float32", pixel_math=PixelMath("float32"))
    with pytest.raises(ValueError, match="full-profile reference"):
        pixel_pipeline._reference_basis(native, reference["frames"][0], (3, 3))


# --------------------------------------------------------- resume and index


class _Interrupted(BaseException):
    pass


def _retired_prefix(tmp_path, config, profile):
    config = deepcopy(config)
    config["storage_profile"] = profile
    bound = _bind(tmp_path / "resume_config.json", config)
    task = dict(run=config["run"], configuration=bound, commit=CODE,
                calibration_sha256=config["calibration"]["sha256"], threads=1,
                storage_profile=profile["profile"])
    old = tmp_path / "attempts" / ("a" * 32)

    def progress(event):
        if event["state"] == "frame_completed":
            raise _Interrupted()

    with pytest.raises(_Interrupted):
        n1.run_n1(task, old, "cpu", "float32", progress)
    proof = checkpoint.retirement_document(task, old, [dict(pid=1, process_create_time="1")],
                                           dict(pid=2, process_create_time="2"))
    write_immutable(old / "retirement.json", proof)
    return task, old, dict(attempt_directory=str(old), retirement=n1.asset(old / "retirement.json"))


def test_lean_prefix_resumes_and_a_changed_store_stops_resume(synthetic, tmp_path):
    task, old, source = _retired_prefix(tmp_path, synthetic, {"profile": "lean"})
    assert not (old / "frames" / "sample_1" / "native_geometry.npz").exists()
    outcome = n1.run_n1(dict(task, resume_from=source), old.parent / ("b" * 32), "cpu", "float32")
    assert outcome["resumed_frame_count"] == 1
    first, second = list(_frames(outcome))
    assert first[1]["lean_storage"]["store"]["index"]["path"] == str(old / lean.STORE_INDEX)
    assert second[1]["lean_storage"]["store"]["index"]["path"] == str(old.parent / ("b" * 32) / lean.STORE_INDEX)
    for root, result in (first, second):
        lean.read_static_fields(result, root)
        lean.read_products(result, root, ["mass_total"])
    archive = old / lean.STORE_ARCHIVE
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Resume payload/receipt differs"):
        n1.run_n1(dict(task, resume_from=source), old.parent / ("c" * 32), "cpu", "float32")


def test_products_index_lists_the_run_store_and_omits_nothing_else(synthetic, tmp_path):
    from darsia.fluidflower.pipeline import publication_runtime as runtime
    lean_run = _run(tmp_path, "lean", synthetic, {"profile": "lean"})
    token = "d" * 32
    claim = types.SimpleNamespace(task=dict(run=synthetic["run"]), token=token, output=str(tmp_path / "lean"))
    producers = []
    for record in lean_run["records"]:
        root = Path(record["nas_directory"])
        producers.append(dict(case_id=record["frame_id"], attempt_directory=str(tmp_path / "lean"),
                              manifest=lean_run["manifest"], commit=n1.asset(root / "commit.json"),
                              frame_registration=lean_run["manifest"], publication=lean_run["manifest"],
                              result=n1.asset(root / "result.json")))
    result = dict(lean_run, attempt_id=token, frame_producers=producers,
                  table=n1.asset(Path(lean_run["manifest"]["path"])))
    rows, metadata, _ = runtime.extract_products(result, claim, "main", producer_origin_for_attempt=None)
    ids = {row["product_id"]: row for row in rows}
    # Producer attempt directories are named by their attempt token in production.
    assert ids["attempts/lean/" + lean.STORE_INDEX]["kind"] == "operational_receipt"
    assert ids["attempts/lean/" + lean.STORE_ARCHIVE]["kind"] == "map"
    assert "attempts/lean/" + lean.STORE_INDEX in metadata
    for record in lean_run["records"]:
        assert {key for key in ids if key.startswith("frames/" + record["frame_id"] + "/")} == {
            "frames/" + record["frame_id"] + "/" + name for name in record["files"]}
    assert not any(key.endswith("native_geometry.npz") for key in ids)


# ---------------------------------------------------- colour intermediates


def test_colour_intermediate_switch(synthetic, tmp_path):
    tiny = _tiny(tmp_path)
    from darsia.fluidflower.mass.production_precision import PixelMath
    pipeline = n1.N1Pipeline(tiny, precision="float32", pixel_math=PixelMath("float32"))
    with pytest.raises(ValueError, match="writes no colour intermediates"):
        lean.LeanRun(tmp_path / "n1", tiny, pipeline, dict(profile="lean", colour_intermediates="drop"),
                     colour_intermediates_produced=False)
    with pytest.raises(ValueError, match="writes no colour intermediates"):
        _run(tmp_path, "refused", synthetic, {"profile": "lean", "colour_intermediates": "drop"})
    stages = {}
    for retention in ("keep", "drop"):
        output = tmp_path / retention
        output.mkdir()
        run = lean.LeanRun(output, tiny, pipeline, dict(profile="lean", colour_intermediates=retention),
                           colour_intermediates_produced=True)
        stages[retention] = []
        for case in ("sample_1", "sample_3"):
            stage = output / "preparation" / case
            stage.mkdir(parents=True)
            for name in ("native_log.npy", "temporal_log.npy", "native_valid.npy", "temporal_valid.npy"):
                (stage / name).write_bytes(b"x")
            run.committed(stage)
            stages[retention].append(stage)
        if retention == "drop":
            first, last = stages[retention]
            assert sorted(p.name for p in first.iterdir()) == ["native_valid.npy", "temporal_valid.npy"]
            assert (last / "native_log.npy").exists() and (last / "temporal_log.npy").exists()
        run.finish()
    for stage in stages["keep"]:
        assert len(list(stage.iterdir())) == 4
    for stage in stages["drop"]:
        assert sorted(p.name for p in stage.iterdir()) == ["native_valid.npy", "temporal_valid.npy"]
    output = tmp_path / "foreign"
    output.mkdir()
    run = lean.LeanRun(output, tiny, pipeline, dict(profile="lean", colour_intermediates="drop"),
                       colour_intermediates_produced=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "native_log.npy").write_bytes(b"x")
    run.committed(outside)
    with pytest.raises(ValueError, match="own preparation"):
        run.committed(outside)
    assert (outside / "native_log.npy").exists()


def test_dropped_colour_intermediates_are_declared_and_tolerated_by_resume_checks(tmp_path):
    from darsia.fluidflower.pipeline import runner
    result = dict(lean_storage=dict(format=lean.LEAN_FORMAT, profile="lean", colour_intermediates="drop",
                                    dropped_after_commit=list(lean.COLOUR_INTERMEDIATES)))
    assert lean.dropped_after_commit(result) == frozenset(lean.COLOUR_INTERMEDIATES)
    assert lean.dropped_after_commit({}) == frozenset()
    with pytest.raises(ValueError):
        lean.dropped_after_commit(dict(lean_storage=dict(result["lean_storage"],
                                                          dropped_after_commit=["phase_area.npz"])))
    stage = tmp_path / "attempt" / "preparation" / "sample_1"
    stage.mkdir(parents=True)
    spec = dict(source=dict(source_sha256="e" * 64))
    history = n1._write_json(stage / "history_input.json", dict(source=spec["source"], full={},
                                                                 temporal_log={}, temporal_valid={}))
    files = {name: dict(path=str(stage / name), sha256="0" * 64, size_bytes=1)
             for name in lean.COLOUR_INTERMEDIATES}
    product = dict(phase=dict(history_input=history), files=files)
    inputs = types.SimpleNamespace(reader=n1.DataReader())
    with pytest.raises(ValueError, match="start a fresh attempt"):
        runner._restore_predecessor(inputs, dict(result, prediction=dict(pipeline=product)), spec)


def test_series_resume_validator_accepts_only_declared_dropped_intermediates(tmp_path):
    from darsia.fluidflower.pipeline import runner
    lifecycle = _module("tests/fluidflower/test_pipeline_public_lifecycle.py", "_lean_lifecycle")
    spec, result, final, config = lifecycle._frame_inventory(tmp_path)
    product = result["prediction"]["pipeline"]
    temporal = Path(product["files"]["temporal_log.npy"]["path"])
    temporal.unlink()
    with pytest.raises(FileNotFoundError):
        runner.validate_frame_artifacts(spec, result, final, config)
    result["lean_storage"] = dict(format=lean.LEAN_FORMAT, profile="lean", colour_intermediates="drop",
                                  dropped_after_commit=list(lean.COLOUR_INTERMEDIATES))
    assert runner.validate_frame_artifacts(spec, result, final, config) is product
    Path(product["files"]["labels.npy"]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        runner.validate_frame_artifacts(spec, result, final, config)
