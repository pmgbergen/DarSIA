"""Source-authored checks of real preset metadata readers; not executed for B4."""
import json
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from darsia.fluidflower.preset_wasserstein_inputs import (
    _Scope, catalogue_preset_wasserstein_inputs, compile_preset_wasserstein_inputs,
    load_preset_wasserstein_runtime, read_bytes,
)
from darsia.fluidflower.pipeline.job_inputs import InputCatalog


def test_unscoped_readers_follow_declared_moves_without_rewriting_contracts(tmp_path, monkeypatch):
    import io
    import toml
    from darsia.fluidflower import preset_wasserstein_inputs as inputs
    from darsia.fluidflower import storage_paths

    old, moved = tmp_path / "authored", tmp_path / "moved"
    moved.mkdir()
    raw = b'{"source":"unchanged"}\n'
    actual = moved / "input.json"
    actual.write_bytes(raw)
    registry = tmp_path / "registry.toml"
    registry.write_text(toml.dumps(dict(schema_version=1,
        storage=dict(path_aliases={str(old): str(moved)}))), encoding="utf-8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    storage_paths.refresh_storage_paths()
    authored = old / "input.json"
    contract = dict(path=str(authored), contents=raw.decode())
    before = deepcopy(contract)

    assert inputs.read_bytes(authored) == raw
    assert inputs.read_text(authored, encoding="utf-8") == raw.decode()
    with inputs.open_read(authored, "rb") as stream:
        assert stream.read() == raw
    assert inputs.file_stat(authored).st_size == len(raw)
    assert inputs.exists(authored, "is_file")
    assert list(inputs.glob_paths(old, "*.json")) == [actual]
    assert inputs.list_names(old) == [actual.name]
    assert inputs.source(authored) == actual
    assert inputs.source(str(authored)) == str(actual)
    assert inputs.table_source(authored) == actual
    stream = io.BytesIO(raw)
    assert inputs.source(stream) is stream
    assert contract == before and not old.exists()
    with pytest.raises(ValueError, match="read-only"):
        inputs.open_read(authored, "wb")
    # This seam is also used by original config constructors. A read alias must
    # never silently redirect an output directory to retained source storage.
    inputs.make_directory(old / "new-output", parents=True)
    assert (old / "new-output").is_dir()
    assert not (moved / "new-output").exists()


def test_explicit_preset_scope_keeps_its_own_admission_and_verified_streams(tmp_path, monkeypatch):
    import io
    from darsia.fluidflower import preset_wasserstein_inputs as inputs
    from darsia.fluidflower import storage_paths

    original = tmp_path / "input.json"
    original.write_bytes(b'{"value":1}\n')
    monkeypatch.setattr(storage_paths, "resolve_storage_path",
        lambda *_a, **_kw: pytest.fail("An explicit input scope used global storage aliases"))
    with _Scope(roots={"inputs": str(tmp_path)}).enter() as scope:
        assert inputs.read_bytes(original) == original.read_bytes()
        assert inputs.read_text(original, encoding="utf-8") == original.read_text(encoding="utf-8")
        with inputs.open_read(original, "rb") as stream:
            assert stream.read() == original.read_bytes()
        assert isinstance(inputs.source(original), io.BytesIO)
        assert isinstance(inputs.table_source(original), io.BytesIO)
        assert inputs.file_stat(original).st_size == original.stat().st_size
        assert inputs.exists(original, "is_file")
        assert list(inputs.glob_paths(tmp_path, "*.json")) == [original]
        assert inputs.list_names(tmp_path) == [original.name]
        assert scope.consumed


def write_original_preset_fixture(tmp_path, *, spreadsheet=False, no_match=False):
    """Write actual original TOMLs/protocol tables and tiny archive fixtures.

    The archives are only catalogued by preparation; none of these tests runs
    the scientific comparison. Return paths for producer/transport tests too.
    """
    root = tmp_path / "original"
    root.mkdir()
    series = root / "series.toml"
    series.write_text('schema_version = 1\n[runs]\nids = ["z", "a"]\n', encoding="utf-8")
    rig = root / "rig"
    rig.mkdir()
    (rig / "meta_data.json").write_text(json.dumps({"baseline_path": str(root / "images" / "baseline.npz")}), encoding="utf-8")
    for name in ("baseline", "depth", "labels", "facies", "porosity", "permeability", "image_porosity"):
        dtype = np.int32 if name in {"labels", "facies"} else np.float32
        np.savez(rig / (name + ".npz"), array=np.ones((2, 3), dtype=dtype),
                 metadata=dict(space_dim=2, dimensions=[1., 1.5], scalar=True, type="darsia.ScalarImage"))
    injection = pd.DataFrame({"id": [0], "location_x": [.5], "location_y": [.5],
        "start": ["2025-01-01 00:00:00"], "end": ["2025-01-01 02:00:00"], "rate_kg/s": [.001]})
    pt = pd.DataFrame({"datetime": ["2025-01-01 00:00:00"], "pressure_bar": [1.],
        "temperature_celsius": [20.], "pressure_gradient_bar": [0.], "temperature_gradient_celsius": [0.]})
    tables = {}
    for name, frame in (("injection", injection), ("pt", pt)):
        path = root / (name + (".xlsx" if spreadsheet else ".csv"))
        if spreadsheet:
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame({"wrong": [1]}).to_excel(writer, sheet_name="unused", index=False)
                frame.to_excel(writer, sheet_name="bound", index=False)
        else:
            frame.to_csv(path, index=False)
        tables[name] = path
    def table_literal(path):
        return json.dumps([path.name, "bound"] if spreadsheet else path.name)
    mass_paths = {}
    for run in ("z", "a"):
        results, images = root / ("results-" + run), root / ("images-" + run)
        results.mkdir(); images.mkdir()
        mass = results / "mass"
        mass.mkdir()
        mass_paths[run] = mass / "one.npz"
        for name in ("early.npz", "one.npz", "late.npz"):
            np.savez(mass / name, array=np.full((2, 3), 2., dtype=np.float32),
                     metadata=dict(space_dim=2, dimensions=[1., 1.5], scalar=True, type="darsia.ScalarImage"))
        imaging = root / ("imaging-" + run + ".csv")
        pd.DataFrame({"path": ["early.npz", "one.npz", "late.npz"],
            "datetime": ["2025-01-01 00:30:00", "2025-01-01 01:00:00", "2025-01-01 01:30:00"]}).to_csv(imaging, index=False)
        (root / (run + ".toml")).write_text(f'''
[data]
folder = "images-{run}"
baseline = "baseline.npz"
results = "results-{run}"
use_cache = true
[rig]
width = 1.5
height = 1.0
dim = 2
path = "rig"
[protocols]
injection = {table_literal(tables['injection'])}
pressure_temperature = {table_literal(tables['pt'])}
[protocols.imaging]
"images-{run}" = "{imaging.name}"
"results-{run}/mass" = "{imaging.name}"
[[color_channel]]
name = "mass_color"
color_space = "RGB"
channel = "r"
[analysis.mass]
color = "mass_color"
''', encoding="utf-8")
    config = root / "comparison.toml"
    config.write_text(f'''
[run.z]
config = "z.toml"
[run.a]
config = "a.toml"
[data]
results = "{(root / 'uncreated-comparison').as_posix()}"
[[roi]]
name = " whole "
corner_1 = [0.0, 1.0]
corner_2 = [1.5, 0.0]
[wasserstein]
runs = ["z", "a"]
roi = ["whole"]
resize = 0.5
[wasserstein.data.time.sample]
times = [{10.0 if no_match else 1.0}]
tol = 0.1
''', encoding="utf-8")
    return dict(config_path=config, roots={"inputs": str(root)}, series_manifest=series,
                rig_folder=rig, mass_paths=mass_paths, protocol_paths=tables)


def _compile(fixture):
    return compile_preset_wasserstein_inputs(fixture["config_path"], roots=fixture["roots"],
        series_manifest=fixture["series_manifest"], path_aliases=[])


def test_original_config_protocol_selection_and_inert_preparation(tmp_path, monkeypatch):
    fixture = write_original_preset_fixture(tmp_path)
    import darsia
    monkeypatch.setattr(darsia, "imread", lambda *_a, **_k: pytest.fail("Preparation decoded an image"))
    frozen = _compile(fixture)
    assert frozen["runs"] == ["z", "a"]
    assert frozen["reference_run"] == "z"
    assert frozen["rois"] == [" whole "]  # Original RoiConfig.name, not trimmed lookup key.
    assert [Path(row["descriptor"]["path"]) for row in frozen["selected_mass"]] == list(fixture["mass_paths"].values())
    assert not (fixture["config_path"].parent / "uncreated-comparison").exists()
    assert not (fixture["config_path"].parent / "results-z" / "cache").exists()
    catalog = InputCatalog(fixture["roots"])
    assert catalogue_preset_wasserstein_inputs(catalog, {"preset_inputs": frozen}) == frozen


def test_original_explicit_workbook_sheet_is_replayed(tmp_path):
    fixture = write_original_preset_fixture(tmp_path, spreadsheet=True)
    frozen = _compile(fixture)
    catalog = InputCatalog(fixture["roots"])
    catalogue_preset_wasserstein_inputs(catalog, {"preset_inputs": frozen})
    assert all(row["descriptor"] is not None for row in frozen["selected_mass"])


def test_no_match_is_explicit_and_not_a_missing_input(tmp_path):
    frozen = _compile(write_original_preset_fixture(tmp_path, no_match=True))
    assert frozen["selected_mass"] == [
        dict(time_order=0, run="z", descriptor=None), dict(time_order=0, run="a", descriptor=None)]


def test_original_reader_omission_cannot_become_an_optional_section(tmp_path):
    fixture = write_original_preset_fixture(tmp_path)
    frozen = _compile(fixture)
    frozen["files"] = [row for row in frozen["files"]
                       if Path(row["descriptor"]["path"]).name != "injection.csv"]
    with pytest.raises(ValueError, match="unbound"):
        catalogue_preset_wasserstein_inputs(InputCatalog(fixture["roots"]), {"preset_inputs": frozen})


def test_replay_uses_mapped_bytes_and_frozen_directory_inventory(tmp_path):
    import darsia
    from darsia.presets.workflows.rig import Rig
    fixture = write_original_preset_fixture(tmp_path)
    frozen = _compile(fixture)
    original = fixture["config_path"].parent
    mapped = tmp_path / "worker-store"
    shutil.copytree(original, mapped)
    aliases = []
    for row in frozen["files"]:
        logical = row["descriptor"]
        physical = dict(logical, path=str(mapped / Path(logical["path"]).relative_to(original)))
        aliases.append(dict(logical=logical, physical=physical))
    # A later extra candidate must not change the saved directory decision.
    (mapped / "results-z" / "mass" / "new.npz").write_bytes(b"not a selected image")
    catalog = InputCatalog({"inputs": str(mapped)}, aliases)
    catalogue_preset_wasserstein_inputs(catalog, {"preset_inputs": frozen})
    # Only the admitted worker copy remains at a readable location. Original
    # logical paths must not be used for a stat, directory walk or parser read.
    original.rename(tmp_path / "source-offline")
    with _Scope(frozen=frozen, catalog=catalog).enter() as scope:
        assert read_bytes(frozen["source_configuration"]["path"]) == (mapped / "comparison.toml").read_bytes()
        chosen = frozen["selected_mass"][0]["descriptor"]
        image = darsia.imread(Path(chosen["path"]))
        assert image.img.dtype == np.float32
        assert np.array_equal(image.img, np.full((2, 3), 2., dtype=np.float32))
        rig = Rig.load(fixture["rig_folder"])
        assert rig.baseline.img.dtype == np.float32
        assert rig.facies.img.dtype == np.int32
        assert np.array_equal(rig.porosity.img, np.ones((2, 3), dtype=np.float32))
        consumed_names = {Path(row["descriptor"]["path"]).name
                          for row in scope.consumed.values() if row["reads"]}
        assert {"meta_data.json", "baseline.npz", "depth.npz", "labels.npz", "facies.npz",
                "porosity.npz", "permeability.npz", "image_porosity.npz", "one.npz"} <= consumed_names
        assert all(p.name != "new.npz" for p in scope.inventory(original / "results-z" / "mass", None))
        scope.verify_final()


@pytest.mark.parametrize("name", ["baseline.npz", "depth.npz", "labels.npz", "porosity.npz"])
def test_original_rig_loader_refuses_omitted_archive(tmp_path, name):
    from darsia.presets.workflows.rig import Rig
    fixture = write_original_preset_fixture(tmp_path)
    frozen = _compile(fixture)
    catalog = InputCatalog(fixture["roots"])
    catalogue_preset_wasserstein_inputs(catalog, {"preset_inputs": frozen})
    omitted = deepcopy(frozen)
    path = fixture["rig_folder"] / name
    omitted["files"] = [row for row in omitted["files"]
                        if Path(row["descriptor"]["path"]) != path]
    with _Scope(frozen=omitted, catalog=catalog).enter():
        with pytest.raises(ValueError, match="unbound|unfrozen"):
            Rig.load(fixture["rig_folder"])


@pytest.mark.parametrize("name", ["meta_data.json", "labels.npz"])
def test_original_native_proof_readers_require_each_bound_source(tmp_path, name):
    """Exercise native byte guards, without claiming a complete native proof."""
    from darsia.fluidflower.rig_analysis_domain import _checked_json, _source_array
    fixture = write_original_preset_fixture(tmp_path)
    frozen = _compile(fixture)
    catalog = InputCatalog(fixture["roots"])
    catalogue_preset_wasserstein_inputs(catalog, {"preset_inputs": frozen})
    path = fixture["rig_folder"] / name
    binding = next(row["descriptor"] for row in frozen["files"]
                   if Path(row["descriptor"]["path"]) == path)
    reader = _checked_json if name.endswith(".json") else _source_array
    with _Scope(frozen=frozen, catalog=catalog).enter() as scope:
        result = reader(binding)
        if name.endswith(".json"):
            assert result["baseline_path"].endswith("baseline.npz")
        else:
            assert result.dtype == np.int32
            assert np.array_equal(result, np.ones((2, 3), dtype=np.int32))
        assert any(row["descriptor"] == binding and row["reads"]
                   for row in scope.consumed.values())
    omitted = deepcopy(frozen)
    omitted["files"] = [row for row in omitted["files"] if row["descriptor"] != binding]
    with _Scope(frozen=omitted, catalog=catalog).enter():
        with pytest.raises(ValueError, match="unbound"):
            reader(binding)


def test_same_size_changed_bytes_are_rejected_at_original_parser_read(tmp_path):
    fixture = write_original_preset_fixture(tmp_path)
    frozen = _compile(fixture)
    catalog = InputCatalog(fixture["roots"])
    catalogue_preset_wasserstein_inputs(catalog, {"preset_inputs": frozen})
    path = fixture["protocol_paths"]["injection"]
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"0.001", b"0.002"))
    assert path.stat().st_size == len(raw)
    with _Scope(frozen=frozen, catalog=catalog).enter():
        with pytest.raises(ValueError, match="actual parser read"):
            read_bytes(path)


def test_runtime_refuses_unadmitted_call():
    with pytest.raises(ValueError, match="exact active"):
        load_preset_wasserstein_runtime({"pipeline": "preset-wasserstein"})


def test_omitted_original_correction_inventory_member_fails(tmp_path):
    fixture = write_original_preset_fixture(tmp_path)
    correction = fixture["rig_folder"] / "shape_correction_0.npz"
    np.savez(correction, class_name="TypeCorrection", data_type=np.dtype("float32"))
    frozen = _compile(fixture)
    frozen["files"] = [row for row in frozen["files"]
                       if row["descriptor"]["path"] != str(correction)]
    with pytest.raises(ValueError, match="unbound"):
        catalogue_preset_wasserstein_inputs(InputCatalog(fixture["roots"]), {"preset_inputs": frozen})


def test_original_correction_loader_consumes_bound_archive_bytes(tmp_path):
    fixture = write_original_preset_fixture(tmp_path)
    correction = fixture["rig_folder"] / "shape_correction_0.npz"
    np.savez(correction, class_name="TypeCorrection", data_type=np.dtype("float32"))
    frozen = _compile(fixture)
    catalog = InputCatalog(fixture["roots"])
    catalogue_preset_wasserstein_inputs(catalog, {"preset_inputs": frozen})
    import darsia
    with _Scope(frozen=frozen, catalog=catalog).enter() as scope:
        result = darsia.read_correction(correction)
        assert result.data_type == np.dtype("float32")
        assert any(Path(row["descriptor"]["path"]) == correction and row["reads"]
                   for row in scope.consumed.values())


def test_frozen_directory_alias_policy_reuses_original_remapper(tmp_path):
    fixture = write_original_preset_fixture(tmp_path)
    original_root = fixture["config_path"].parent
    alias_root = tmp_path / "historical-spelling"
    # Alias source strings are frozen locator choices; no old root is accessed.
    config = fixture["config_path"]
    raw = config.read_text(encoding="utf-8")
    raw = raw.replace('config = "z.toml"', 'config = ' + json.dumps((alias_root / "z.toml").as_posix()))
    config.write_text(raw, encoding="utf-8")
    aliases = [dict(source=alias_root.as_posix().casefold(), destination=original_root.as_posix())]
    frozen = compile_preset_wasserstein_inputs(config, roots=fixture["roots"],
        series_manifest=fixture["series_manifest"], path_aliases=aliases)
    assert frozen["path_aliases"] == aliases
    catalogue_preset_wasserstein_inputs(InputCatalog(fixture["roots"]), {"preset_inputs": frozen})
