"""Per-machine native configuration paths: storage aliases and series roots.

Local fixtures only. The configuration files are never rewritten; only the
values the native loaders return depend on the reading machine.
"""

import json
from pathlib import Path
import tomllib

import pytest

from darsia.fluidflower import storage_paths
from darsia.fluidflower.pipeline import run_io
from darsia.presets.workflows.config import utils
from darsia.presets.workflows.config.color_to_mass import ColorToMassConfig
from darsia.presets.workflows.config.data import DataConfig
from darsia.presets.workflows.config.depth import DepthConfig
from darsia.presets.workflows.config.facies import FaciesConfig
from darsia.presets.workflows.config.labeling import LabelingConfig
from darsia.presets.workflows.config.protocols import ProtocolsConfig
from darsia.presets.workflows.config.rig import RigConfig
from darsia.presets.workflows.setup.setup_protocols import _imaging_protocol_paths


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Write a per-user registry with the given [storage.path_aliases]."""
    path = tmp_path / "user" / "series_registry.toml"
    path.parent.mkdir()
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(path))

    def write(aliases):
        lines = ["schema_version = 1", "[storage.path_aliases]"]
        lines += [
            f"{json.dumps(str(a))} = {json.dumps(str(b))}" for a, b in aliases.items()
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return write


@pytest.fixture
def no_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "absent.toml"))


@pytest.fixture
def worker(monkeypatch):
    """Simulate an alias-job worker task with an active per-run I/O binding."""
    token = run_io._active.set(object())
    yield
    run_io._active.reset(token)


def _absolute_config(path, old_raw, old_project):
    path.parent.mkdir(parents=True, exist_ok=True)
    folder = f"{old_raw}/AC1"
    path.write_text(
        f"""[data]
folders = [{json.dumps(folder)}]
baseline = "first.jpg"
results = {json.dumps(f"{old_project}/results/ac1")}
format = "JPG"

[protocols]
injection = {json.dumps(f"{old_project}/protocols/injection.csv")}
pressure_temperature = [{json.dumps(f"{old_project}/protocols/pt.xlsx")}, "Sheet1"]
imaging_mode = "Load from CSV"

[protocols.imaging]
{json.dumps(folder)} = {json.dumps(f"{old_project}/protocols/imaging.csv")}

[facies]
props = {json.dumps(f"{old_project}/data/facies.csv")}

[facies.1]
labels = [1, 2]
""",
        encoding="utf-8",
    )
    return path


def _relative_config(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """[data]
folders = ["raw/AC1"]
baseline = "first.jpg"
results = "results/ac1"

[protocols]
injection = "protocols/injection.csv"
pressure_temperature = ["protocols/pt.xlsx", "Sheet1"]
start_reference = "image"
start_reference_image = "raw/AC1/first.jpg"

[protocols.imaging]
"raw/AC1" = "protocols/imaging.csv"

[protocols.blacklist]
"raw/AC1" = "protocols/blacklist.csv"

[protocols.imaging_interval_seconds]
"raw/AC1" = 30

[facies]
props = "data/facies.csv"
path = "results/facies.npz"

[labeling]
colored_image = "data/colored.png"
labels = "results/labels.npz"

[depth]
measurements = "data/depth.csv"
depth_map = "results/depth.npz"

[rig]
width = 0.9
height = 0.5
dim = 2
path = "results/rig"

[color_to_mass]
calibration_folder = "results/calibration"
""",
        encoding="utf-8",
    )
    return path


def _load_all(config):
    data = DataConfig().load(config, require_data=False, require_results=False)
    results = data.results
    return dict(
        data=data,
        protocols=ProtocolsConfig().load(config),
        facies=FaciesConfig().load(config, results),
        labeling=LabelingConfig().load(config, results),
        depth=DepthConfig().load(config, results),
        rig=RigConfig().load(config, results),
        color_to_mass=ColorToMassConfig().load(config, None, results),
    )


# --- storage_path_remapper / _alias_rows ------------------------------------


@pytest.mark.parametrize(
    "value", ["/srv/nas/series/a.toml", r"\\NAS\share\a.toml", "X:/FF/a", "data/x", ""]
)
def test_remap_without_aliases_returns_the_authored_text(no_registry, value):
    assert storage_paths.storage_path_remapper()(value) is value


def test_remap_maps_only_values_under_an_alias_prefix(registry, tmp_path):
    clone = tmp_path / "clone"
    source = tmp_path / "srv/nas/series"
    registry({str(source): clone, r"\\NAS\share\raw": tmp_path / "raw"})
    remap = storage_paths.storage_path_remapper()
    assert remap(str(source / "config/a.toml")) == str(clone / "config/a.toml")
    assert remap(str(source)) == str(clone)
    # A Windows-saved path is readable on another machine through its alias.
    assert remap(r"\\NAS\share\raw\AC1\img.jpg") == str(tmp_path / "raw/AC1/img.jpg")
    for value in (str(source) + "2/a.toml", "data/a.toml", "JPG", str(source.parent)):
        assert remap(value) is value


def test_alias_rows_are_normalized_and_longest_first(registry, tmp_path):
    source = tmp_path / "srv/A"
    destination = tmp_path / "new/a"
    path = registry({str(source): str(destination), str(source / "deeper"): "relative/b"})
    assert storage_paths._alias_rows() == [
        ((source / "deeper").as_posix().casefold(), (path.parent / "relative/b").as_posix()),
        (source.as_posix().casefold(), destination.as_posix()),
    ]


def test_resolve_storage_path_is_unchanged_by_the_refactor(registry, tmp_path):
    source = tmp_path / "srv/old"
    other = tmp_path / "srv/other/x"
    registry({str(source): tmp_path / "new"})
    assert storage_paths.resolve_storage_path(str(source / "x")) == tmp_path / "new/x"
    assert storage_paths.resolve_storage_path(str(other)) == other


def test_resolve_with_rows_maps_chains_and_refuses_cycles():
    assert storage_paths._resolve_with("~/x", []) == Path("~/x").expanduser()
    rows = [("/srv/b", "/srv/c"), ("/srv/a", "/srv/b")]
    assert storage_paths._resolve_with("/srv/a/x", rows) == Path("/srv/c/x")
    with pytest.raises(ValueError, match="Cyclic"):
        storage_paths._resolve_with(
            "/srv/a/x", [("/srv/a", "/srv/b"), ("/srv/b", "/srv/a")]
        )


# --- _get_section_from_toml with storage aliases ----------------------------


def test_without_aliases_loaded_values_are_identical(no_registry, tmp_path):
    raw, project = (tmp_path / "nas/raw").as_posix(), (
        tmp_path / "nas/project"
    ).as_posix()
    config = _absolute_config(tmp_path / "plain.toml", raw, project)
    document = tomllib.loads(config.read_text())
    for name in ("data", "protocols", "facies"):
        section = utils._get_section_from_toml(config, name)
        assert section == document[name]
        assert utils._with_storage_aliases(section) is section
    data = DataConfig().load(config, require_data=False, require_results=False)
    protocols = ProtocolsConfig().load(config)
    assert data.folders == [Path(raw) / "AC1"]
    assert data.results == Path(project) / "results/ac1"
    assert protocols.imaging == {
        Path(raw) / "AC1": Path(project) / "protocols/imaging.csv"
    }
    assert protocols.pressure_temperature == (
        Path(project) / "protocols/pt.xlsx",
        "Sheet1",
    )
    assert (
        FaciesConfig().load(config, data.results).props
        == Path(project) / "data/facies.csv"
    )


def test_aliases_remap_values_and_table_keys_of_native_sections(registry, tmp_path):
    raw, project = tmp_path / "student_raw", tmp_path / "student_clone"
    old_raw = (tmp_path / "nas/raw").as_posix()
    old_project = (tmp_path / "nas/project").as_posix()
    registry({old_raw: raw, old_project: project})
    config = _absolute_config(tmp_path / "cfg" / "run.toml", old_raw, old_project)
    before = config.read_bytes()
    data = DataConfig().load(config, require_data=False, require_results=False)
    protocols = ProtocolsConfig().load(config)
    facies = FaciesConfig().load(config, data.results)
    assert data.folders == [raw / "AC1"] and data.baseline == raw / "AC1/first.jpg"
    assert data.results == project / "results/ac1" and data.results.is_dir()
    assert protocols.imaging == {raw / "AC1": project / "protocols/imaging.csv"}
    assert protocols.injection == project / "protocols/injection.csv"
    assert protocols.pressure_temperature == (project / "protocols/pt.xlsx", "Sheet1")
    assert protocols.imaging_mode == "Load from CSV" and data.format == "JPG"
    assert facies.props == project / "data/facies.csv"
    assert facies.facies_to_labels_map == {1: [1, 2]}
    # [data] folders and [protocols.imaging] keys stay one consistent set.
    assert _imaging_protocol_paths(protocols.imaging, data.folders)
    assert config.read_bytes() == before


def test_alias_keys_that_meet_one_path_are_refused(registry, tmp_path):
    source, destination = tmp_path / "srv/old", tmp_path / "srv/new"
    registry({str(source): str(destination)})
    with pytest.raises(ValueError, match="same path"):
        utils._with_storage_aliases({str(source / "a"): "x", str(destination / "a"): "y"})


def test_alias_job_workers_keep_their_authored_paths(registry, tmp_path, worker):
    old_raw = (tmp_path / "nas/raw").as_posix()
    registry({old_raw: tmp_path / "raw"})
    config = _absolute_config(tmp_path / "run.toml", old_raw, str(tmp_path / "p"))
    section = utils._get_section_from_toml(config, "data")
    assert section["folders"] == [old_raw + "/AC1"]
    assert utils._in_alias_job_worker()


def test_worker_signal_is_only_an_active_binding():
    assert not utils._in_alias_job_worker()


def test_map_strings_rebuilds_only_changed_containers():
    value = {"a": ["x", {"b": "y"}], "c": 1}
    assert utils._map_strings(value, lambda text: text) is value
    changed = utils._map_strings(value, lambda text: text.upper())
    assert changed == {"A": ["X", {"B": "Y"}], "C": 1} and value["a"] == [
        "x",
        {"b": "y"},
    ]


# --- relative paths of a series project -------------------------------------


def test_relative_values_resolve_against_the_series_root_not_the_cwd(
    no_registry, tmp_path, monkeypatch
):
    root = tmp_path / "clone" / "series"
    root.mkdir(parents=True)
    (root / "series.toml").write_text('schema_version = 1\nseries_id = "local"\n')
    config = _relative_config(root / "config" / "runs" / "ac1.toml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = _load_all(config)
    data, protocols = loaded["data"], loaded["protocols"]
    assert data.folders == [root / "raw/AC1"]
    assert data.baseline == root / "raw/AC1/first.jpg"  # Still folder-relative.
    assert data.results == root / "results/ac1" and data.results.is_dir()
    assert protocols.imaging == {root / "raw/AC1": root / "protocols/imaging.csv"}
    assert protocols.blacklist == {root / "raw/AC1": root / "protocols/blacklist.csv"}
    assert protocols.imaging_interval_seconds == {root / "raw/AC1": 30.0}
    assert protocols.start_reference_image == root / "raw/AC1/first.jpg"
    assert protocols.injection == root / "protocols/injection.csv"
    assert protocols.pressure_temperature == (root / "protocols/pt.xlsx", "Sheet1")
    assert _imaging_protocol_paths(protocols.imaging, data.folders)
    assert loaded["facies"].props == root / "data/facies.csv"
    assert loaded["facies"].path == root / "results/facies.npz"
    assert loaded["labeling"].colored_image == root / "data/colored.png"
    assert loaded["labeling"].labels == root / "results/labels.npz"
    assert loaded["depth"].measurements == root / "data/depth.csv"
    assert loaded["depth"].depth_map == root / "results/depth.npz"
    assert loaded["rig"].path == root / "results/rig"
    assert loaded["color_to_mass"].calibration_folder == root / "results/calibration"
    assert not (elsewhere / "results").exists()


def test_relative_values_outside_a_series_keep_the_working_directory_meaning(
    no_registry, tmp_path, monkeypatch
):
    config = _relative_config(tmp_path / "plain" / "run.toml")
    monkeypatch.chdir(tmp_path)
    loaded = _load_all(config)
    assert loaded["data"].folders == [Path("raw/AC1")]
    assert loaded["data"].results == Path("results/ac1")
    assert loaded["protocols"].imaging == {
        Path("raw/AC1"): Path("protocols/imaging.csv")
    }
    assert loaded["facies"].props == Path("data/facies.csv")
    assert loaded["rig"].path == Path("results/rig")


def test_series_root_is_the_nearest_folder_with_series_toml(tmp_path):
    root = tmp_path / "a" / "series"
    (root / "config" / "runs").mkdir(parents=True)
    (root / "series.toml").write_text("schema_version = 1\n")
    common, run = root / "config/common.toml", root / "config/runs/ac1.toml"
    other = tmp_path / "b" / "series"
    other.mkdir(parents=True)
    (other / "series.toml").write_text("schema_version = 1\n")
    assert utils._series_root(run) == root
    assert utils._series_root([common, run]) == root
    assert utils._series_root([run, tmp_path / "outside.toml"]) == root
    assert utils._series_root([run, other / "x.toml"]) is None
    assert utils._series_root(tmp_path / "outside.toml") is None


@pytest.mark.parametrize(
    "value,relative",
    [
        ("raw/AC1", True),
        ("raw\\AC1", True),
        ("", False),
        ("   ", False),
        ("/srv/raw", False),
        ("C:\\raw", False),
        ("C:raw", False),
        ("\\\\nas\\share\\raw", False),
        ("\\raw", False),
    ],
)
def test_relative_path_means_no_drive_and_no_root_in_any_spelling(value, relative):
    assert utils._is_relative_path(value) is relative


def test_config_path_keeps_absolute_empty_and_unbound_values(tmp_path):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text("schema_version = 1\n")
    config = root / "run.toml"
    assert utils._config_path("data/a.csv", config) == root / "data/a.csv"
    assert utils._config_path("/srv/a.csv", config) == Path("/srv/a.csv")
    assert utils._config_path("", config) == Path("")
    assert utils._config_path("data/a.csv", None) == Path("data/a.csv")


def test_config_path_in_an_alias_job_worker_keeps_the_relative_value(tmp_path, worker):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text("schema_version = 1\n")
    assert utils._config_path("data/a.csv", root / "run.toml") == Path("data/a.csv")
