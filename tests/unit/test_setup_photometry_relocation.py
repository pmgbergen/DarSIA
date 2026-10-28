"""Optical preparation keeps declared relocations inside original Setup I/O."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml

from darsia.fluidflower import image_normalisation as choices
from darsia.fluidflower.mass.n1_pipeline import DataReader, _write_json, asset
from darsia.fluidflower.pipeline import current_image_preparation as optical
from darsia.fluidflower.pipeline.current_inputs import CurrentInputReader
from darsia.fluidflower.pipeline.series_setup_run_io import _OriginalSetupReader


@contextmanager
def original_reader(runtime=None):
    from darsia.fluidflower.pipeline import run_io
    runtime = runtime or SimpleNamespace(setup_controls={}, reader_verifications=lambda: {})
    token = run_io._active.set(_OriginalSetupReader(runtime))
    try:
        yield CurrentInputReader(DataReader())
    finally:
        run_io._active.reset(token)


def moved_build(tmp_path, monkeypatch, *, declared=True):
    import darsia
    from darsia.fluidflower import storage_paths
    old, current = tmp_path / "old-layout", tmp_path / "current-layout"
    current.mkdir()
    registry = tmp_path / "registry.toml"
    registry.write_text(toml.dumps(dict(schema_version=1, storage=dict(path_aliases=
        {str(old): str(current)} if declared else {}))), encoding="utf-8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    storage_paths.refresh_storage_paths()

    def logical(pin):
        return dict(pin, path=str(old / Path(pin["path"]).relative_to(current)))

    def document(name, value):
        return logical(_write_json(current / name, value))

    raw = current / "baseline.ARW"
    raw.write_bytes(b"selected acquisition identity, not decoded by this location test")
    profile = document("camera.json", dict(camera="selected"))
    table = current / "materials.csv"
    table.write_text("id,name,analysis_role\n12,Sand,porous_medium\n4,Water,free_water\n")
    adopted = document("revision.json", dict(artifacts=dict(source_mode_files=dict(
        facies_properties=logical(asset(table)))), configuration=dict(regions=[
            dict(label_id=31, facies_id=12), dict(label_id=17, facies_id=4)])))
    plan = document("plan.json", dict(adopted_revision=adopted))
    request = document("request.json", dict(run_id="run", baseline=logical(asset(raw)), profile=profile))
    labels, facies = np.full((24, 24), 31), np.full((24, 24), 12)
    labels[:4], facies[:4] = 17, 4
    archives = []
    for name, values in (("labels", labels), ("facies", facies),
                         ("shape_corrected_baseline", np.full((24, 24, 3), .3))):
        path = current / (name + ".npz")
        darsia.Image(values, space_dim=2, dimensions=[.024, .024],
                     scalar=values.ndim == 2).save(path)
        archives.append(logical(asset(path)))
    certificate = document("renderer.json", dict(run_id="run", outputs=archives,
        renderer={}, inputs=dict(raw=logical(asset(raw)), profile=profile,
                                 camera_binding=dict(profile_id="camera"))))
    operation = document("operation.json", dict(run="run", operation="build", complete=True,
        worker_references_closed=True, raw_build_request=request, setup_plan=plan,
        baseline_renderer=certificate, produced_files=archives))
    return SimpleNamespace(old=old, current=current, table=table, operation=operation)


def prepare(case, reader, output):
    loaded, supports_seen = [], []
    def collect(context, actual_reader, *, require_chart):
        # The real Build context and stable mask have already consumed their
        # exact relocated archives/CSV. Check adjacent original RAW/profile
        # locators through the same optical reader without decoding RAW again.
        assert actual_reader.json(context.provenance["native_configuration"])["camera"] == "selected"
        assert actual_reader.path(context.provenance["raw_acquisition"]).name == "baseline.ARW"
        loaded.append(context.run)
        return dict(run=context.run, rgb_native=context.baseline_rgb)
    def producer(baselines, choice, supports, root, **kwargs):
        value = baselines["run"]()
        support = supports.get("run")
        supports_seen.append(support)
        assert value["rgb_native"].shape == (24, 24, 3)
        assert support["mask"][10, :3].all()
        assert not support["mask"][:5, 8:16].any()
        return dict(runs={"run": dict(source_binding={"path": "prepared"})})
    choice = choices.choice_record("portable", ["run"], choices.MATCH_REFERENCE, "run",
        color_method=choices.COLOR_OFF, stable_band_mm=2.)
    result = optical.prepare_image_photometry({"run": case.operation}, choice, {}, output,
        reader=reader, collector=collect, producer=producer)
    assert loaded == ["run"] and supports_seen[0]["width_mm"] == 2.
    return result


def test_original_setup_worker_prepares_relocated_build_and_stable_band(tmp_path, monkeypatch):
    case = moved_build(tmp_path, monkeypatch)
    with original_reader() as reader:
        assert prepare(case, reader, tmp_path / "output")["runs"]
        # The following colour-path stage wraps this same reader. Its real
        # material-role consumer must retain the same relocation and IDs.
        from darsia.fluidflower.pipeline import setup_colour_paths
        from darsia.fluidflower.pipeline.setup_image_preparation import image_basis_reader
        colour_reader = image_basis_reader(reader, dict(source_aliases=[]))
        context = optical.load_image_context(case.operation, reader=colour_reader)
        roles, known = setup_colour_paths._roles(context, colour_reader)
        assert roles == [dict(facies_id=12, name="Sand", analysis_role="porous_medium"),
                         dict(facies_id=4, name="Water", analysis_role="free_water")]
        assert known.all()
        # The exact material-table provenance remains its original descriptor;
        # source documents and old layout are never recreated or edited.
        adopted = reader.json(reader.json(reader.json(case.operation)["setup_plan"])["adopted_revision"])
        material = adopted["artifacts"]["source_mode_files"]["facies_properties"]
        assert reader.path(material) == case.table
        assert str(case.old) in material["path"] and not case.old.exists()
        case.table.write_bytes(case.table.read_bytes() + b"9,Changed,excluded\n")
        with pytest.raises(ValueError, match="(size differs|changed|SHA)"):
            reader.path(material)


def test_original_setup_does_not_search_for_undeclared_relocated_inputs(tmp_path, monkeypatch):
    case = moved_build(tmp_path, monkeypatch, declared=False)
    with original_reader() as reader, pytest.raises(FileNotFoundError):
        prepare(case, reader, tmp_path / "output")
    assert not case.old.exists()


def test_exact_admitted_native_locator_is_not_replaced_by_storage_alias(tmp_path, monkeypatch):
    from darsia.fluidflower import storage_paths
    original = dict(path=str(tmp_path / "logical.ARW"), sha256="a" * 64, size_bytes=3)
    physical = dict(original, path=str(tmp_path / "admitted.ARW"))
    monkeypatch.setattr(storage_paths, "storage_path_remapper", lambda: lambda path:
        str(tmp_path / "unlisted.ARW") if path == physical["path"] else path)
    runtime = SimpleNamespace(setup_controls=dict(native_inputs=[dict(logical=original, physical=physical)]),
        reader_verifications=lambda: {}, physical_descriptor=lambda pin: dict(pin, path=physical["path"]))
    with original_reader(runtime) as reader:
        assert reader.physical_descriptor(original) == physical
