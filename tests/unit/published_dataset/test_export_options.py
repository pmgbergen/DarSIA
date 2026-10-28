"""Export choices retain the saved frame selection and original expected results."""
from dataclasses import replace
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from darsia.fluidflower.published_dataset import builder
from darsia.fluidflower.published_dataset.archives import Member, write_stored_archive
from darsia.fluidflower.published_dataset.export_mass_maps import mass_map_members
from darsia.fluidflower.published_dataset.export_software import software_members
from darsia.fluidflower.published_dataset.job_reader import FinishedJob, Product
from .synthetic import COMMIT, as_json, make_series, sha, write
from .test_builder import request, fictional_machine  # noqa: F401


def _wheel_bytes(*, missing_member=False):
    files = {"numpy-2.4.4.dist-info/METADATA": b"Name: numpy\nVersion: 2.4.4\n",
             "numpy-2.4.4.dist-info/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
             "numpy/__init__.py": b"# Synthetic library fixture\n"}
    records = io.StringIO()
    writer = csv.writer(records, lineterminator="\n")
    for name, raw in files.items():
        writer.writerow([name, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode(), len(raw)])
    writer.writerow(["numpy-2.4.4.dist-info/RECORD", "", ""])
    files["numpy-2.4.4.dist-info/RECORD"] = records.getvalue().encode()
    if missing_member:
        del files["numpy/__init__.py"]
    with io.BytesIO() as stream:
        with zipfile.ZipFile(stream, "w") as archive:
            for name, raw in files.items():
                archive.writestr(name, raw)
        return stream.getvalue()


def software_inventory(series, root):
    """An exact reference inventory, with no dependency on this test host's environment."""
    root.mkdir(parents=True, exist_ok=True)
    write(root / "requirements-worker.lock", series.lock)
    value = dict(format="darsia-export-software-input-v1", python="3.13.9",
                 code=dict(commit=COMMIT, url="https://example.org/darsia"),
                 environment_lock=dict(path="requirements-worker.lock", sha256=sha(series.lock),
                                       size_bytes=len(series.lock)),
                 libraries=[dict(name="numpy", version="2.4.4", url="https://example.org/numpy/2.4.4")])
    path = root / "software.json"
    write(path, as_json(value))
    return path, value


def test_export_keeps_exact_saved_selection_without_a_tier_file(tmp_path):
    series = make_series(tmp_path)
    plan, report = builder.check_build(request(series, tier=None, mass_map_resolution_mm=1,
                                              tiers_file=tmp_path / "no-tier-file.json"))
    assert plan is not None, report.text()
    assert builder.TIER_BUDGETS[plan.manifest["identity"]["tier"]] == max(builder.TIER_BUDGETS.values())
    assert plan.manifest["selection"]["kind"] == "finished-job"
    assert plan.manifest["selection"]["runs"]["r1"]["ordered_case_ids"] == ["r1_f1", "r1_f2", "r1_f3"]
    assert plan.manifest["export_mass_maps"]["resolution_mm"] == 1
    assert plan.manifest["expected"]["runs"] == plan.manifest["export_mass_maps"]["runs"]
    assert not request(series).output.exists()


def test_saved_job_export_never_subsets_frames_and_legacy_tier_checks_stay_strict(tmp_path):
    series = make_series(tmp_path, tiers={"S50": {"runs": {"r1": ["IMG_0001.ARW"]}}})
    plan, report = builder.check_build(request(series, tier=None))
    assert plan is not None, report.text()
    assert len(plan.manifest["runs"][0]["frames"]) == 3
    plan, report = builder.check_build(request(series, tier="S50"))
    assert plan is None and any("exactly the job's frames" in message for message in report.refusals)


def test_saved_job_selection_refuses_changed_prepared_frame_order(tmp_path):
    series = make_series(tmp_path)
    task_path = series.job / "results" / "r1" / "task.json"
    task = json.loads(task_path.read_bytes())
    task["ordered_case_ids"].reverse()
    write(task_path, as_json(task))
    plan, report = builder.check_build(request(series, tier=None))
    assert plan is None and any("original completion" in message for message in report.refusals)


def test_numerical_grid_and_map_spacing_are_independent_without_native_replay(tmp_path):
    from .test_simulator_geometry import FIXTURE, FLAT_DEPTH, layered, write_revision

    series = make_series(tmp_path)
    write_revision(series.series / "geometry", layered())
    wanted = tmp_path / "geometry_request.json"
    wanted.write_text(json.dumps(dict(format="darsia-simulator-geometry-request-v1",
        geometry="geometry/geometry.json", inner_frame="canvas", sand_surface="free_water",
        depth_points=FLAT_DEPTH, fixtures_m=FIXTURE)))
    # Numerical-model products do not consume or publish the original RAW bytes.
    series.camera.rename(tmp_path / "camera-not-mounted")
    report = builder.build_record(request(series, record="S", tier=None, simulator_resolution_mm=5,
                                         mass_map_resolution_mm=1, simulator_geometry=wanted))
    assert report.built, report.text()
    assert report.manifest["simulator"]["resolution_mm"] == 5
    assert report.manifest["simulator"]["geometry"]["resolution_mm"] == 5
    assert report.manifest["export_mass_maps"]["resolution_mm"] == 1
    assert report.manifest["simulator"]["layer"]["resolution_mm"] == 1
    assert report.manifest["simulator"]["layer"]["shape"] == [4, 5]
    assert all("source" in frame and "raw" not in frame
               for run in report.manifest["runs"] for frame in run["frames"])
    readme = (request(series).output / "README.md").read_text()
    assert "series.zip" not in readme and "grid spacing of 5 mm" in readme
    with zipfile.ZipFile(request(series).output / "simulator_r1.zip") as archive:
        assert "products/r1/timeseries/series/protocols/injection.csv" in archive.namelist()
        geometry_files = report.manifest["simulator"]["geometry"]["runs"]["r1"]
        geometry_document = json.loads(archive.read(geometry_files["json"]["path"]))
        assert geometry_document["grid"]["resolution_mm"] == 5
        assert geometry_document["grid"]["shape"] == [8, 12]
        assert geometry_document["grid"]["extent_m"] == [0.0, 0.06, 0.0, 0.04]
        with np.load(io.BytesIO(archive.read(geometry_files["npz"]["path"])), allow_pickle=False) as arrays:
            assert arrays["porosity"].shape == (8, 12)
        # The independently selected 1 mm map is still the producer's exact
        # saved product; the simulator choice neither regenerates nor relabels it.
        frame = report.manifest["simulator"]["runs"]["r1"]["frames"]["r1_f1"]
        original = FinishedJob(series.job)
        assert archive.read(frame["layer"]["path"]) == original.product_bytes(original.layer("r1", "r1_f1"))
    _, report = builder.check_build(request(series, record="S", simulator_resolution_mm=5,
                                            mass_map_resolution_mm=1, require_simulator_geometry=True))
    assert any("simulator geometry request" in message for message in report.refusals)


def test_saved_selection_keeps_the_existing_maximum_record_budget(tmp_path):
    series = make_series(tmp_path)
    report = builder.BuildReport()
    limit = max(builder.TIER_BUDGETS.values())
    oversized = Member("large.npz", sha256="a" * 64, size_bytes=limit + 1, generate=lambda: b"")
    builder.check_size(request(series, tier=None), {"expected.zip": [oversized]}, {}, report)
    assert any("saved-job export budget" in message for message in report.refusals)


def test_coarser_maps_refuse_absent_original_extensive_inputs_without_writing(tmp_path):
    series = make_series(tmp_path)
    plan, report = builder.check_build(request(series, mass_map_resolution_mm=2))
    assert plan is None and any("Export mass maps" in message for message in report.refusals)
    assert not request(series).output.exists()


def test_exact_software_references_travel_with_reproduction_inputs(tmp_path):
    series = make_series(tmp_path)
    path, _ = software_inventory(series, tmp_path / "software-input")
    report = builder.build_record(request(series, tier=None, software_mode="references", software_inventory=path))
    assert report.built, report.text()
    with zipfile.ZipFile(request(series).output / "series.zip") as archive:
        assert archive.read("software/requirements-worker.lock") == series.lock
        versions = json.loads(archive.read("software/versions.json"))
        assert versions["code"]["commit"] == COMMIT
        assert versions["python"] == "3.13.9" and versions["libraries"][0]["version"] == "2.4.4"
    assert {key: value for key, value in report.manifest["software"].items() if key != "inventory"} == versions
    assert "with Python 3.13.9" in (request(series).output / "README.md").read_text()
    from darsia.fluidflower.published_dataset.archives import extract_members
    from darsia.fluidflower.published_dataset.dataset_check import full_check
    for name in ("series.zip", "expected.zip"):
        extract_members(request(series).output / name, request(series).output)
    assert full_check(request(series).output, report.manifest) == []
    (request(series).output / "software" / "versions.json").write_text("changed after unpacking")
    assert "Changed: software/versions.json" in full_check(request(series).output, report.manifest)


def test_software_folder_builds_exact_references_without_an_inventory_file(tmp_path):
    series = make_series(tmp_path / "series")
    folder = tmp_path / "software"
    write(folder / "requirements-worker.lock", series.lock)
    before = {path.name for path in folder.iterdir()}
    plan, report = builder.check_build(request(series, software_mode="references", software_folder=folder,
        software_code_url="https://example.org/darsia", software_python="3.13.9"))
    assert plan is not None, report.text()
    assert plan.manifest["software"]["libraries"] == [dict(name="numpy", version="2.4.4")]
    assert {path.name for path in folder.iterdir()} == before
    assert not request(series).output.exists()


def test_software_folder_refuses_a_missing_locked_library_build(tmp_path):
    from darsia.fluidflower.published_dataset.export_software import folder_inventory
    series = make_series(tmp_path / "series")
    folder = tmp_path / "software"
    write(folder / "requirements-worker.lock", series.lock)
    write(folder / "source.bundle", b"not admitted before all required wheels exist")
    with pytest.raises(ValueError, match="locked library wheel is missing"):
        folder_inventory(request(series, software_mode="include", software_folder=folder,
            software_code_url="https://example.org/darsia", software_python="3.13.9"),
            FinishedJob(series.job), COMMIT)


@pytest.mark.parametrize("change, message", [
    ("commit", "commit differs"), ("lock", "lock differs"),
    ("missing_library", "each locked library"), ("duplicate_library", "each locked library"),
    ("version", "Library version"), ("python", "exact Python"),
    ("credentials", "without credentials"), ("wrong_lock_bytes", "differs from its explicit")])
def test_software_never_substitutes_unbound_code_or_libraries(tmp_path, change, message):
    series = make_series(tmp_path)
    path, value = software_inventory(series, tmp_path / "software-input")
    if change == "commit":
        value["code"]["commit"] = "f" * 40
    elif change == "lock":
        value["environment_lock"]["sha256"] = "f" * 64
    elif change == "missing_library":
        value["libraries"] = []
    elif change == "duplicate_library":
        value["libraries"] *= 2
    elif change == "version":
        value["libraries"][0]["version"] = "2.0.0"
    elif change == "python":
        value["python"] = "3.12.9"
    elif change == "credentials":
        value["code"]["url"] = "https://user:secret@example.org/darsia"
    else:
        (path.parent / "requirements-worker.lock").write_bytes(b"changed")
    write(path, as_json(value))
    with pytest.raises(ValueError, match=message):
        software_members(request(series, software_mode="references", software_inventory=path),
                         FinishedJob(series.job), COMMIT)


def test_include_mode_refuses_a_wheel_with_other_library_metadata(tmp_path):
    series = make_series(tmp_path)
    path, value = software_inventory(series, tmp_path / "software-input")
    wheel = path.parent / "numpy-2.4.4-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("numpy-2.4.4.dist-info/METADATA", "Name: other\nVersion: 2.4.4\n")
    value["libraries"][0].update(path=wheel.name, sha256=sha(wheel.read_bytes()), size_bytes=wheel.stat().st_size)
    write(path, as_json(value))
    with pytest.raises(ValueError, match="Wheel metadata"):
        software_members(request(series, software_mode="include", software_inventory=path),
                         FinishedJob(series.job), COMMIT)


def test_include_mode_supplies_the_exact_git_bundle_and_every_locked_wheel(tmp_path):
    series = make_series(tmp_path / "series")
    path, value = software_inventory(series, tmp_path / "software-input")
    repository = tmp_path / "repository"
    repository.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()
    git("init")
    (repository / "source.txt").write_text("export fixture\n")
    git("add", "source.txt")
    git("-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-m", "Source fixture")
    commit = git("rev-parse", "HEAD")
    bundle = path.parent / "source.bundle"
    git("bundle", "create", str(bundle), "HEAD")
    value["code"].update(commit=commit, bundle=dict(path=bundle.name,
        sha256=sha(bundle.read_bytes()), size_bytes=bundle.stat().st_size))
    wheel = path.parent / "numpy-2.4.4-py3-none-any.whl"
    wheel.write_bytes(_wheel_bytes())
    value["libraries"][0].update(path=wheel.name, sha256=sha(wheel.read_bytes()), size_bytes=wheel.stat().st_size)
    write(path, as_json(value))
    members, document = software_members(request(series, software_mode="include", software_inventory=path),
                                         FinishedJob(series.job), commit)
    by_name = {member.name: b"".join(member.chunks()) for member in members}
    assert by_name["software/source.bundle"] == bundle.read_bytes()
    assert by_name["software/wheels/" + wheel.name] == wheel.read_bytes()
    assert document["code"]["commit"] == commit and document["mode"] == "include"
    # A matching outer hash and advertised commit do not admit a damaged pack.
    bundle.write_bytes(bundle.read_bytes()[:-20])
    value["code"]["bundle"].update(sha256=sha(bundle.read_bytes()), size_bytes=bundle.stat().st_size)
    write(path, as_json(value))
    with pytest.raises(ValueError, match="complete, readable Git object closure"):
        software_members(request(series, software_mode="include", software_inventory=path),
                         FinishedJob(series.job), commit)


@pytest.mark.parametrize("external", ["submodule", "lfs"])
def test_included_source_refuses_external_payloads_absent_from_the_bundle(tmp_path, external):
    from darsia.fluidflower.published_dataset.export_software import _bundle
    repository = tmp_path / "repository"
    repository.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()
    git("init")
    (repository / "source.txt").write_text("source fixture\n")
    git("add", "source.txt")
    git("-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-m", "Base fixture")
    if external == "submodule":
        git("update-index", "--add", "--cacheinfo", "160000," + git("rev-parse", "HEAD") + ",external")
    else:
        (repository / "source.txt").write_text("version https://git-lfs.github.com/spec/v1\n"
            + "oid sha256:" + "a" * 64 + "\nsize 5\n")
        git("add", "source.txt")
    git("-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-m", "External fixture")
    commit = git("rev-parse", "HEAD")
    bundle = tmp_path / "source.bundle"
    git("bundle", "create", str(bundle), "HEAD")
    with pytest.raises(ValueError, match="absent from the bundle"):
        _bundle(bundle, commit)


def test_included_library_refuses_missing_files_even_with_a_matching_outer_hash(tmp_path):
    from darsia.fluidflower.published_dataset.export_software import _wheel
    path = tmp_path / "numpy-2.4.4-py3-none-any.whl"
    path.write_bytes(_wheel_bytes(missing_member=True))
    with pytest.raises(ValueError, match="complete contents"):
        _wheel(path, "numpy", "2.4.4")


def test_generated_archive_member_is_checked_again_while_written(tmp_path):
    bound = b"checked export"
    member = Member("map.bin", sha256=sha(bound), size_bytes=len(bound), generate=lambda: bound)
    result = write_stored_archive(tmp_path / "correct.zip", [member])
    assert result["members"][0]["sha256"] == sha(bound)
    changed = replace(member, generate=lambda: b"changed export")
    with pytest.raises(ValueError, match="SHA-256"):
        write_stored_archive(tmp_path / "changed.zip", [changed])


def _job_from_index(index):
    """Use actual committed storage fixtures with the original FinishedJob access API."""
    from darsia.fluidflower.pipeline import products_publication as publication
    root, document, rows = publication._read_index(index)
    completed = json.loads(publication.read_published_metadata(index))
    run = document["run"]
    products = {key: Product(key, row["machine_id"], row["local_path"], row["sha256"], row["bytes"],
                            row["kind"], None) for key, row in rows.items()}
    return SimpleNamespace(runs=[run], products={run: products}, completed={run: completed},
                           frames=lambda _run: completed["ordered_case_ids"],
                           product_source=lambda product: Path(product.local_path))


def test_attached_coarser_maps_use_the_original_reader_and_exact_conserving_reducer(tmp_path, monkeypatch):
    from tests.unit.test_storage_coarsened_export import _indexed, _options, _read
    from darsia.fluidflower.storage_coarsening import coarsen_verified_extensive

    _, index, machine = _indexed(tmp_path / "source", monkeypatch)
    job = _job_from_index(index)
    original = next(_read(index, _options(machine))["frames"])
    expected = coarsen_verified_extensive(original, target_resolution_mm=2)
    members, section = mass_map_members(job, 2, "expected")
    archive, metadata = members[job.runs[0]]
    with np.load(io.BytesIO(b"".join(archive.chunks())), allow_pickle=False) as arrays:
        for name, value in expected["arrays"].items():
            np.testing.assert_array_equal(arrays[name], value)
    receipt = json.loads(metadata.data)
    assert receipt["conservation"] == expected["metadata"]["conservation"]
    assert section["production_permission_granted"] is False
    assert receipt["native_to_1mm_reexport_gate"] == "NOT_RUN"
    assert b"local_path" not in metadata.data and b"nas_directory" not in metadata.data
    source_archive = Path(original["loaded_assets"][-1]["path"])
    source_archive.write_bytes(b"changed after preflight")
    with pytest.raises(ValueError):
        b"".join(archive.chunks())


def test_mass_map_resolver_refuses_conflicting_original_producer_identity():
    from darsia.fluidflower.published_dataset.export_mass_maps import _resolver
    original = Product("frames/a/map.npz", "first", "/original/map.npz", "a" * 64, 10, "map", None)
    job = SimpleNamespace(products={"r1": {"one": original, "two": replace(original, machine_id="second")}})
    with pytest.raises(ValueError, match="conflicting producer machines"):
        _resolver(job, "r1")
