"""Standalone runtime probes validate real local interpreters before science.

Fixtures use disposable Git repositories and a stdlib-created virtual
environment containing only distribution metadata. No analysis is executed.
"""
import json
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
from types import SimpleNamespace
import venv

import pytest

from darsia.fluidflower.published_dataset import staged_runtime as runtime
from .synthetic import as_json, sha, write


def _fixture(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    template = tmp_path / "empty-template"
    template.mkdir()
    marker = tmp_path / "scientific-import-marker"

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), "-c", "core.hooksPath=" + str(template),
            *args], text=True).strip()

    git("init", "--template=" + str(template))
    source = ("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('imported')\n"
              + "IDENTITY = 'original tracked fixture'\n").encode()
    write(root / "src" / "darsia" / "__init__.py", source)
    write(root / ".gitignore", b"src/ignored.py\n__pycache__/\n*.pyc\n")
    lock = ("# Python " + str(sys.version_info.major) + "." + str(sys.version_info.minor)
            + ". Exact fixture environment.\nfixture-library==1.2.3\n").encode()
    write(root / "requirements-worker.lock", lock)
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.org", "-c", "commit.gpgsign=false",
        "commit", "-m", "Exact runtime fixture")
    commit = git("rev-parse", "HEAD")
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False, symlinks=False).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    scheme = sysconfig.get_paths(vars={"base": str(environment), "platbase": str(environment)})
    site = Path(scheme["purelib"])
    metadata = site / "fixture_library-1.2.3.dist-info" / "METADATA"
    write(metadata, b"Metadata-Version: 2.1\nName: fixture-library\nVersion: 1.2.3\n")
    dataset = tmp_path / "dataset"
    lock_name = "software/stages/phase1/requirements-worker.lock"
    write(dataset / lock_name, lock)
    software = dict(mode="references", python=platform.python_version(),
        code=dict(commit=commit, url="https://example.org/fixture"),
        environment_lock=dict(path=lock_name, sha256=sha(lock)),
        libraries=[dict(name="fixture-library", version="1.2.3")])
    versions = as_json(software)
    name = "software/stages/phase1/versions.json"
    write(dataset / name, versions)
    software["inventory"] = dict(path=name, sha256=sha(versions), size_bytes=len(versions))
    expected = dict(commit=commit, environment_lock=dict(path="requirements-worker.lock", sha256=sha(lock)),
                    pipeline="native-phase-statistics", precision="float32", series_id="SYN",
                    expected_runs=["r1"], software=software)
    selected = dict(stage="phase1", python=str(python.absolute()), repository=str(root.resolve()))
    return SimpleNamespace(root=root, git=git, marker=marker, metadata=metadata, site=site,
        dataset=dataset, expected=expected, selected=selected, lock=lock, source=source)


def _rewrite_versions(fixture):
    software = fixture.expected["software"]
    raw = as_json({key: value for key, value in software.items() if key != "inventory"})
    write(fixture.dataset / software["inventory"]["path"], raw)
    software["inventory"].update(sha256=sha(raw), size_bytes=len(raw))


def test_probe_uses_the_selected_real_interpreter_without_importing_science(tmp_path):
    fixture = _fixture(tmp_path)
    # An extra installer distribution is reported, never invented as a lock pin.
    write(fixture.site / "pip-1.0.dist-info" / "METADATA", b"Name: pip\nVersion: 1.0\n")
    # Site startup must not execute arbitrary .pth import hooks either.
    startup = tmp_path / "site-startup-marker"
    write(fixture.site / "startup.pth", ("import pathlib; pathlib.Path(" + repr(str(startup))
          + ").write_text('executed')\n").encode())
    result = runtime.validate_runtime(fixture.selected, fixture.expected, dataset_root=fixture.dataset)
    assert result["format"] == runtime.PROBE_FORMAT
    assert result["code"]["commit"] == fixture.expected["commit"]
    assert result["code"]["tree"] == fixture.git("rev-parse", "HEAD^{tree}")
    assert result["code"]["clean"] and result["code"]["source_files"] == 1
    assert result["python"] == platform.python_version()
    assert result["libraries"] == {"fixture-library": "1.2.3"}
    assert result["extra_distributions"] == {"pip": "1.0"}
    assert result["scientific_imports_started"] is False and result["site_startup_executed"] is False
    assert not fixture.marker.exists() and not startup.exists()


@pytest.mark.parametrize("change, message", [
    ("python", "Python patch"), ("commit", "stage commit"),
    ("dirty_source", "clean Git"), ("ignored_source", "overlays"),
    ("missing_library", "installed libraries"), ("library_version", "installed libraries"),
    ("duplicate_distribution", "duplicate distribution"),
    ("saved_lock", "SHA/size"), ("original_lock", "original stage job"),
    ("inventory", "SHA/size"), ("software_commit", "software commit"),
])
def test_probe_refuses_changed_runtime_identity_before_any_scientific_import(tmp_path, change, message):
    fixture = _fixture(tmp_path)
    if change == "python":
        fixture.expected["software"]["python"] = "0.0.0"
        _rewrite_versions(fixture)
    elif change == "commit":
        fixture.expected["commit"] = "f" * 40
        fixture.expected["software"]["code"]["commit"] = "f" * 40
        _rewrite_versions(fixture)
    elif change == "dirty_source":
        write(fixture.root / "src" / "darsia" / "__init__.py", b"IDENTITY = 'changed'\n")
    elif change == "ignored_source":
        write(fixture.root / "src" / "ignored.py", b"raise RuntimeError('untracked overlay')\n")
    elif change == "missing_library":
        fixture.metadata.parent.rename(fixture.metadata.parent.with_name("fixture-library.unavailable"))
    elif change == "library_version":
        write(fixture.metadata, b"Name: fixture-library\nVersion: 9.9.9\n")
    elif change == "duplicate_distribution":
        write(fixture.site / "fixture_library-9.9.9.dist-info" / "METADATA",
              b"Name: fixture-library\nVersion: 9.9.9\n")
    elif change == "saved_lock":
        write(fixture.dataset / fixture.expected["software"]["environment_lock"]["path"], b"changed")
    elif change == "original_lock":
        fixture.expected["environment_lock"]["sha256"] = "f" * 64
    elif change == "inventory":
        write(fixture.dataset / fixture.expected["software"]["inventory"]["path"], b"changed")
    elif change == "software_commit":
        fixture.expected["software"]["code"]["commit"] = "f" * 40
        _rewrite_versions(fixture)
    with pytest.raises(ValueError, match=message):
        runtime.validate_runtime(fixture.selected, fixture.expected, dataset_root=fixture.dataset)
    assert not fixture.marker.exists()


def test_stage_mapping_has_exact_keys_and_explicit_absolute_paths(tmp_path):
    fixture = _fixture(tmp_path)
    path = tmp_path / "runtimes.json"
    selected = {key: value for key, value in fixture.selected.items() if key != "stage"}
    write(path, as_json(dict(format=runtime.FORMAT, stages={"phase1": selected})))
    assert runtime.load_runtimes(path, {"phase1": fixture.expected}) == {"phase1": fixture.selected}
    for stages in ({}, {"phase1": selected, "unbound": selected}):
        write(path, as_json(dict(format=runtime.FORMAT, stages=stages)))
        with pytest.raises(ValueError, match="every exact reproduction stage"):
            runtime.load_runtimes(path, {"phase1": fixture.expected})
    write(path, as_json(dict(format=runtime.FORMAT, stages={"phase1": dict(selected, python="python")})))
    with pytest.raises(ValueError, match="explicit absolute"):
        runtime.load_runtimes(path, {"phase1": fixture.expected})


def test_source_import_guard_uses_exact_tracked_source_without_site_or_cached_bytecode(tmp_path):
    fixture = _fixture(tmp_path)
    request = dict(runtime=fixture.selected, expected_job=fixture.expected, dataset_root=str(fixture.dataset),
                   driver=str(Path(runtime.__file__).resolve()))
    # This child imports only a tiny original-source fixture, not scientific code.
    script = "\n".join([
        "import json, pathlib, runpy, sys",
        "value = json.load(sys.stdin)",
        "driver = runpy.run_path(value['driver'])",
        "receipt, snapshot, prefix, sites = driver['_probe'](value['runtime'], value['expected_job'], value['dataset_root'])",
        "driver['_install_source_guard'](pathlib.Path(value['runtime']['repository']), snapshot, prefix, sites)",
        "import darsia",
        "assert darsia.IDENTITY == 'original tracked fixture'",
        "assert pathlib.Path(darsia.__file__).resolve().is_relative_to(pathlib.Path(value['runtime']['repository']))",
    ])
    cache = fixture.root / "src" / "darsia" / "__pycache__" / ("__init__." + sys.implementation.cache_tag + ".pyc")
    write(cache, b"unreadable stale bytecode; original tracked .py must be compiled")
    process = subprocess.run([fixture.selected["python"], "-I", "-S", "-B", "-c", script],
        input=as_json(request), capture_output=True, timeout=60)
    assert process.returncode == 0, process.stderr.decode(errors="replace")
    assert fixture.marker.read_text() == "imported"


def test_source_guard_refuses_a_changed_module_after_the_initial_probe(tmp_path):
    fixture = _fixture(tmp_path)
    request = dict(runtime=fixture.selected, expected_job=fixture.expected, dataset_root=str(fixture.dataset),
                   driver=str(Path(runtime.__file__).resolve()))
    script = "\n".join([
        "import json, pathlib, runpy, sys",
        "value = json.load(sys.stdin)",
        "driver = runpy.run_path(value['driver'])",
        "receipt, snapshot, prefix, sites = driver['_probe'](value['runtime'], value['expected_job'], value['dataset_root'])",
        "repository = pathlib.Path(value['runtime']['repository'])",
        "driver['_install_source_guard'](repository, snapshot, prefix, sites)",
        "(repository / 'src/darsia/__init__.py').write_text('IDENTITY = 0')",
        "try:",
        "    import darsia",
        "except ImportError as error:",
        "    assert 'differs from the admitted' in str(error)",
        "else:",
        "    raise AssertionError('changed source admitted')",
    ])
    process = subprocess.run([fixture.selected["python"], "-I", "-S", "-B", "-c", script],
        input=as_json(request), capture_output=True, timeout=60)
    assert process.returncode == 0, process.stderr.decode(errors="replace")
    assert not fixture.marker.exists()


def test_launch_revalidates_runtime_and_preserves_failed_receipt_without_science(tmp_path):
    fixture = _fixture(tmp_path)
    runtime.validate_runtime(fixture.selected, fixture.expected, dataset_root=fixture.dataset)
    write(fixture.metadata, b"Name: fixture-library\nVersion: 9.9.9\n")
    receipt, log = tmp_path / "failed.json", tmp_path / "failed.log"
    process = runtime.launch_stage(fixture.selected, fixture.expected, {}, dataset_root=fixture.dataset,
                                   receipt_path=receipt, log_path=log)
    assert process.wait(timeout=60) == 1
    value = json.loads(receipt.read_bytes())
    assert value["state"] == "failed" and "installed libraries" in value["error"]
    assert value["qualified"] is False and value["production"] is False
    assert not fixture.marker.exists()


def test_pause_before_stage_preserves_receipt_without_importing_or_coupling(tmp_path):
    fixture = _fixture(tmp_path)
    control = tmp_path / "control.json"
    write(control, as_json(dict(action="pause")))
    execution = dict(task=dict(path=str(tmp_path / "unused-task.json"), sha256="a" * 64, size_bytes=1),
                     control=str(control))
    receipt, log = tmp_path / "paused.json", tmp_path / "paused.log"
    process = runtime.launch_stage(fixture.selected, fixture.expected, execution, dataset_root=fixture.dataset,
                                   receipt_path=receipt, log_path=log)
    assert process.wait(timeout=60) == 0
    value = json.loads(receipt.read_bytes())
    assert value["state"] == "paused" and "coupling" not in value and "result" not in value
    assert value["runtime"]["code"]["commit"] == fixture.expected["commit"]
    assert not fixture.marker.exists()


def test_launcher_refuses_existing_receipts_or_writes_inside_selected_source(tmp_path):
    fixture = _fixture(tmp_path)
    receipt = tmp_path / "exists.json"
    receipt.write_text("retain existing result")
    with pytest.raises(FileExistsError, match="fresh receipt"):
        runtime.launch_stage(fixture.selected, fixture.expected, {}, dataset_root=fixture.dataset,
            receipt_path=receipt, log_path=tmp_path / "unused.log")
    with pytest.raises(ValueError, match="outside the selected source"):
        runtime.launch_stage(fixture.selected, fixture.expected, {}, dataset_root=fixture.dataset,
            receipt_path=fixture.root / "result.json", log_path=tmp_path / "unused.log")
    assert receipt.read_text() == "retain existing result"
