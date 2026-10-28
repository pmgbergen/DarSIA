"""The native route is untouched by the lattice pipeline.

Native job preparation and the native runner (up to its run manifest) run in
two fresh processes: one as usual and one in which every lattice module is
blocked from import. Their job, task and manifest files must be the same, and
the usual run must load no lattice module. The worker dispatches the native
pipeline exactly as before. Synthetic metadata only.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import darsia
from darsia.fluidflower.pipeline import alias_job, lattice_runner, worker

LATTICE_MODULES = ("darsia.fluidflower.computing_lattice", "darsia.fluidflower.phase.physical_units",
                   "darsia.fluidflower.pipeline.lattice_inputs", "darsia.fluidflower.mass.lattice_mass",
                   "darsia.fluidflower.pipeline.lattice_runner", "darsia.fluidflower.lattice_storage",
                   "darsia.fluidflower.pipeline.lattice_capacity")

SCRIPT = r'''
import json, sys
from copy import deepcopy
from pathlib import Path

modules, flag, fixture, root = json.loads(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
if flag == "blocked":
    for name in modules:
        sys.modules[name] = None  # any import of a lattice module now fails
from darsia.fluidflower import job_filesystem
from darsia.fluidflower.pipeline import alias_job, runner, series, worker  # noqa: F401
data = json.loads(fixture.read_text(encoding="utf-8"))
series.read_series = lambda *a, **k: (deepcopy(data["value"]), [deepcopy(data["config"])], dict(data["source"]))
series.require_unc = alias_job.require_unc = runner.require_unc = lambda path: None
job_filesystem.inspect_shared_folder = lambda path: job_filesystem.SharedFolderAdmission(
    True, "smb", "unit transport", "no filesystem qualification")
alias_job.prepare_alias_job(data["source"]["path"], root / "job", expected_sha256=data["source"]["sha256"],
                            series_manifest=data["manifest"])


class Stop(Exception):
    pass


def stop(*args, **kwargs):
    raise Stop


series.load_run = lambda task: deepcopy(data["run_config"])
series.SeriesInputs = stop
try:
    runner._run_series(data["task"], root / "attempt", "cpu", "float32", None, {"fixed": True})
except Stop:
    pass
print(json.dumps(dict(loaded=sorted(name for name in modules if sys.modules.get(name) is not None))))
'''


def put(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def fixture(tmp_path):
    """One shared synthetic native series, so both processes read the same bytes."""
    root = tmp_path / "source_series"
    manifest = put(root / "series.toml", b'schema_version=1\nseries_id="Synthetic"\n[roots]\nseries="."\n')

    def data(key):
        return put(root / (key + ".npy"), b"payload never decoded")

    def document(key, value):
        return put(root / (key + ".json"), json.dumps(value, sort_keys=True).encode())
    full = {"native_centre": {"log_linear_rgb": data("native_log"), "valid_mask": data("native_valid")},
            "ring_windows": []}
    frame = {"case_id": "f1", "current": full, "baseline": dict(full),
             **{key: data(key) for key in ("labels", "classes", "area", "photo_top")},
             "original_domain": {"parent_labels_map": put(root / "original_domain.png", b"opaque source labels"),
                                 "native_shape_hw": [2, 2], "frame_index": 0}}
    config = {"format": "darsia-n1-run-v1", "run": "r1", "frames": [frame],
              "calibration": document("calibration", {}), "physical_correction": document("correction", {}),
              "boxes": {},
              "shared": {"physical_reference": data("physical"), "floor_curve": data("floor"), "depth_points": [],
                         "reference_matrix": [],
                         "clearwater": {"rgb": data("clear_rgb"), "valid": data("clear_valid")},
                         "optics": {"metadata": document("optics", {}), "color_paths": {}, "signals": {},
                                    "flash": document("flash", {})}},
              "storage": {"format": "darsia-storage-output-request-v1", "gas_score_encoding": "uint8-la-q255-v1",
                          "resolution_mm": 1.0, "extent_m": [0.0, 0.01, 0.0, 0.01]}}
    bound = put(root / "config.json", json.dumps(config, sort_keys=True).encode())
    source = put(root / "execution.toml", b"# inert synthetic admission fixture\n")
    value = {"job": {"commit": "a" * 40, "precision": "float32", "calibration_sha256": "b" * 64,
                     "environment_lock": {"path": "requirements.lock", "sha256": "c" * 64}},
             "job_id": "synthetic", "runs": [{"run": "r1", "configuration": bound,
                                              "resources": {"backend": "cpu", "threads": 1}}]}
    task = {"run": "r1", "pipeline": "native-phase-n1", "configuration": {"sha256": "d" * 64},
            "commit": "a" * 40, "calibration_sha256": "b" * 64, "precision": "float32", "threads": 1,
            "job_id": "synthetic"}
    run_config = {"run": "r1", "frames": [{"case_id": "f1", "frame_index": 0}], "mass": {"template": {}}}
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(dict(manifest=manifest, config=config, source=source, value=value, task=task,
                                    run_config=run_config)), encoding="utf-8")
    return path


def run(tmp_path, fixture_path, flag):
    """Run native preparation and the runner start in a fresh process."""
    root = tmp_path / flag
    environment = dict(os.environ, PYTHONPATH=str(Path(darsia.__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, "-c", SCRIPT, json.dumps(LATTICE_MODULES), flag, str(fixture_path),
                             str(root)], capture_output=True, text=True, env=environment, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    return root, json.loads(result.stdout.strip().splitlines()[-1])


def normalized(path, root):
    """A produced JSON document with its own root and the random attempt token removed."""
    text = path.read_text(encoding="utf-8")
    for form in (str(root), json.dumps(str(root))[1:-1]):
        text = text.replace(form, "<root>")
    value = json.loads(text)
    value.pop("attempt_id", None)
    return value


def test_the_native_route_runs_the_same_without_the_lattice_modules(tmp_path):
    fixture_path = fixture(tmp_path)
    usual_root, usual = run(tmp_path, fixture_path, "usual")
    blocked_root, blocked = run(tmp_path, fixture_path, "blocked")
    assert usual["loaded"] == [] and blocked["loaded"] == []
    for relative in ("job/job.json", "job/results/r1/task.json", "job/queue/r1.json", "attempt/manifest.json",
                     "attempt/resume.json"):
        assert normalized(usual_root / relative, usual_root) == normalized(blocked_root / relative, blocked_root), relative
    task = normalized(usual_root / "job/results/r1/task.json", usual_root)
    assert task["pipeline"] == "native-phase-n1" and "computing_resolution_mm" not in task
    manifest = normalized(usual_root / "attempt/manifest.json", usual_root)
    assert "lattice" not in manifest and "pipeline" not in manifest


def test_the_worker_dispatches_native_and_lattice_pipelines_to_their_own_runners():
    tree = ast.parse(Path(worker.__file__).read_text(encoding="utf-8"))
    branches = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "pipeline"
                and isinstance(node.test.comparators[0], ast.Constant)):
            imports = [item for item in node.body if isinstance(item, ast.ImportFrom)]
            branches[node.test.comparators[0].value] = (imports[0].module, imports[0].names[0].name)
    assert branches["native-phase-n1"] == ("runner", "run_series")
    assert branches[lattice_runner.PIPELINE] == ("lattice_runner", "run_lattice_series")
    source = Path(worker.__file__).read_text(encoding="utf-8")
    assert source.count("lattice_runner") == 1  # the lattice branch is the worker's only lattice reference


def test_job_preparation_imports_lattice_code_only_for_a_lattice_request():
    source = Path(alias_job.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    guarded = [node for node in ast.walk(tree) if isinstance(node, ast.If) and "computing_resolution_mm" in ast.dump(node)]
    assert len(guarded) == 1
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "lattice_runner"]
    assert len(imports) == 1 and any(imports[0] is node for node in ast.walk(guarded[0]))
