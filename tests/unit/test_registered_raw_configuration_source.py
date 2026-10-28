"""Registered RAW configuration selection with exact private file bindings."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import toml

from darsia.fluidflower import corrected_setup as setup


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _document(path, value):
    return _write(path, (json.dumps(value) + "\n").encode())


@pytest.fixture
def raw_bundle(tmp_path):
    """Keep historical map metadata separate from declared RAW Apply targets."""
    series, run, revision = tmp_path / "series", "sample", "a" * 32
    project = series / "geometry_facies"
    historical = _write(series / "legacy" / "source.toml", b'[data]\nformat = "jpg"\n')
    # The live historical path changed after the original map was made.
    Path(historical["path"]).write_bytes(b'[data]\nformat = "arw"\n# later revision\n')
    raw = b'[data]\nformat = "arw"\n[depth]\ndepth_map = "unchanged-depth.npz"\n'
    targets = [_write(series / "config" / role / (run + ".toml"), raw)
               for role in ("runs", "merged")]
    basis = _document(project / "model" / "basis.json", dict(
        format="darsia-raw-rig-baseline-build-v1", run_id=run, artifact_kind="raw_shape_basis"))
    model = _document(project / "model" / "model.json", dict(
        run_id=run, baseline_native_grid=dict(renderer=basis)))
    current = dict(revision_id=revision, configuration=dict(
        painted_adoption_table={},
        frame_binding=dict(runs={run: dict(configuration=historical)}),
        run_geometry=dict(runs=[dict(run_id=run, registration=dict(target_model=model))])))
    _document(project / "results" / "geometry" / revision / "revision.json", current)
    (series / "series.toml").write_text(toml.dumps(dict(runs=dict(ids=[run]))), encoding="utf8")
    return dict(series=series, project=project, run=run, revision=revision,
                historical=historical, targets=targets, raw=raw)


def _bundle(data):
    return setup._configuration_bundle(data["series"], data["run"], data["historical"],
        project_root=data["project"], revision=data["revision"])


def test_registered_raw_reader_uses_exact_declared_current_settings(raw_bundle):
    data = raw_bundle
    bundle = _bundle(data)
    dual = bundle["dual_configuration_binding"]
    assert dual["format"] == setup.RAW_CONFIGURATION_FORMAT
    assert dual["source_configuration"] == data["historical"]
    assert dual["current_configuration"] == data["targets"][-1]
    assert setup._export_configuration_bytes(data["historical"], None, bundle=bundle) == data["raw"]
    # This is not a relaxed historical hash check.
    with pytest.raises(ValueError, match="source identity mismatch"):
        setup._original_configuration_bytes(data["historical"], None)


@pytest.mark.parametrize("mutation", ["changed_raw", "foreign_source", "foreign_current"])
def test_registered_raw_selection_rejects_changed_or_mismatched_binding(raw_bundle, mutation):
    data = raw_bundle
    bundle = deepcopy(_bundle(data))
    dual = bundle["dual_configuration_binding"]
    if mutation == "changed_raw":
        Path(data["targets"][-1]["path"]).write_bytes(data["raw"] + b"# unexpected change\n")
    elif mutation == "foreign_source":
        dual["source_configuration"] = dict(data["historical"], sha256="b" * 64)
    else:
        dual["current_configuration"] = data["targets"][0]
    with pytest.raises(ValueError, match="source identity mismatch|selected configuration bundle"):
        setup._export_configuration_bytes(data["historical"], None, bundle=bundle)


def test_registered_raw_targets_cannot_escape_selected_series(raw_bundle):
    data = raw_bundle
    manifest = data["series"] / "series.toml"
    manifest.write_text(toml.dumps(dict(runs=dict(ids=[data["run"]],
        merged_files={data["run"]: str(data["series"].parent / "foreign.toml")}))), encoding="utf8")
    with pytest.raises(ValueError, match="escapes the selected series"):
        _bundle(data)


def test_ordinary_source_reader_keeps_exact_identity_and_changed_source_refusal(tmp_path):
    original = b'[data]\nformat = "jpg"\n'
    source = _write(tmp_path / "source.toml", original)
    assert setup._export_configuration_bytes(source, None) == original
    Path(source["path"]).write_bytes(original + b"# unexpected edit\n")
    with pytest.raises(ValueError, match="source identity mismatch"):
        setup._export_configuration_bytes(source, None)
