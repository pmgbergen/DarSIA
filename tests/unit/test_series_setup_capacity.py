"""Actual request-byte admission for Setup caps; no RAW decode or execution.

The fixture's tiny RAW/profile/basis files are deliberately not scientific
certificates. This layer tests exact request metadata and byte ceiling identity, keeping
scientific source verification on the execution route. Original producer/reader tests own native/grid qualification.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from darsia.fluidflower import rig_raw_build as build
from darsia.fluidflower.pipeline import series_setup_capacity as setup


def _descriptor(path):
    """Bind actual private bytes without importing a scientific reader."""
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _json(path, value):
    """Create a private fixture input before invoking the read-only helper."""
    path.write_text(json.dumps(value), encoding="utf-8")
    return _descriptor(path)


@pytest.fixture
def prepared_raw_request(tmp_path):
    """A real full-request descriptor with explicit synthetic leaf inputs."""
    raw = tmp_path / "raw"
    raw.mkdir()
    baseline = raw / "baseline.ARW"
    baseline.write_bytes(b"synthetic acquisition; never decoded")
    config = tmp_path / "run.toml"
    config.write_bytes(b"# exact applied private configuration\n")
    profile = tmp_path / "profile.json"
    profile.write_bytes(b'{"synthetic_profile":true}\n')
    basis = tmp_path / "shape-basis.json"
    basis.write_bytes(b'{"synthetic_basis":true}\n')
    series = tmp_path / "series.toml"
    series.write_bytes(b"# private series\n")
    protocols = []
    for role in ("imaging", "injection", "pressure_temperature"):
        path = tmp_path / (role + ".csv")
        path.write_bytes(b"time,value\n0,1\n")
        protocols.append(dict(role=role, folder=None, sheet=None, source=_descriptor(path)))
    root = tmp_path / "products"
    root.mkdir()
    document = dict(format=build.REQUEST_FORMAT, run_id="sample",
        configuration_files=[_descriptor(config)],
        series_context=dict(root=str(tmp_path), manifest=_descriptor(series)),
        baseline=_descriptor(baseline), profile=_descriptor(profile),
        operation_overrides={}, protocol_inputs=protocols,
        output=dict(destination=str(root / "fresh"), intended_root=str(root),
                    forbidden_raw_roots=[str(raw)], maximum_output_bytes=100_000,
                    maximum_file_bytes=20_000), shape_basis_certificate=_descriptor(basis))
    path = tmp_path / "request.json"
    bound = _json(path, document)
    task = dict(pipeline="series-setup", run="sample", ordered_case_ids=["build"])
    operation = dict(format=setup.SETUP_FORMAT, run="sample", operation="build",
        raw_build_request=bound, metadata_output=dict(maximum_output_bytes=10_000, maximum_file_bytes=2_000))
    return task, operation, path, document


def test_build_caps_are_exact_original_request_plus_separate_metadata_without_writes(prepared_raw_request, tmp_path, monkeypatch):
    """No measurement/profile shortcut supplies or shrinks the two authorities."""
    task, config, _path, document = prepared_raw_request
    from darsia.fluidflower.pipeline import capacity
    def forbidden(*args, **kwargs):
        raise AssertionError("No measured/common/phase capacity route may be used")
    monkeypatch.setattr(capacity, "load_estimate", forbidden)
    monkeypatch.setattr(capacity, "settings", forbidden)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    expected_config, expected_task = deepcopy(config), deepcopy(task)
    actual = setup.capped_setup_output(task, config)
    assert actual.case_count == 1 and actual.operation == "build"
    assert actual.raw_maximum_output_bytes == 100_000
    assert actual.raw_maximum_file_bytes == 20_000
    assert actual.metadata_maximum_output_bytes == 10_000
    assert actual.metadata_maximum_file_bytes == 2_000
    assert actual.products_estimated_bytes == actual.products_per_frame_estimated_bytes == 110_000
    assert actual.transient_estimated_bytes == 0
    assert actual.needed(other_remaining_bytes=31, reserve_bytes=47) == 110_078
    assert actual.frame_progress_release_allowed is False
    assert "ENFORCED OUTPUT CAPS" in actual.model and "not a measurement" in actual.model
    value = asdict(actual)
    assert value["raw_build_request"] == config["raw_build_request"]
    assert value["raw_output_authority"] == document["output"]
    assert value["metadata_output_authority"] == config["metadata_output"]
    assert config["raw_build_request"] in actual.raw_request_inputs
    assert document["shape_basis_certificate"] in actual.raw_request_inputs
    assert all(row["source"] in actual.raw_request_inputs for row in document["protocol_inputs"])
    assert config == expected_config and task == expected_task
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not Path(document["output"]["destination"]).exists()


def test_source_mutation_is_rejected_by_execution_without_reading_payload_for_caps(prepared_raw_request):
    """The output cap is metadata; scientific input drift still refuses execution."""
    task, config, _path, document = prepared_raw_request
    path = Path(document["configuration_files"][0]["path"])
    old = path.read_bytes()
    path.write_bytes(b"X" + old[1:])
    assert path.stat().st_size == len(old)
    assert setup.capped_setup_output(task, config).raw_maximum_output_bytes == 100_000
    with pytest.raises(ValueError, match="changed"):
        build.read_build_request(config["raw_build_request"], artifact_kind="raw_rig_baseline")


def test_repeated_candidate_caps_do_not_read_any_source_payload(prepared_raw_request, monkeypatch):
    task, config, request_path, document = prepared_raw_request
    inputs = {Path(pin["path"]) for pin in [document["baseline"], document["profile"],
        document["series_context"]["manifest"], document["shape_basis_certificate"],
        *document["configuration_files"], *(row["source"] for row in document["protocol_inputs"])]}
    original, opened = Path.open, []
    def guarded(path, *args, **kwargs):
        opened.append(path)
        if path in inputs:
            raise AssertionError("An unclaimed capacity candidate opened its source payload")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    for _ in range(20):
        assert setup.capped_setup_output(task, config).raw_output_authority == document["output"]
    assert opened and set(opened) == {request_path}


def test_request_bytes_cannot_be_changed_under_the_old_descriptor(prepared_raw_request):
    """Whole-output entitlement comes from exactly the already-bound request."""
    task, config, path, document = prepared_raw_request
    document["output"]["maximum_output_bytes"] *= 2
    _json(path, document)
    with pytest.raises(ValueError, match="changed"):
        setup.capped_setup_output(task, config)


def test_valid_request_for_another_run_is_refused(prepared_raw_request):
    """Do not carry a genuine but foreign request into a different run's claim."""
    task, config, path, document = prepared_raw_request
    document["run_id"] = "another"
    config["raw_build_request"] = _json(path, document)
    with pytest.raises(ValueError, match="another Setup run"):
        setup.capped_setup_output(task, config)


@pytest.mark.parametrize("whole,per_file", [(True, 1), (0, 1), (1, 0), (10, 11), (1.5, 1)])
def test_byte_valid_raw_request_still_requires_the_original_writer_ceiling_types(prepared_raw_request, whole, per_file):
    """Preserve the exact OwnedRawOutput positive-integer/whole-file relation."""
    task, config, path, document = prepared_raw_request
    document["output"].update(maximum_output_bytes=whole, maximum_file_bytes=per_file)
    config["raw_build_request"] = _json(path, document)
    with pytest.raises(ValueError, match="positive integer|file ceiling"):
        setup.capped_setup_output(task, config)


@pytest.mark.parametrize("change", ["missing", "extra", "file_larger", "boolean"])
def test_metadata_allowance_is_independent_explicit_and_exact(prepared_raw_request, change):
    """RAW output space does not authorize uncapped result-publication writes."""
    task, config, _path, _document = prepared_raw_request
    if change == "missing":
        config.pop("metadata_output")
    elif change == "extra":
        config["metadata_output"]["measured"] = True
    elif change == "file_larger":
        config["metadata_output"]["maximum_file_bytes"] = 10_001
    else:
        config["metadata_output"]["maximum_output_bytes"] = True
    with pytest.raises(ValueError, match="metadata|Metadata"):
        setup.capped_setup_output(task, config)


def test_apply_has_only_metadata_authority_and_does_not_read_a_future_raw_request(monkeypatch):
    """No RAW request is consumed before actual Apply postimages exist."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Apply must not read a future full RAW request")
    monkeypatch.setattr(build, "read_build_request", forbidden)
    task = dict(pipeline="series-setup", run="sample", ordered_case_ids=["apply"])
    config = dict(format=setup.SETUP_FORMAT, run="sample", operation="apply",
                  metadata_output=dict(maximum_output_bytes=700, maximum_file_bytes=100))
    actual = setup.capped_setup_output(task, config)
    assert actual.products_estimated_bytes == 700
    assert actual.raw_maximum_output_bytes == actual.raw_maximum_file_bytes == 0
    assert actual.raw_build_request is None and actual.raw_output_authority is None
    assert actual.raw_request_inputs == ()
    config["raw_build_request"] = dict(path="future.json", sha256="a" * 64, size_bytes=1)
    with pytest.raises(ValueError, match="future RAW"):
        setup.capped_setup_output(task, config)


@pytest.mark.parametrize("change", ["pipeline", "format", "run", "operation", "multiple_cases"])
def test_only_the_explicit_single_setup_operation_uses_this_capacity_route(prepared_raw_request, change):
    """Generic/Common/phase tasks cannot accidentally inherit a RAW cap model."""
    task, config, _path, _document = prepared_raw_request
    if change == "pipeline":
        task["pipeline"] = "native-phase-statistics"
    elif change == "format":
        config["format"] = "darsia-native-phase-statistics-run-v1"
    elif change == "run":
        task["run"] = "another"
    elif change == "operation":
        config["operation"] = "mass"
    else:
        task["ordered_case_ids"].append("another")
    with pytest.raises(ValueError):
        setup.capped_setup_output(task, config)


@pytest.mark.parametrize("where", ["other_remaining_bytes", "reserve_bytes"])
def test_reserve_and_concurrent_claim_arithmetic_rejects_coercion(prepared_raw_request, where):
    """Do not accept booleans as evidence of concurrent reservations or reserve."""
    task, config, _path, _document = prepared_raw_request
    actual = setup.capped_setup_output(task, config)
    with pytest.raises(ValueError, match="nonnegative integer"):
        actual.needed(**{where: True})
