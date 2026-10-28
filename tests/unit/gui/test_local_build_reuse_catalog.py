"""Metadata-only regressions; synthetic closure grants no Build acceptance.

Uses the original catalogue fixture. No RAW/archive bytes are decoded, and no
worker, publication authentication, numerical consumer or job is executed.
"""
from copy import deepcopy
import json

import pytest

from darsia.fluidflower.local_build_reuse import verify_local_build_frame_configuration
from darsia.fluidflower.pipeline.job_inputs import InputCatalog
from .test_current_build_catalog import closure_TESTONLY


def local_reuse_TESTONLY():
    files, config = closure_TESTONLY()
    config.pop("mass")
    config.update(format="darsia-native-phase-statistics-run-v1", statistics={})
    config["frames"][0].update(frame_index=1, elapsed_h=0.5)
    binding = config["current_build_binding"]
    operation = binding["operation_result"]
    result = json.loads(files.files[operation["path"]])
    request = json.loads(files.files[result["raw_build_request"]["path"]])
    old_manifest = deepcopy(request["series_context"]["manifest"])
    native = request["profile"]
    raw = ("series_id = \"synthetic\"\n[runs]\nids = [\"ac52\"]\n[native]\n"
           + "configuration = " + json.dumps(native["path"]) + "\n"
           + "sha256 = " + json.dumps(native["sha256"]) + "\n").encode()
    current_manifest = files.file("series.toml", raw)
    frame_data = dict(format="darsia-build-raw-phase-inputs-v1", run=config["run"],
                      native=deepcopy(config["native"]), frames=deepcopy(config["frames"]))
    frame_bound = files.doc("current-frame-data.json", frame_data)
    record = dict(format="darsia-local-build-reuse-v1", scope="local-p1-selection-only",
                  operation_result=deepcopy(operation), current_series_manifest=current_manifest,
                  frame_configuration=frame_bound, accepted_by="Reviewer", reason="Synthetic selection-only test")
    reuse = files.doc("local-reuse.json", record)
    binding["local_build_reuse"] = reuse

    def read(path):
        key = str(path).replace("\\", "/")
        if key.endswith("/series.toml"):
            files.opened.append(key)
            return files.files[key]
        return files.read(path)

    def catalogue():
        return InputCatalog({"series": files.root}, metadata_bytes=read, file_stat=files.stat)

    return files, config, request, record, frame_data, old_manifest, catalogue


def test_current_reuse_catalogue_replaces_only_the_obsolete_manifest_role():
    files, config, request, record, _frames, old, factory = local_reuse_TESTONLY()
    before = deepcopy(config)
    catalogue = factory()
    assert catalogue.current_build(config) == config["current_build_binding"]
    assert config == before
    uses = catalogue.consumers
    assert "current_build.series_manifest" not in uses
    current = [row["reference"] for row in catalogue.entries.values()
               if row["reference"]["relative_path"] == "series.toml"]
    assert current and {row["sha256"] for row in current} == {record["current_series_manifest"]["sha256"]}
    assert old["sha256"] != current[0]["sha256"]
    assert uses["current_build.configuration_files.0"]["sha256"] == request["configuration_files"][0]["sha256"]
    assert uses["current_build.baseline_raw"]["sha256"] == request["baseline"]["sha256"]
    assert "current_build.shape_basis_certificate" in uses
    assert "current_build.protocol.0" in uses
    assert "current_build.ownership.owner" in uses
    assert any(path.endswith("porosity.npz") for path in files.statted)
    assert all(path.endswith((".json", "/series.toml")) for path in files.opened)
    replay = factory()
    replay.current_build(config)
    assert catalogue.entries == replay.entries


def test_no_opt_in_keeps_original_obsolete_manifest_refusal():
    _files, config, _request, _record, _frames, _old, factory = local_reuse_TESTONLY()
    del config["current_build_binding"]["local_build_reuse"]
    with pytest.raises(ValueError, match="size changed"):
        factory().current_build(config)


@pytest.mark.parametrize("field", ["case_id", "frame_index", "elapsed_h", "source", "geometry", "phase_roles"])
def test_reuse_rejects_changed_actual_queued_frame_identity(field):
    _files, config, request, _record, _frames, _old, factory = local_reuse_TESTONLY()
    row = config["frames"][0]
    if field == "case_id":
        row[field] += "_changed"
    elif field in {"frame_index", "elapsed_h"}:
        row[field] += 1
    else:
        row[field] = dict(row[field], changed_TESTONLY=True)
    with pytest.raises(ValueError, match="frame DATA differs"):
        factory().local_build_reuse(config["current_build_binding"]["local_build_reuse"],
            operation_result=config["current_build_binding"]["operation_result"], request=request, config=config)


@pytest.mark.parametrize("target", ["current_series_manifest", "frame_configuration"])
def test_reuse_rejects_metadata_tampering(target):
    files, config, request, record, _frames, _old, factory = local_reuse_TESTONLY()
    path = record[target]["path"]
    raw = files.files[path]
    files.files[path] = raw.replace(b"ac52", b"ac53", 1)
    with pytest.raises(ValueError, match="changed"):
        factory().local_build_reuse(config["current_build_binding"]["local_build_reuse"],
            operation_result=config["current_build_binding"]["operation_result"], request=request, config=config)


def test_reuse_still_refuses_changed_build_toml_size():
    files, config, request, _record, _frames, _old, factory = local_reuse_TESTONLY()
    files.files[request["configuration_files"][0]["path"]] += b"changed"
    with pytest.raises(ValueError, match="size changed"):
        factory().current_build(config)


def test_shared_runtime_join_refuses_enclosing_p1_configuration_as_frame_data():
    _files, config, _request, _record, frames, _old, _factory = local_reuse_TESTONLY()
    verify_local_build_frame_configuration(config, frames)
    with pytest.raises(ValueError, match="separate matching RAW-phase"):
        verify_local_build_frame_configuration(config, config)


def test_literal_frame_index_type_is_not_coerced_by_shared_runtime_join():
    _files, config, _request, _record, frames, _old, _factory = local_reuse_TESTONLY()
    config["frames"][0]["frame_index"] = True
    with pytest.raises(ValueError, match="frame DATA differs"):
        verify_local_build_frame_configuration(config, frames)
