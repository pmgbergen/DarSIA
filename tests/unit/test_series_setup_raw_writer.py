"""Original RAW serialization beside an active, separately bounded metadata writer.

The existing decoder, camera pixels, registration and runtime observations are
explicitly synthetic. This proves writer routing and original archive replay;
it does not execute the scientific full rig setup or qualify its outputs.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml

import darsia
from darsia.fluidflower import rig_raw_build as build
from darsia.fluidflower.pipeline import capacity, job_queue
from darsia.fluidflower.pipeline.series_setup_admission import prepare_setup_admission
from darsia.fluidflower.pipeline.series_setup_sink import SetupMetadataSink, current_writer
from darsia.presets.workflows.rig import Rig
from .test_rig_raw_build import prepared as prepared_raw
from .test_series_setup_admission import _admission
from .test_series_setup_protocol import _operation


def test_original_raw_save_and_certificate_do_not_spend_metadata_allowance(tmp_path, monkeypatch):
    """Two real bounded authorities retain disjoint serialized files and counts."""
    science = tmp_path / "scientific-inputs"
    science.mkdir()
    case = prepared_raw.__wrapped__(science, monkeypatch)
    basis = case.context.publish(case.rig)
    request = deepcopy(case.context.request)
    destination = case.destination.parent / "full-save"
    case.configuration.write_text(toml.dumps(dict(
        data=dict(raw_camera=case.camera, use_cache=False), rig=dict(path=str(destination)))), encoding="utf-8")
    request.update(format=build.REQUEST_FORMAT,
        configuration_files=list(build.capture_configuration_sources(case.configuration)),
        operation_overrides={}, shape_basis_certificate=basis)
    request["output"]["destination"] = str(destination)
    request_path = science / "full-request.json"
    request_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
    request_pin = build._descriptor(request_path)
    original_inputs = {path: path.read_bytes() for path in
                       (case.source, case.profile, case.configuration, request_path, Path(basis["path"]))}

    # Keep the real original ledgers; only their OS/volume observation seam is
    # the same explicit private fixture used by the admission tests.
    seed = _admission(tmp_path / "capacity", monkeypatch, metadata_cap=4096, bind=False)
    context = seed.test_context
    task = deepcopy(seed.task)
    task.update(operation="build", ordered_case_ids=["build"])
    config = _operation("build", run=task["run"])
    config.update(raw_build_request=request_pin, configuration_files=request["configuration_files"],
                  metadata_output=seed.metadata_limits, output_scope=seed.configuration["output_scope"])

    def ledger_factory(stores, series_id, root):
        """Use genuine reservations with the fixture's synthetic OS facts."""
        return capacity.CapacityLedger(stores, series_id, root, volume_provider=context.volume,
            context_provider=lambda: dict(context.context))

    admission = prepare_setup_admission(context.stores, "sample-series", task, config,
        metadata_roots=[context.metadata_root], scratch_roots=[], ledger_factory=ledger_factory)
    with admission.locked():
        admission.check()
        admission.bind(context.job_root, SimpleNamespace(task=task, token=task["attempt_id"]), context.owner)
    assert admission.raw_authority == request["output"]
    sink = SetupMetadataSink(admission)
    metadata_before, metadata_after = context.metadata_root / "before.json", context.metadata_root / "after.json"
    image = case.rig.shape_corrected_baseline
    scalars = dict(depth=.02, labels=1, facies=1, porosity=.4, permeability=1., image_porosity=.5)
    with sink.activate():
        job_queue.write_immutable(metadata_before, dict(state="before RAW save"))
        spent_before = sink.written_bytes
        operation = build.begin_raw_rig_build(case.configuration, case.source, case.camera, destination,
            configuration_sources=build.capture_configuration_sources(case.configuration), request_binding=request_pin)
        rig = Rig()
        rig.baseline_path = case.source
        rig.baseline = image.copy()
        rig.shape_corrected_baseline = image.copy()
        rig.shape_corrections = list(case.rig.shape_corrections)
        rig.color_corrections = []
        rig._raw_setup_input_metadata = deepcopy(case.rig._raw_setup_input_metadata)
        for field, value in scalars.items():
            setattr(rig, field, darsia.Image(np.full(image.img.shape[:2], value, dtype=np.float64)))
        operation.begin_setup(rig, destination / "figures")
        rig.save(destination)
        certificate = operation.publish(rig)
        assert current_writer() is sink and sink.written_bytes == spent_before
        assert operation.output.save_finished and not operation.output.failed
        for field, value in scalars.items():
            np.testing.assert_array_equal(darsia.imread(destination / (field + ".npz")).img,
                                          getattr(rig, field).img)
        document = json.loads(Path(certificate["path"]).read_bytes())
        assert document["artifact_kind"] == "raw_rig_baseline"
        assert document["shape_basis_consistency"]["byte_exact_verified"] is True
        assert document["production_qualification_claimed"] is False
        assert document["complete_rig_model"] is False
        assert document["inputs"]["raw_camera_origin"] == "ordered configuration files"
        assert Path(certificate["path"]).stat().st_size > sink.maximum
        assert not any(Path(path).is_relative_to(destination) for path in sink.files)
        with pytest.raises(ValueError):
            sink.write_bytes(destination / "wrong-metadata-writer.json", b"{}")
        with pytest.raises(ValueError, match="exclusively reserved"):
            with operation.output.open(context.metadata_root / "wrong-raw-writer.json"):
                pass
        job_queue.write_immutable(metadata_after, dict(state="after RAW save"))
    evidence = sink.close()
    assert evidence["actual_created_bytes"] == metadata_before.stat().st_size + metadata_after.stat().st_size
    assert all(path.read_bytes() == raw for path, raw in original_inputs.items())
    assert not (destination / "wrong-metadata-writer.json").exists()
    assert not (context.metadata_root / "wrong-raw-writer.json").exists()
