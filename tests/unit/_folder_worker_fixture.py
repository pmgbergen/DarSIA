"""Spawnable tiny science fixture; production queue/IO/ownership run unchanged."""
from pathlib import Path
import os


def work(*args):
    from darsia.fluidflower.pipeline import worker, run_io, series_setup_operations
    from darsia.fluidflower.pipeline.job_queue import descriptor, write_immutable
    # The fixture's native parameters are intentionally opaque. This is the
    # scientific/code-input test boundary, not a cache or worker substitute.
    run_io.RunIOBinding._admit_native_code = lambda *_args, **_kwargs: None
    if os.environ.get("DARSIA_TEST_FOLDER_IMAGE_ADMISSION_FAILURE") == "1":
        from darsia.fluidflower.pipeline.series_setup_worker_admission import _ClaimControlSink
        original_mkdir = _ClaimControlSink.mkdir
        def fail_before_image_owner(self, path, **options):
            path = Path(path)
            if path.parent.name == ".attempts" and path.parent.parent.name == "setup-image-preparation":
                raise ValueError("A Setup claim control is outside its exact metadata roots")
            return original_mkdir(self, path, **options)
        _ClaimControlSink.mkdir = fail_before_image_owner

    def execute(task, output, backend, precision, progress, *, stage_workers=1):
        binding = run_io.current_binding()
        image_mode = os.environ.get("DARSIA_TEST_FOLDER_IMAGE_MODE")
        if task["operation"] == "image-preparation" and image_mode:
            from darsia.fluidflower.pipeline.series_setup_operations import _sink
            config = binding.configuration
            partial = Path(config["output_root"]) / task["attempt_id"]
            _sink().mkdir(partial, parents=True, exist_ok=True)
            write_immutable(partial / "corrected-baseline.json", dict(fixture=True, attempt=task["attempt_id"],
                                                                     stage_workers=stage_workers))
            progress(dict(state="setup_progress", operation="image-preparation",
                          run=task["run"], step="prepare_selected_images"))
            if (os.environ.get("DARSIA_TEST_FOLDER_IMAGE_PARTIAL_BASIS") == "1"
                    and config.get("reuse_image_basis") is None):
                from darsia.fluidflower.pipeline.setup_image_preparation import independent_choice, completed_build_operations
                from darsia.fluidflower.native.full_photometry import ORDER
                import hashlib
                repository = Path(__file__).resolve().parents[2]
                identity = {role: hashlib.sha256((repository / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                            for role, relative in run_io.RAW_CODE_ROLES.items()}
                _series, choice = independent_choice(config["choice_authority"])
                photometry = partial / "photometry"
                _sink().mkdir(photometry, parents=True)
                write_immutable(photometry / "image_normalisation_choice.json",
                    dict(choice=choice, source_series=config["choice_authority"]))
                runs = {}
                for run, operation in completed_build_operations(config["job_root"], config["runs"]).items():
                    source = photometry / run
                    _sink().mkdir(source)
                    fit = source / "model.json"
                    write_immutable(fit, dict(source=dict(operation_result=operation)))
                    binding_path = source / "raw_source_binding.json"
                    write_immutable(binding_path, dict(execution_identity=identity, correction_order=ORDER,
                        normalisation=dict(choice=choice)))
                    runs[run] = dict(source_binding=descriptor(binding_path), fit=descriptor(fit))
                write_immutable(photometry / "independent_photometry.json", dict(runs=runs))
            if image_mode == "failed":
                raise ValueError("Synthetic image preparation stopped after a saved partial")
            if image_mode == "cancelled":
                from darsia.fluidflower.job_control import read_control, publish_control
                previous = read_control(args[0])
                publish_control(args[0], "stop", machine="private-host",
                                expected_previous=previous.get("command"))
                worker._job_control_boundary(args[0])
            basis = partial / "prepared_image_basis.json"
            basis_value = dict(format="darsia-setup-image-preparation-v1", complete=True)
            if os.environ.get("DARSIA_TEST_FOLDER_IMAGE_COMPLETE_BASIS") == "1":
                from darsia.fluidflower.pipeline.setup_image_preparation import (
                    independent_choice, completed_build_operations,
                )
                _series, choice = independent_choice(config["choice_authority"])
                from darsia.fluidflower.native.full_photometry import ORDER
                import hashlib
                repository = Path(__file__).resolve().parents[2]
                # The fixture deliberately has no admitted native code lease;
                # bind its synthetic metadata directly to actual source bytes.
                identity = {role: hashlib.sha256((repository / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                            for role, relative in run_io.RAW_CODE_ROLES.items()}
                basis_value.update(choice=choice, choice_authority=config["choice_authority"],
                    operations=completed_build_operations(config["job_root"], config["runs"]),
                    raw_execution_identity=identity, correction_order=ORDER)
            write_immutable(basis, basis_value)
            core = dict(format="darsia-series-setup-result-v1", pipeline="series-setup", complete=True,
                run=task["run"], operation=task["operation"], backend=backend, precision=precision,
                setup_plan=config["setup_plan"], worker_references_closed=True,
                image_preparation=descriptor(basis), phase_inference_performed=False,
                model_training_performed=False, optical_calibration_run=False)
            receipt = Path(output) / "operation_result.json"
            write_immutable(receipt, core)
            return dict(core, files={"operation_result.json": descriptor(receipt)})
        prepared = os.environ.get("DARSIA_TEST_FOLDER_COMPLETED_APPLY")
        if prepared:
            import json
            original_progress = series_setup_operations._operation_progress(progress)
            if task["operation"] == "apply":
                core = json.loads(Path(prepared).read_text())
                original_progress(dict(state="setup_progress", operation="apply", step="reuse_completed_apply",
                                       run=core["children"][0]["run_id"]))
                receipt = Path(output) / "operation_result.json"
                write_immutable(receipt, core)
                return dict(core, files={"operation_result.json": descriptor(receipt)})
            original_progress(dict(state="setup_progress", operation="build", step="verify_saved_geometry",
                                   run=task["run"], run_index=1, run_count=1))
            original_progress(dict(state="setup_progress", operation="build", step="build", run=task["run"]))
            retained_success = task["run"] in os.environ.get("DARSIA_TEST_FOLDER_COMPLETED_BUILD_RUNS", "").split(",")
            if os.environ.get("DARSIA_TEST_FOLDER_PARTIAL_BUILD") == "1" or retained_success:
                import uuid
                from darsia.fluidflower.rig_raw_write import OwnedRawOutput
                request = json.loads(Path(binding.configuration["raw_build_request"]["path"]).read_text())
                owner = OwnedRawOutput(request["output"]["destination"], request["output"],
                    operation_id=uuid.uuid4().hex, request=binding.configuration["raw_build_request"])
                save_before_failure = os.environ.get("DARSIA_TEST_FOLDER_SAVE_BEFORE_FAILURE") == "1"
                if save_before_failure:
                    from darsia.presets.workflows.rig import Rig
                    rig = Rig()
                else:
                    rig = object()
                owner.begin_setup(rig, owner.destination / "figures")
                with owner.open(owner.destination / "figures/corrected_baseline.png") as stream:
                    stream.write(b"synthetic saved image; not a completed rig model")
                if save_before_failure:
                    # Exercise the real Rig.save and its owner completion,
                    # stopping before native qualification/outer publication.
                    import numpy as np
                    import darsia
                    from types import SimpleNamespace
                    saved_rig = rig
                    saved_rig.raw_camera = dict(fixture=True)
                    saved_rig._raw_build = SimpleNamespace(output=owner)
                    saved_rig._raw_build_id = owner.operation_id
                    saved_rig.raw_baseline_sha256 = request["baseline"]["sha256"]
                    saved_rig.baseline_path = Path(request["baseline"]["path"])
                    saved_rig.baseline = darsia.OpticalImage(np.zeros((2, 3, 3)), color_space="RGB")
                    saved_rig.shape_corrected_baseline = saved_rig.baseline.copy()
                    saved_rig.shape_corrections = saved_rig.color_corrections = []
                    for name in ("depth", "labels", "facies", "porosity", "permeability", "image_porosity"):
                        setattr(saved_rig, name, darsia.ScalarImage(np.ones((2, 3))))
                    saved_rig.save(owner.destination)
            if os.environ.get("DARSIA_TEST_FOLDER_BUILD_SUCCEEDS") == "1" or retained_success:
                core = dict(format="darsia-series-setup-result-v1", pipeline="series-setup", complete=True,
                    run=task["run"], operation=task["operation"], backend=backend, precision=precision,
                    setup_plan=binding.configuration["setup_plan"], worker_references_closed=True,
                    fixture_worker_pid=os.getpid())
                if retained_success:
                    import json
                    owner.begin_save(rig, owner.destination)
                    image = owner.destination / "shape_corrected_baseline.npz"
                    with owner.open(image) as stream:
                        stream.write(b"synthetic completed Build field")
                    outputs = {image.name: descriptor(image)}
                    certificate = owner.destination / "baseline_native_renderer.json"
                    with owner.open(certificate) as stream:
                        stream.write(json.dumps(dict(inputs=dict(operation_request=binding.configuration["raw_build_request"]),
                            outputs=outputs)).encode())
                    owner.finish_save()
                    core.update(raw_build_request=binding.configuration["raw_build_request"],
                        baseline_renderer=descriptor(certificate), produced_files=outputs)
                receipt = Path(output) / "operation_result.json"
                write_immutable(receipt, core)
                if task["run"] == os.environ.get("DARSIA_TEST_FOLDER_PAUSE_AFTER_COMPLETED_BUILD"):
                    from darsia.fluidflower.job_control import read_control, publish_control
                    previous = read_control(args[0])
                    publish_control(args[0], "pause", machine="private-host",
                                    expected_previous=previous.get("command"))
                return dict(core, files={"operation_result.json": descriptor(receipt)})
            raise ValueError("Synthetic configuration-parser failure before computation")
        if os.environ.get("DARSIA_TEST_FOLDER_FAIL_BEFORE_APPLY") == "1":
            event = dict(state="setup_progress", operation="apply", step="apply",
                         run=os.environ["DARSIA_TEST_FOLDER_APPLY_REFERENCE"])
            original_progress = series_setup_operations._operation_progress(progress)
            original_progress(event)
            if os.environ.get("DARSIA_TEST_FOLDER_FAIL_CHECK_NOTIFICATIONS") == "1":
                original_progress(dict(event, message="Verifying saved source: reference_model.json"))
            raise FileNotFoundError("Fixture retained metadata source moved before Apply")
        if os.environ.get("DARSIA_TEST_FOLDER_PAUSE_CURRENT") == "1":
            from darsia.fluidflower.job_control import read_control, publish_control
            progress(dict(state="setup_progress", operation=task["operation"], step="tiny_fixture_compute"))
            current = read_control(args[0])
            publish_control(args[0], "pause", machine="private-host", expected_previous=current.get("command"))
            worker._job_control_boundary(args[0])
        if os.environ.get("DARSIA_TEST_FOLDER_WAIT_FOR_CANCEL") == "1":
            import time
            from darsia.fluidflower.job_control import read_control
            write_immutable(Path(output) / "fixture_checkpoint.json", dict(saved=True))
            progress(dict(state="setup_progress", operation=task["operation"], step="tiny_fixture_wait"))
            while read_control(args[0])["action"] != "stop":
                time.sleep(0.05)
            worker._job_control_boundary(args[0])
        if os.environ.get("DARSIA_TEST_FOLDER_PAUSE") == "1":
            from darsia.fluidflower.job_control import ensure_setup_master, publish_control
            ensure_setup_master(args[0], machine="private-host", actor="tiny fixture")
            publish_control(args[0], "pause", machine="private-host", expected_previous=None)
        progress(dict(state="setup_progress", operation=task["operation"], step="tiny_fixture_compute"))
        core = dict(format="darsia-series-setup-result-v1", pipeline="series-setup", complete=True,
            run=task["run"], operation=task["operation"], backend=backend, precision=precision,
            setup_plan=binding.configuration["setup_plan"], worker_references_closed=True)
        receipt = Path(output) / "operation_result.json"
        write_immutable(receipt, core)
        return dict(core, files={"operation_result.json": descriptor(receipt)})

    series_setup_operations.execute_setup = execute
    worker._work(*args)


def main():
    """Run the real parent/child worker lifecycle around this tiny science seam."""
    import argparse
    from types import SimpleNamespace
    from darsia.fluidflower.pipeline import worker
    from darsia.fluidflower.pipeline.job_queue import read_json
    from darsia.fluidflower.store_map import read_store_map
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--stores', required=True)
    parser.add_argument('--machine', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--once', action='store_true')
    options = parser.parse_args()
    stores = read_store_map(options.stores)
    job = read_json(options.job/'job.json')
    args = SimpleNamespace(job=options.job, machine=options.machine, backend='cpu',
        workers=options.workers, once=options.once, exit_when_idle=True, watchdog=False,
        watchdog_seconds=30, _alias_stores=stores, stores=options.stores,
        _cache_terminal_events=[])
    worker._work = work
    print('parent-started', os.getpid(), flush=True)
    with (options.job/'logs'/(options.machine+'-cpu.log')).open('a', encoding='utf-8') as log:
        worker._parent(args, options.machine+'-cpu', job, job['commit'], log)


if __name__ == '__main__':
    main()
