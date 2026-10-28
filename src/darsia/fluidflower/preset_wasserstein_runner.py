"""Execute the original preset Wasserstein calculation in an owned attempt."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import re
import socket
import time


PIPELINE = "preset-wasserstein"
RESULT_FORMAT = "darsia-preset-wasserstein-result-v1"


def _owned(path, output):
    """Require the actual output and its descendants without link traversal."""
    path, output = Path(path), Path(output)
    root = output.resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError("Preset comparison output escapes its owned attempt")
    current = path
    while current != output:
        if current.is_symlink() or current.parent == current:
            raise ValueError("Preset comparison output must not traverse links")
        current = current.parent
    if output.is_symlink():
        raise ValueError("Preset comparison output must not be a link")
    return path


def _expected_comparisons(config, output, selected):
    """Name the existing traversal without changing its run or ROI order."""
    from darsia.presets.workflows.comparison.comparison_wasserstein import (
        WassersteinDistanceResult,
    )

    if config.wasserstein.times != [
        (selected["time_hours"], selected["tolerance_hours"])
    ]:
        raise ValueError("Preset runtime does not contain the exact selected time")
    expected, names = [], set()
    vtk_root = output / "vtk"
    for run_1 in config.wasserstein.runs:
        for run_2 in config.wasserstein.runs:
            if run_1 >= run_2:
                continue
            for roi in config.wasserstein.roi.values():
                filename = WassersteinDistanceResult.get_filename(
                    run_1, run_2, selected["time_hours"], roi.name
                )
                if filename.name != str(filename):
                    raise ValueError("Original comparison filename is not a leaf name")
                _owned(output / filename, output)
                # Check the unchanged VTK path components before the writer runs.
                _owned(
                    vtk_root / f"roi_{roi.name}" / f"run_{run_1}_to_{run_2}"
                    / f"w1_{selected['time_hours']}",
                    output,
                )
                key = filename.name.casefold()
                if key in names:
                    raise ValueError("Original comparison output filenames collide")
                names.add(key)
                expected.append(dict(
                    run_1=run_1, run_2=run_2, time_hours=selected["time_hours"],
                    roi_name=roi.name, filename=filename.name,
                ))
    if not expected:
        raise ValueError("A preset comparison task needs actual run pairs and ROIs")
    return expected


@contextmanager
def _thread_limits():
    """Cap execution resources without casting or changing original images."""
    from darsia._opencv_runtime import completed_frame, scientific_thread
    import cv2
    import scipy.linalg  # Load the same BLAS libraries before applying their cap.
    from threadpoolctl import threadpool_info, threadpool_limits

    before = scientific_thread("preset-wasserstein:start")
    previous = cv2.getNumThreads()
    try:
        cv2.setNumThreads(1)
        with threadpool_limits(limits=1):
            observed = dict(
                requested_threads=1, cv2_threads=cv2.getNumThreads(),
                threadpools=threadpool_info(), input_dtypes=[],
            )
            yield observed
            observed["opencv_ipp"] = completed_frame(
                before, "preset-wasserstein:complete"
            )
    finally:
        cv2.setNumThreads(previous)


def _observe_dtype(observations, consumer, value):
    """Record only an actual array dtype; never infer a fixed scientific dtype."""
    array = getattr(value, "img", value)
    dtype = getattr(array, "dtype", None)
    if dtype is not None:
        observations.append(dict(
            consumer=consumer, dtype=str(dtype), shape=list(array.shape)
        ))


def _queue_progress_file(path, task, events):
    """Recognize only exact operational records written by JobQueue.progress."""
    from .mass.n1_pipeline import asset
    from .pipeline.products_publication import _bound, _json, _verified_bytes, MAX_METADATA_BYTES

    bound = asset(path)
    record = _json(_verified_bytes(path, _bound(bound), limit=MAX_METADATA_BYTES))
    sequence = int(path.stem.removeprefix("progress_"))
    if (set(record) != {"run", "attempt_id", "sequence", "event", "written_unix_ns", "written_utc"}
            or type(record["sequence"]) is not int or record["sequence"] != sequence
            or sequence >= len(events) or record["run"] != task["run"]
            or record["attempt_id"] != task.get("attempt_id")
            or record["event"] != events[sequence]
            or type(record["written_unix_ns"]) is not int or record["written_unix_ns"] < 1
            or not isinstance(record["written_utc"], str) or not record["written_utc"]):
        raise ValueError("Queue progress differs from this exact comparison callback")
    return sequence, bound


def run_preset_wasserstein(task, output, backend, precision, on_progress=None):
    """Run one original time selection after exact shared-worker admission."""
    from .preset_wasserstein_protocol import validate_preset_wasserstein_task
    from .pipeline.run_io import for_task
    from .pipeline.job_queue import retry_transient
    from .mass.n1_pipeline import asset
    from .preset_wasserstein_inputs import load_preset_wasserstein_runtime
    from darsia.presets.workflows.comparison.comparison_wasserstein import (
        _compute_wasserstein_distances,
    )
    from darsia.presets.workflows.rig import Rig

    validate_preset_wasserstein_task(task)
    if (backend != "cpu" or precision != "original"
            or task["backend"] != backend or task["precision"] != precision):
        raise ValueError("Preset Wasserstein preserves original CPU image dtypes")
    if task.get("resume_from") is not None:
        raise ValueError("Preset comparison times retry whole; frame resume is unsupported")
    binding = for_task(task)
    if binding is None:
        raise ValueError("Preset comparison needs its exact admitted task I/O binding")
    binding.require_output(output)
    output = Path(output)
    _owned(output, output)
    retry_transient(lambda: output.mkdir(parents=True, exist_ok=True))
    if retry_transient(lambda: any(output.iterdir())):
        raise FileExistsError("A preset comparison attempt must be fresh")

    selected = deepcopy(task["comparison_time"])
    started = time.perf_counter()
    comparisons = []
    progress_events = []
    with _thread_limits() as execution:
        with load_preset_wasserstein_runtime(task) as runtime:
            if any(getattr(runtime, name, None) is None for name in (
                "config", "reference_rig", "reference_experiment"
            )) or not all(callable(getattr(runtime, name, None)) for name in (
                "load_mass", "verify_consumed"
            )):
                raise ValueError("Preset runtime lacks its bound original dependencies")
            expected = _expected_comparisons(runtime.config, output, selected)
            observations = execution["input_dtypes"]
            for name in ("baseline", "permeability"):
                _observe_dtype(
                    observations, "reference_rig." + name,
                    getattr(runtime.reference_rig, name),
                )
            for name in ("weight", "depth"):
                _observe_dtype(
                    observations, "reference_rig.geometry." + name,
                    getattr(runtime.reference_rig.geometry, name),
                )

            def mass_reader(run_config, *, data, time, tol):
                image = runtime.load_mass(run_config, data=data, time=time, tol=tol)
                if image is None:
                    # The bound reader admits None only for the frozen original
                    # no-match selection; preserve the original missing branch.
                    return None
                runs = [run for run, config in runtime.config.runs.config.items()
                        if config is run_config]
                _observe_dtype(observations, "mass:" + ",".join(map(str, runs)), image)
                return image

            def on_result(result, path):
                index = len(comparisons)
                if index >= len(expected):
                    raise ValueError("Original comparison produced an undeclared result")
                declaration = expected[index]
                identity = dict(
                    run_1=result.run_1, run_2=result.run_2, time_hours=result.time,
                    roi_name=result.roi_name,
                )
                if (identity != {key: declaration[key] for key in identity}
                        or Path(path) != output / declaration["filename"]):
                    raise ValueError("Original comparison result order or identity differs")
                descriptor = asset(_owned(path, output))
                record = dict(identity, status=result.status, file=descriptor)
                comparisons.append(record)
                if on_progress is not None:
                    event = dict(
                        state="preset_wasserstein_progress", run=task["run"],
                        stage="comparison_written", comparison_time=deepcopy(selected),
                        completed_count=len(comparisons), expected_count=len(expected),
                        comparison=deepcopy(record),
                    )
                    progress_events.append(deepcopy(event))
                    on_progress(event)

            _compute_wasserstein_distances(
                Rig, runtime.config, False,
                reference_rig=runtime.reference_rig,
                reference_experiment=runtime.reference_experiment,
                mass_reader=mass_reader, results_dir=output, vtk_root=output / "vtk",
                on_result=on_result,
            )
            if len(comparisons) != len(expected):
                raise ValueError("Original comparison did not produce every declared result")
            source_consumption = runtime.verify_consumed()
            if not isinstance(source_consumption, dict):
                raise ValueError("Preset runtime did not provide actual source consumption")

    files, queue_progress = {}, {}
    original_names = {declaration["filename"] for declaration in expected}
    for path in sorted(output.rglob("*")):
        _owned(path, output)
        if path.is_file():
            relative = path.relative_to(output).as_posix()
            if relative in original_names or relative.startswith("vtk/"):
                files[relative] = asset(path)
            elif re.fullmatch(r"progress_[0-9]{8}\.json", relative):
                sequence, bound = _queue_progress_file(path, task, progress_events)
                queue_progress[sequence] = bound
            else:
                raise ValueError("Preset output contains an undeclared producer file: " + relative)
        elif not path.is_dir():
            raise ValueError("Preset output contains an unsupported filesystem entry")
    if queue_progress and sorted(queue_progress) != list(range(len(progress_events))):
        raise ValueError("Comparison queue progress does not cover the actual callback sequence")
    for declaration, comparison in zip(expected, comparisons):
        if files.get(declaration["filename"]) != comparison["file"]:
            raise ValueError("Original comparison output changed after its completion event")
    vtk_files = [name for name in files if name.startswith("vtk/")]
    result = dict(
        format=RESULT_FORMAT, pipeline=PIPELINE, complete=True,
        run=task["run"], job_id=task["job_id"], comparison_time=selected,
        backend=backend, precision=precision, host=socket.gethostname(), frames=0,
        expected_count=len(expected), comparison_count=len(comparisons),
        comparisons=comparisons, files=files, output_count=len(files),
        queue_progress=[queue_progress[index] for index in sorted(queue_progress)],
        vtk=dict(files=vtk_files, actual_count=len(vtk_files),
                 state="files_written" if vtk_files else "no_files_written"),
        source_consumption=source_consumption, runtime=execution,
        wall_seconds=time.perf_counter() - started, worker_references_closed=True,
        validation=dict(state="not_scored", reason="Execution closure is not scientific acceptance"),
    )
    from .preset_wasserstein_results import write_preset_wasserstein_receipt

    return write_preset_wasserstein_receipt(result, task=task, output_root=output)
