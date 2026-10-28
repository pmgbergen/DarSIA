"""Installed worker capabilities and conservative scheduling hints for Jobs."""

from __future__ import annotations


# These are the dispatchers used by pipeline.worker. Hints describe operations;
# the existing queue schedules whole runs, not their internal steps.
_PIPELINES = {
    "n1": ("Phase and mass calculation", ("cpu", "cuda"), "mass.n1_pipeline.run_n1"),
    "native-phase-n1": ("Native phase and mass calculation", ("cpu", "cuda"), "pipeline.runner.run_series"),
    "lattice-phase-n1": ("Lattice phase calculation", ("cpu", "cuda"), "pipeline.lattice_runner.run_lattice_series"),
    "native-phase-statistics": ("Phase statistics", ("cpu", "cuda"), "pipeline.phase_statistics.run_phase1"),
    "n1-calibration-refit": ("Calibration refit", ("cpu",), "pipeline.refit_runner.run_refit"),
    "series-setup": ("Setup operations", ("cpu",), "pipeline.series_setup_operations.execute_setup"),
    "preset-wasserstein": ("Wasserstein comparison", ("cpu",), "preset_wasserstein_runner.run_preset_wasserstein"),
}


def job_capabilities(facts):
    """Separate supported hardware from measured preference; never guess speed."""
    pipeline = facts.get("pipeline", "n1")
    specification = _PIPELINES.get(pipeline)
    if specification is None:
        return dict(supported_backends=[], scheduling_unit="whole_run", stages=[],
                    reason="This pipeline has no registered installed-worker capability")
    title, supported, dispatcher = specification
    supported = list(supported)
    reason = "CPU/GPU preference is measured from completed local batches"
    if facts.get("execution_mode") == "current_build":
        supported = ["cpu"]
        reason = "Current Build uses CPU workers on the machine that owns the Build"
    elif facts.get("backend", "any") in {"cpu", "cuda"}:
        supported = [backend for backend in supported if backend == facts["backend"]]
        reason = "The saved job restricts its calculation backend"
    elif supported == ["cpu"]:
        reason = "The installed calculation for this pipeline supports CPU only"
    stages = [dict(name="Read inputs and publish results", supported_backends=["cpu"],
                   preference="cpu", reason="Filesystem work is performed by the CPU worker process",
                   independently_scheduled=False),
              dict(name=title, supported_backends=supported,
                   preference="measured" if len(supported) > 1 else "cpu",
                   reason=reason, independently_scheduled=False)]
    return dict(supported_backends=supported, scheduling_unit="whole_run", stages=stages,
                reason=reason, dispatcher=dispatcher)
