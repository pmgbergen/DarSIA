"""Continue Build preparation from the original committed Apply controls.

This reuses the completed transactions, not a new approval or numerical proof.
The ordinary Build catalogue independently consumes the same selected controls.
"""
from copy import deepcopy
import json
from pathlib import Path


def read_completed_apply(config, plan, revision):
    """Verify the existing coordinator and exact per-run CAS postimages only."""
    from . import corrected_setup as original
    from .pipeline.series_setup_inputs import _completed_controls, _same
    path = Path(plan["control_root"]) / "apply-complete.json"
    if not path.exists():
        return None
    pins = {}

    class Controls:
        def source(self, pin, role, use, *, metadata=False):
            original._verified(pin)
            pins[pin["path"]] = deepcopy(pin)
            return json.loads(original._small(pin["path"])) if metadata else None

        def named(self, selected, role, use, *, metadata=False):
            pin = original._descriptor(selected)
            return pin, self.source(pin, role, use, metadata=metadata)

    controls = Controls()
    bound = original._descriptor(path)
    coordinator = controls.source(bound, "configuration", "coordinator", metadata=True)
    rows, geometry = [], []
    for run in plan["runs"]:
        matches = [row for row in coordinator.get("runs", []) if row.get("run_id") == run]
        if len(matches) != 1:
            raise ValueError("The retained Apply coordinator changed its run scope")
        row = matches[0]
        selected = dict(config, run=run, coordinator_completion=bound,
                        apply_completion=row["apply_completion"],
                        configuration_files=row["configuration_files"])
        checked = _completed_controls(controls, selected, plan, revision)
        for pin in checked["configuration_files"]:
            controls.source(pin, "configuration", "current_configuration")
        exported = controls.source(row["export"], "configuration", "export", metadata=True)
        expected_scope = dict(format="darsia-job-corrected-export-scope-v1",
            job_id=Path(plan["control_root"]).parent.name, project_root=plan["project_root"],
            setup_plan=config["setup_plan"]) if Path(plan["control_root"]).name == "setup-controls" else None
        if not _same(exported.get("setup_operation"), expected_scope):
            raise ValueError("The retained Apply export belongs to another job or plan")
        binding = exported["binding"]
        registered = binding.get("registered_geometry")
        sources = [*binding["sources"].values(), binding["properties"], binding["configured_depth_source"]]
        if registered is not None:
            sources += [*registered["guarded_sources"], *registered["files"].values()]
        guarded = []
        for pin in sources:
            if pin not in guarded:
                guarded.append(deepcopy(pin))
        geometry.append(dict(run_id=run, registered=deepcopy(registered), guarded_sources=guarded))
        rows.append(deepcopy(checked))
    return dict(coordinator=coordinator, bound=bound, runs=rows,
                guarded_geometry=geometry, metadata=list(pins.values()))
