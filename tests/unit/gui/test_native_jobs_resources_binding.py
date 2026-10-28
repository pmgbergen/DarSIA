"""B6 opt-in owner wiring; real Qt, no probe or job is started."""

from pathlib import Path

from .test_native_jobs_owner import ui, local_registry, snapshot
from darsia.fluidflower.resource_advisor_panel import ResourceAdvisorPanel


def test_b6_factory_connects_actual_resources_without_automatic_probe(
    local_registry, qtbot
):
    def forbidden():
        raise AssertionError("No automatic hardware probe")

    owner = ui.NativeJobsWorkspace(
        registry=local_registry,
        local_machine="Main",
        reader=snapshot,
        inventory_provider=forbidden,
        resource_factory=ResourceAdvisorPanel,
    )
    qtbot.addWidget(owner)
    assert isinstance(owner.resources, ResourceAdvisorPanel)
    assert owner.panel.resource_panel is owner.resources
    assert owner.can_close()
    owner.resources.hardware_inspected.emit(
        dict(host="Main recorded facts", memory={"available_bytes": 128})
    )
    assert owner._machines_provider()[0]["inventory"]["host"] == "Main recorded facts"
    owner.resources.workers_selected.emit(
        dict(machine="MAIN", workers=2, backend="cpu")
    )
    assert owner.panel.workers_spin.value() == 2
    assert not local_registry.root.exists()
