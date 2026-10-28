"""The reproduction launcher's machine settings: workers from the published measurement.

The arithmetic is the Jobs panel's (``resource_advisor_panel.resource_advice``):
workers = min(physical cores / threads per worker, (available RAM - reserve) /
memory per worker), never more than the runs in scope. The memory per worker
is the producer's measured peak from the manifest, labelled as measured on the
reference machine, and the reserve is the launcher's own small one; the Jobs
panel's rules for the shared production machine stay as they are.
"""

from __future__ import annotations

from ..resource_advisor_panel import GIB, resource_advice

LAUNCHER_RESERVE_BYTES = 1 * GIB


def machine_inventory() -> dict:
    """This machine's cores and memory (no host or user name)."""
    from ..execution import detect_host_resources

    return detect_host_resources()


def default_workers(manifest: dict, *, runs_in_scope: int, threads: int | None = None,
                    backend: str = "cpu", inventory: dict | None = None,
                    reserve_bytes: int = LAUNCHER_RESERVE_BYTES) -> dict:
    """The worker default for a dataset on this machine, with the reason in words.

    ``workers`` is None when the dataset records no memory per worker (or the
    backend needs a measurement it lacks), and 0 when not even one worker fits;
    then nothing starts unless the user chooses a worker count.
    """
    machine_facts = manifest.get("machine") or {}
    threads = int(threads or machine_facts.get("threads") or 1)
    inventory = machine_inventory() if inventory is None else inventory
    memory = machine_facts.get("peak_memory_bytes")
    measured = None if memory is None else {
        "bytes": int(memory),
        "basis": "Memory per worker measured on the reference machine"
                 + (f" ({machine_facts['peak_memory_basis']})" if machine_facts.get("peak_memory_basis") else "")
                 + "."}
    job = {"precision": (manifest.get("plan") or {}).get("precision"),
           "pipeline": (manifest.get("plan") or {}).get("pipeline"),
           "commit": (manifest.get("code") or {}).get("published_commit")}
    advice = {"workers": None, "reason": "The dataset records no measured memory per worker; "
                                         "choose the worker count.", "memory_basis": "unknown"}
    if measured is not None:
        advice = resource_advice({"inventory": inventory, "history": []}, job, backend, threads,
                                 reserve_bytes=reserve_bytes, measured_memory=measured,
                                 run_limit=max(0, int(runs_in_scope)))
    available = (inventory.get("memory") or {}).get("available_bytes")
    result = dict(advice, threads=threads, backend=backend, reserve_bytes=reserve_bytes,
                  available_bytes=available, memory_per_worker_bytes=None if memory is None else int(memory))
    if advice.get("workers") == 0 and runs_in_scope > 0 and memory is not None and available is not None:
        result["message"] = (f"Not even one worker fits: {available / GIB:.1f} GiB available minus a "
                             f"{reserve_bytes / GIB:g} GiB reserve, against {int(memory) / GIB:.1f} GiB measured "
                             "per worker on the reference machine. Nothing starts unless you choose a worker count.")
    return result
