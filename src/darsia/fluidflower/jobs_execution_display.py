"""Display existing job observations without inferring worker liveness."""


def execution_report_text(snapshot, local_machine, *, compact=False):
    """Describe reported execution, never the selected or registered target.

    The original snapshot reader owns the evidence. A shared job folder, master,
    registration or launch request alone does not establish where work ran.
    Worker rows may be historical, so this label makes no liveness claim.
    """
    if not snapshot:
        return "No job selected."
    if snapshot.get("errors"):
        return "Execution location unavailable: job metadata has errors."
    machines = {}
    unknown = False
    reports = list(snapshot.get("workers") or []) + [
        row for row in snapshot.get("machines") or [] if row.get("committed", 0)
    ] + [
        row for row in snapshot.get("runs") or [] if row.get("state") in {"Complete", "Completed", "Failed"}
    ]
    for report in reports:
        machine = str(report.get("machine") or "").strip()
        if not machine or machine.casefold() in {"not reported", "not recorded", "unknown"}:
            unknown = True
        else:
            machines.setdefault(machine.casefold(), machine)
    if not machines:
        return (
            "Execution location not reported."
            if unknown else "No worker execution reported yet."
        )
    local_key = str(local_machine).strip().casefold()
    remote = sorted(value for key, value in machines.items() if key != local_key)
    local = machines.get(local_key)
    if remote:
        targets = ([f"this computer ({local})"] if local else []) + remote
        text = "Shared compute reported: " + ", ".join(targets)
    else:
        text = f"This computer reported: {local}"
    if unknown:
        text += "; another report has no machine name"
    return text if compact else text + ". Liveness is not established by these reports."


def selected_target_text(target, workers, backend):
    """Explain the next explicit start independently of recorded execution."""
    if not target:
        return "Choose the machine for the next worker start."
    machine = target["machine"]
    location = (
        f"this computer ({machine})" if target.get("local")
        else f"shared compute on {machine}"
    )
    return (
        f"Next start: {location} · {workers} worker(s) · {str(backend).upper()}. "
        "This count applies to the selected machine. Other machines keep their existing workers."
    )
