"""Location copy uses reported evidence, never a pending launch or UI choice."""

from copy import deepcopy

from darsia.fluidflower.jobs_execution_display import (
    execution_report_text,
    selected_target_text,
)


def test_registration_master_and_launch_do_not_claim_execution():
    snapshot = {
        "root": r"\\shared\jobs\one",
        "master_machine": "Main",
        "registered_machines": [{"machine": "Remote"}],
        "launch_requests": [{"machine": "Remote", "state": "requested"}],
    }
    before = deepcopy(snapshot)
    assert execution_report_text(snapshot, "Main") == "No worker execution reported yet."
    assert snapshot == before


def test_local_worker_reports_keep_the_liveness_qualification():
    snapshot = {"workers": [{"machine": "MAIN", "state": "Idle"}]}
    text = execution_report_text(snapshot, "Main")
    assert "This computer reported: MAIN" in text
    assert "Liveness is not established" in text


def test_shared_execution_combines_worker_and_completed_frame_locations():
    snapshot = {
        "workers": [{"machine": "Main"}, {"machine": "Remote"}],
        "machines": [{"machine": "Archive", "committed": 4},
                     {"machine": "Never ran", "committed": 0}],
    }
    text = execution_report_text(snapshot, "main")
    assert text.startswith("Shared compute reported: this computer (Main), Archive, Remote.")
    assert "Never ran" not in text


def test_missing_and_unreadable_evidence_never_becomes_a_location():
    assert execution_report_text(None, "Main") == "No job selected."
    assert execution_report_text({"workers": [{"machine": "Not reported"}]}, "Main") == (
        "Execution location not reported."
    )
    assert "unavailable" in execution_report_text(
        {"errors": ["unreadable"], "workers": [{"machine": "Main"}]}, "Main"
    )
    assert "another report has no machine name" in execution_report_text(
        {"workers": [{"machine": "Main"}, {}]}, "Main"
    )


def test_terminal_setup_and_failed_runs_report_machine_without_image_progress():
    snapshot = {
        "workers": [], "machines": [],
        "runs": [{"state": "Complete", "machine": "Remote", "completed_operations": 1},
                 {"state": "Failed", "machine": "Main", "committed": 0},
                 {"state": "Queued", "machine": "Planned target"}],
    }
    text = execution_report_text(snapshot, "Main")
    assert "Shared compute reported: this computer (Main), Remote" in text
    assert "Planned target" not in text


def test_next_start_names_machine_count_and_backend_without_claiming_execution():
    local = selected_target_text({"machine": "Main", "local": True}, 2, "cpu")
    remote = selected_target_text({"machine": "Remote", "local": False}, 3, "cuda")
    assert "Next start: this computer (Main) · 2 worker(s) · CPU" in local
    assert "Next start: shared compute on Remote · 3 worker(s) · CUDA" in remote
    assert "selected machine" in remote
    assert "Choose the machine" in selected_target_text(None, 1, "cpu")
