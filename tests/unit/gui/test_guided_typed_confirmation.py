"""Real typed confirmations with finite GUI selection and display seams only.

The original tiny both_gap_case owns every scientific source guard. The host
isolates GUI selection; the display adapter adds only native image-role/grid
metadata to the ORIGINAL verified photo/gap packet. No approval is fabricated,
and collector dispatch is captured without running the production replay.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import threading

from darsia.fluidflower import layer_painting as painting
from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower import painted_gap_regions as regions
from darsia.fluidflower.painting_guided_review_bindings import guided_gap_dialog, dispatch_findings
from darsia.fluidflower.painting_guided_review_ui import GuidedGapReviewDialog

from .test_painting_gap_review_ui import _controller
from .test_painting_guided_review_ui import GuidedBackend, _bound_review, _settle, _stage
from ..test_painted_gap_regions_integration import (
    _fill, both_gap_case, gap_case, imported, registered_setup, synthetic_payload_admission,
)


def test_opening_owned_guided_review_dispatches_initial_findings_once_without_human_choices(qtbot, monkeypatch):
    host, _unused_backend, controller = _controller(qtbot, monkeypatch)
    backend, requests = GuidedBackend(), []
    def factory(source, **options):
        return GuidedGapReviewDialog(source, review_reader=backend.read,
            preview_reader=backend.preview, comparison_reader=backend.comparison, **options)
    controller._gap_dialog_factory = factory
    controller.catalogue_refresher = requests.append  # Finite dispatcher DATA, no scientific replay claim.
    widget = controller.open_gap_review(_bound_review())
    assert widget is not None
    _settle(qtbot, widget)
    qtbot.waitUntil(lambda: bool(requests), timeout=5000)
    for _ in range(3):
        widget._buttons()  # Ordinary readiness notifications never duplicate the job.
    assert len(requests) == 1 and host.calls == []
    assert requests[0]['input_bindings']['review'] == widget.review['review_binding']
    assert requests[0]['input_bindings']['analysis_exclusions'] is None
    assert requests[0]['input_bindings']['source_choices'] == []
    assert widget.approval is None and controller.preparation is controller.dialog is None
    assert widget.operation_waiting and controller.findings_block_reason()
    widget.operation_waiting = False  # Captured dispatcher did not start a collector.


def test_initial_real_collector_keeps_ownership_until_result_then_runs_pending_preview(
        qtbot, monkeypatch, both_gap_case):
    """Reproduce readiness dispatch INSIDE original review retirement.

    The finite wait wraps and delegates the actual tiny-source collector;
    original source, recipe, diagnostic and result-admission guards remain.
    No human action, preparation, candidate or successful Check is invented.
    """
    from darsia.fluidflower import painted_transfer_findings

    case = both_gap_case
    host, _unused_backend, controller = _controller(qtbot, monkeypatch)
    host.project = case.data.project
    host.record = deepcopy(case.review["map_version"])
    # A preparation's outward version summary omits the GUI selection's run.
    # Use its original frozen painting reference, not an inferred/default run.
    host.record["run_id"] = case.review["source"]["reference_run"]
    host.approval = deepcopy(host.record["approval"])
    geometry = case.review["geometry_index"]
    host.selection = replace(host.selection, index_path=geometry["path"],
        sha256=geometry["sha256"], size_bytes=geometry["size_bytes"])
    # This original native fixture declares no comparison/display context, so
    # the finite GUI submits no display preferences. No source reader or
    # scientific validator is replaced; the actual collector still runs.
    monkeypatch.setattr(controller, "_display_preferences", lambda: {})
    entered, release = threading.Event(), threading.Event()
    requests, previews, collector_lifetimes = [], [], []
    actual_collect = painted_transfer_findings.collect_transfer_findings
    def finite_collect(*args, **kwargs):
        entered.set()
        assert release.wait(5), "The original collector's finite wait must be released"
        return actual_collect(*args, **kwargs)
    monkeypatch.setattr(painted_transfer_findings, "collect_transfer_findings", finite_collect)

    def display_preview(review, run_id, component_id=None):
        previews.append((run_id, component_id))
        packet = gaps.read_gap_preview(review, run_id, component_id)
        packet.update(image_role="current", display_grid=dict(
            format="synthetic-native-display-only", shape_hw=list(packet["gap_mask"].shape),
            native_to_photo_h=packet["native_to_photo_h"]))
        return packet
    def factory(source, **options):
        return guided_gap_dialog(source, review_reader=gaps.read_gap_review,
            preview_reader=display_preview, **options)
    def dispatch(request):
        requests.append(deepcopy(request))
        dispatch_findings(controller, request)
        lifetime = dict(thread=controller.gap_dialog.thread, destroyed=False)
        lifetime["thread"].destroyed.connect(lambda *_args: lifetime.update(destroyed=True))
        collector_lifetimes.append(lifetime)
    controller._gap_dialog_factory = factory
    controller.catalogue_refresher = dispatch
    widget = controller.open_gap_review(case.review)
    assert widget is not None, host.use_map_progress.text()
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        collector = collector_lifetimes[0]["thread"]
        assert len(requests) == 1 and widget.thread is collector
        assert widget.worker.key == ("findings-catalogue", requests[0]["generation"])
        assert collector.isRunning() and widget.operation_waiting
        assert widget.packet is None and previews == []
        pending = widget._pending
        assert pending == widget._preview_key and pending is not None
        widget._load(pending)  # Another request stays queued behind this same owner.
        assert widget.thread is collector and widget._pending == pending and previews == []
        release.set()
        qtbot.waitUntil(lambda: not widget.busy() and widget.packet is not None
            and controller._catalogue_request is None, timeout=15000)
        # The authenticated collector may select a larger group representative.
        # The queued read must follow that latest visible question, once.
        expected = max(((component["pixels"], (row["run_id"], component["component_id"]))
                        for row in case.review["runs"] for component in row["components"]),
                       key=lambda item: item[0])[1]
        assert expected != pending  # This fixture must exercise a changed representative.
        assert widget._preview_key == expected
        assert previews == [expected] and len(requests) == 1
        assert widget.packet["run_id"] == expected[0] and widget.packet["component_id"] == expected[1]
        assert not widget.operation_waiting and not widget._catalogue_refresh_pending
        assert controller._findings_update_reason == "" and widget._pending is None
        assert widget._report is not None and widget._candidate is None
        assert widget.approval is None and controller.preparation is controller.dialog is None
        assert host.calls == []
    finally:
        release.set()
        # Also join an overwritten collector on the OLD broken implementation,
        # so this regression fails its assertion without leaking a native thread.
        # Destroyed threads already passed the original wait/deleteLater barrier.
        for lifetime in collector_lifetimes:
            if not lifetime["destroyed"]:
                lifetime["thread"].wait()
        widget.wait_for_workers()


def test_two_human_confirms_use_original_workers_and_distinct_publication_namespaces(
        qtbot, monkeypatch, both_gap_case):
    case = both_gap_case
    host, _unused_backend, controller = _controller(qtbot, monkeypatch)
    # These are the existing finite host's loaded selection fields, not a
    # replacement for any scientific reader, source binding or C validator.
    host.project = case.data.project
    host.record = deepcopy(case.review["map_version"])
    host.approval = deepcopy(host.record["approval"])
    controller.context = controller._findings_context_snapshot()
    controller._findings_context = controller.context
    controller._findings_review_binding = deepcopy(case.review["review_binding"])
    dispatches, requests, accepted = [], [], []
    controller.catalogue_refresher = dispatches.append

    def display_preview(review, run_id, component_id=None):
        packet = gaps.read_gap_preview(review, run_id, component_id)
        packet.update(image_role="current", display_grid=dict(
            format="synthetic-native-display-only",
            shape_hw=list(packet["gap_mask"].shape),
            native_to_photo_h=packet["native_to_photo_h"]))
        return packet

    widget = guided_gap_dialog(case.review, actor=host._reviewer(), parent=host,
        review_reader=gaps.read_gap_review, review_validator=gaps.read_gap_review,
        preview_reader=display_preview)
    controller.gap_dialog = widget
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    widget.region_confirmation_requested.connect(requests.append)
    widget.region_confirmation_requested.connect(controller._confirm_gap_regions)
    widget.approval_changed.connect(controller._areas_changed)
    widget.approval_changed.connect(lambda value: accepted.append(deepcopy(value)))
    widget.draft_choices_changed.connect(
        lambda payload: controller.set_findings_draft_pending(widget, payload))
    widget.resize(1100, 850)
    widget.show()
    _settle(qtbot, widget)
    assert widget.approval is None and accepted == []

    original_review = Path(case.review["review_binding"]["path"]).read_bytes()
    for index in range(2):
        _stage(qtbot, widget, "deviation", index)
        draft = widget._deviation_draft
        assert isinstance(draft, regions.GapRegionDraft)
        _fill(draft, "air")  # Original native polygon kernel, clipped by its real gap.
        widget.reason.setText(f"Explicit synthetic inspection of area {index + 1}")
        widget.deviation_editor._buttons()
        assert len(accepted) == index and len(requests) == index
        assert widget.deviation_editor.continue_button.isEnabled()
        widget.deviation_editor.continue_button.click()
        qtbot.waitUntil(lambda: len(accepted) == index + 1 and not widget.busy(), timeout=10000)
        _settle(qtbot, widget)
        current = accepted[-1]
        assert current == widget.approval == controller._area_approval
        assert current["format"] == gaps.APPROVAL_FORMAT_V2
        assert len(current["runs"]) == index + 1
        if index:
            assert requests[-1]["previous"] == accepted[0]
            assert current["runs"][0] == accepted[0]["runs"][0]
        else:
            assert requests[-1]["previous"] is None
        regions.validate_gap_region_approval(current, case.review)

    parent = painting.painting_folder(case.data.project).parent / "gap_region_approvals"
    publications = [Path(action["partition"]["artifact"]["path"]).parent
                    for action in accepted[-1]["runs"]]
    assert len(set(publications)) == 2
    assert all(folder.parent == parent and (folder / "approval.json").is_file()
               for folder in publications)
    assert Path(case.review["review_binding"]["path"]).read_bytes() == original_review
    assert host.calls == [] and controller.preparation is None and controller.dialog is None
    # Capture the ordinary dispatcher request only; never collect or install a
    # successful findings result, candidate, proof, preparation or adoption.
    assert controller.refresh_findings()
    assert dispatches and dispatches[-1]["input_bindings"]["analysis_exclusions"] == accepted[-1]
    assert not any((folder / "index.json").exists() for folder in publications)
    widget.operation_waiting = False  # No collector was started by this capture seam.
