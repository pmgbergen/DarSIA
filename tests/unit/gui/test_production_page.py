"""Production page uses the actual Jobs controls; launchers never run here."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from darsia.fluidflower import jobs_panel
from darsia.fluidflower.production_page import ProductionPage
from ..test_production_local_facts import local_production_plan, phase1_data  # noqa: F401


def calibration():
    return dict(name="Morning", chemistry_model="titration", chemistry_label="Titration",
                calibration=dict(path="frozen.json", sha256="a" * 64, size_bytes=25),
                job_root="calibration-job", series_id="series-one")


@pytest.fixture
def page(qtbot):
    panel = jobs_panel.JobsPanel(local_machine="Main")
    qtbot.addWidget(panel)
    value = ProductionPage(panel)
    panel.attach_new_analysis(value)
    qtbot.addWidget(value)
    panel.series_root, panel.series_id = "series", "series-one"
    panel.create_button.setEnabled(True)
    # This UI unit test injects already-read metadata only. Real plan and series
    # identity refusal are covered by the production selection tests.
    value.form.plan_combo.blockSignals(True)
    value.form.plan_combo.addItem("Plan", dict(path="plan.toml", sha256="b" * 64, size_bytes=12))
    value.form.plan_combo.blockSignals(False)
    value._facts_key = ("plan.toml", "b" * 64)
    value.facts = dict(pipeline="n1", runs=["r1"], frames=3, storage_mm=[None], precision="float32",
                       calibration=calibration()["calibration"], route_blocker="")
    value.form.job_edit.setText("fresh")
    value._show_facts()
    yield value
    assert value.can_close() and panel.can_close()


def select(page):
    page._calibration_ready(dict(generation=page._calibration_generation,
                                series=("series", "series-one"), choice=calibration()))


def test_selecting_exact_calibration_retains_full_storage_and_locked_selection(page):
    assert not page.create_button.isEnabled()
    select(page)
    assert page.create_button.isEnabled()
    assert "Morning" in page.calibration_combo.currentText()
    assert "Titration" in page.calibration_combo.currentText()
    assert "blacklists" in page.selection_label.text()
    assert "Float64 requires" in page.precision_label.text()
    assert page.form.profile_combo.currentData() == "full"
    assert page.form.profile_combo.isEnabled()
    assert "qualified provider" in page.form.profile_combo.toolTip()
    page.facts["calibration"] = dict(calibration()["calibration"], sha256="c" * 64)
    page.sync_actions()
    assert not page.create_button.isEnabled()
    assert "different mass calibration" in page.calibration_note.text()


def test_create_passes_exact_choice_without_mutating_registered_plan(page):
    select(page)
    calls = []
    original = deepcopy(page.form.plan_combo.currentData())
    page.panel.create_job = lambda **kwargs: calls.append(kwargs)
    page.panel.local_stores_path_provider = lambda: "chosen-stores.json"
    page.create()
    assert len(calls) == 1
    assert calls[0]["plan"] == dict(original, production_calibration=calibration(),
                                    production_stores_path="chosen-stores.json")
    assert calls[0]["storage_profile"] == "full"
    assert page.form.plan_combo.currentData() == original


def test_local_and_shared_use_one_original_launch_with_chosen_workers(page):
    select(page)
    page._creation_choice = page._current_choice()
    page.created_root = "fresh"
    page.panel.current = dict(root="fresh")
    page.panel.workers_spin.setValue(3)
    # This fixture supplies admission explicitly, after the worker choice has
    # triggered the original Jobs refresh against its synthetic metadata.
    page.panel.start_button.setEnabled(True)
    starts = []
    def start():
        starts.append((page.panel.target_combo.currentData(), page.panel.workers_spin.value()))
        page.panel.local_worker_launch_requested.emit({})
    page.panel.start_selected_machine = start
    page.sync_actions()
    assert page.start_button.isEnabled(), (page.start_button.toolTip(),
        page.panel.target_combo.currentData(), page._creation_choice, page._current_choice())
    page.start()
    assert starts[0][0]["local"] is True and starts[0][1] == 3
    # Target changes remain guarded by Jobs, and a remote target cannot be
    # launched under the local choice even when the Jobs Start is enabled.
    page.panel.target_combo.blockSignals(True)
    page.panel.target_combo.addItem("Other machine", dict(key="remote", machine="Other", local=False))
    page.panel.target_combo.setCurrentIndex(page.panel.target_combo.count() - 1)
    page.panel.target_combo.blockSignals(False)
    page.start()
    assert len(starts) == 1
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("shared"))
    page.panel.start_button.setEnabled(True)
    page.sync_actions()
    page.start()
    assert len(starts) == 2 and starts[-1][0]["machine"] == "Other"
    assert starts[-1][1] == 3
    page.panel.start_button.setEnabled(False)
    page.panel.start_button.setToolTip("A previous launch has not retired")
    page.start()
    assert len(starts) == 2


def test_changed_production_choices_cannot_launch_the_previously_created_job(page):
    select(page)
    page._creation_choice = page._current_choice()
    page.created_root = "fresh"
    page.panel.current = dict(root="fresh")
    page.panel.start_button.setEnabled(True)
    page.form.name_edit.setText("A different job")
    assert not page.start_button.isEnabled()
    assert "differ from the created job" in page.start_button.toolTip()


def test_late_calibration_read_cannot_cross_series_and_jobs_navigation_is_passive(page):
    select(page)
    generation = page._calibration_generation
    page.reset()
    page.panel.series_id = "another-series"
    page._calibration_ready(dict(generation=generation, series=("series", "series-one"), choice=calibration()))
    assert page.calibration_combo.count() == 0
    opened = []
    page.jobs_requested.connect(lambda: opened.append(True))
    page.jobs_button.click()
    assert opened == [True]


def test_same_plan_read_from_previous_series_cannot_restore_stale_admission(page):
    old_facts = dict(page.facts, _production_context=("series", "series-one", page._production_context_generation))
    key = page._facts_key
    page.reset()
    page.panel.series_id = "another-series"
    page._facts_key = key  # Same imported plan selected in the new context.
    page._facts_ready(dict(key=key, facts=old_facts))
    assert page.facts is None
    assert not page.create_button.isEnabled()


def test_existing_choice_can_recover_after_failed_calibration_browse(page):
    select(page)
    page.calibration_combo.addItem("Another named job", dict(calibration(), name="Evening"))
    page._choice_error = "The selected job could not be read"
    page.sync_actions()
    assert not page.create_button.isEnabled()
    page.calibration_combo.setCurrentIndex(1)
    assert page._choice_error == ""
    assert page.create_button.isEnabled()


def test_initial_phase_one_preparation_does_not_require_a_mass_calibration(page):
    from darsia.fluidflower.analysis_preparation_ui import AnalysisPreparationDialog

    page.facts = None
    page.prepare_plan()
    editor = page._preparation_dialog
    assert isinstance(editor, AnalysisPreparationDialog) and not editor.isWindow()
    assert editor.stage.currentData() == "phase1" and editor._thread is None
    page.prepare_plan()
    assert page._preparation_dialog is editor


@pytest.mark.parametrize("failed", [False, True])
def test_current_facts_keep_base_request_generation_and_series_context(page, failed):
    expected = dict(page.facts, pipeline="native-phase-n1")
    callbacks = []
    def read(plan):
        if failed:
            raise ValueError("The bound plan changed")
        return expected
    def start(call, ready, closed):
        callbacks.append((call, ready, closed))
        return SimpleNamespace(wait=lambda: None), object()
    page.facts_reader = read
    page.panel._start_call = start
    page._read_facts()
    call, ready, closed = callbacks.pop()
    result = call()
    ready(result)
    closed()
    if failed:
        assert page.facts is None and "bound plan changed" in page.facts_error
    else:
        assert page.facts == expected and page.facts_error is None
    assert page.can_close()


@pytest.mark.parametrize("lean_ready", [False, True])
def test_profile_change_discards_old_admission_and_captures_new_profile(page, lean_ready):
    """The original NewAnalysisPage reader captures each requested profile."""
    select(page)
    source = deepcopy(page.facts)
    callbacks, profiles = [], []
    page.facts_reader = lambda plan: deepcopy(source)
    def inspect(plan, *, series_root, stores_path, storage_profile):
        profiles.append(storage_profile)
        return dict(ready=storage_profile == "full" or lean_ready,
                    reason="" if lean_ready else "A completed Lean provider is required",
                    products_expected_bytes=4096)
    def start(call, ready, closed):
        callbacks.append((call, ready, closed))
        return SimpleNamespace(wait=lambda: None), object()
    page._inspect_resources = inspect
    page.panel._start_call = start
    page._read_facts()
    page.form.profile_combo.setCurrentIndex(page.form.profile_combo.findData("lean"))
    assert page.facts is None and not page.create_button.isEnabled()
    call, ready, closed = callbacks.pop(0)
    ready(call())  # The old Full result must not restore Create for Lean.
    assert page.facts is None and not page.create_button.isEnabled()
    closed()
    call, ready, closed = callbacks.pop(0)
    ready(call())
    closed()
    assert profiles == ["full", "lean"]
    assert page.form.profile_combo.currentData() == "lean"
    assert page.create_button.isEnabled() is lean_ready
    if lean_ready:
        assert "4,096 bytes" in page.form.estimate_label.text()
    else:
        assert "completed Lean provider" in page.create_button.toolTip()
    assert not callbacks and page.can_close()


def test_phase_one_full_selection_clears_previous_lean_mass_capacity_text(page):
    page.form.profile_combo.blockSignals(True)
    page.form.profile_combo.setCurrentIndex(page.form.profile_combo.findData("lean"))
    page.form.profile_combo.blockSignals(False)
    page._facts_storage_profile = "lean"
    page.facts["resource_admission"] = dict(ready=True, products_expected_bytes=4096)
    page._show_facts()
    assert "4,096 bytes" in page.form.estimate_label.text()
    page.facts["pipeline"] = "native-phase-statistics"
    page._facts_storage_profile = "full"  # The subsequent Full read has completed.
    page._show_facts()
    assert page.form.profile_combo.currentData() == "full"
    assert not page.form.profile_combo.isEnabled()
    assert page.form.estimate_label.text() == jobs_panel.choice_estimate_text("full", page.form.free_space)


@pytest.fixture
def local_facts_page(qtbot, local_production_plan, monkeypatch):
    """Real default rooted Facts; only scheduling/resources are isolated.

    The shared fixture supplies original sealed metadata and local aliases.
    No facts, plan admission or catalogue reader is replaced. The inert resource
    result deliberately leaves Create unavailable; this is not provider or
    capacity qualification. Deferred calls run only during authorized tests.
    """
    from darsia.fluidflower.production_selection import read_production_facts

    source = local_production_plan
    panel = jobs_panel.JobsPanel(local_machine="Main")
    qtbot.addWidget(panel)
    value = ProductionPage(panel)
    panel.attach_new_analysis(value)
    qtbot.addWidget(value)
    panel.series_root, panel.series_id = str(source["root"]), source["series_id"]
    panel.local_stores_path_provider = lambda: "captured-stores.json"
    callbacks, resources = [], []

    def start(call, ready, closed):
        callbacks.append((call, ready, closed))
        return SimpleNamespace(wait=lambda: None), object()

    def inspect(plan, *, series_root, stores_path, storage_profile):
        resources.append(dict(plan=deepcopy(plan), series_root=series_root,
                              stores_path=stores_path, storage_profile=storage_profile))
        return dict(ready=False, reason="Resource inspection is isolated in this GUI test.")

    monkeypatch.setattr(panel, "_start_call", start)
    monkeypatch.setattr(value, "_inspect_resources", inspect)
    value.form.plan_combo.blockSignals(True)
    value.form.plan_combo.addItem("Local adopted plan", deepcopy(source["plan"]))
    value.form.plan_combo.blockSignals(False)
    assert value.facts_reader is read_production_facts
    yield SimpleNamespace(page=value, panel=panel, source=source,
                          callbacks=callbacks, resources=resources)
    assert not callbacks and value.can_close() and panel.can_close()


def test_default_production_facts_reads_actual_local_alias_plan(local_facts_page):
    from darsia.fluidflower.pipeline.job_queue import descriptor

    state = local_facts_page
    page, source = state.page, state.source
    plan_before, manifest_before = descriptor(source["plan_path"]), descriptor(source["manifest"])
    page._read_facts()
    assert page.facts is None and not page.create_button.isEnabled()
    call, ready, closed = state.callbacks.pop()
    result = call()
    assert "error" not in result and "_production_error" not in result["facts"]
    ready(result)
    closed()
    assert page.facts_error is None
    assert page.facts["pipeline"] == "n1"
    assert page.facts["runs"] == [source["config"]["run"]]
    assert page.facts["frames"] == len(source["config"]["frames"]) == 1
    assert page.facts["precision"] == "float32"
    assert page.facts["calibration"] == {key: source["config"]["calibration"][key]
                                         for key in ("path", "sha256", "size_bytes")}
    assert page.facts["route_blocker"] == "Resource inspection is isolated in this GUI test."
    assert state.resources == [dict(plan=source["plan"], series_root=str(source["root"]),
                                   stores_path="captured-stores.json", storage_profile="full")]
    assert descriptor(source["plan_path"]) == plan_before == source["plan"]
    assert descriptor(source["manifest"]) == manifest_before
    assert not page.create_button.isEnabled()


@pytest.mark.parametrize("changed", ["root", "series_id"])
def test_default_production_facts_captures_context_and_discards_changed_series(local_facts_page, changed):
    state = local_facts_page
    page, panel, source = state.page, state.panel, state.source
    page._read_facts()
    call, ready, closed = state.callbacks.pop()
    original_key, original_generation = page._facts_key, page._facts_generation
    # Change only series context while the exact same plan and base request
    # generation remain selected. Reading live panel fields would now fail.
    if changed == "root":
        panel.series_root = str(source["root"].parent / "unregistered-other-series")
    else:
        panel.series_id = "different-series"
    panel.local_stores_path_provider = lambda: "different-stores.json"
    result = call()
    assert "error" not in result and "_production_error" not in result["facts"]
    assert result["facts"]["pipeline"] == "n1"
    assert result["facts"]["_production_context"][:2] == (str(source["root"]), source["series_id"])
    assert state.resources == [dict(plan=source["plan"], series_root=str(source["root"]),
                                   stores_path="captured-stores.json", storage_profile="full")]
    assert page._facts_key == original_key and page._facts_generation == original_generation
    ready(result)
    closed()
    assert page.facts is None and page.facts_error is None
    assert not page.create_button.isEnabled() and page.can_close()
