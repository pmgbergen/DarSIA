"""New analysis (N141, spec 3.1) replaces the Start analysis dialog.

Real offscreen Qt with a local job registry, a fake producer and fake plan
facts; no plan is adopted, no job is prepared and no worker starts.
"""

from pathlib import Path
from threading import Event

import pytest
from PySide6.QtWidgets import QScrollArea

from darsia.fluidflower import jobs_panel as jobs
from darsia.fluidflower import native_jobs_workspace as ui

from .test_native_jobs_owner import job, local_registry, plan, series, snapshot  # noqa: F401

FACTS = dict(runs=["ac52", "ac53"], frames=78, precision="float32", storage_mm=[None, None])


def test_positive_phase1_facts_force_full_and_disable_create_for_the_actual_route_blocker(qtbot):
    panel = jobs.JobsPanel()
    qtbot.addWidget(panel)
    page = jobs.NewAnalysisPage(panel)
    qtbot.addWidget(page)
    page.facts = dict(runs=["run-one"], frames=2, precision="float32", storage_mm=[None],
        pipeline="native-phase-statistics", gas_probability=True, running_pressure_runs=["run-one"],
        route_blocker="The selected profile has another commit")
    page._show_facts()
    assert page.form.profile_combo.currentData() == "full"
    assert not page.form.profile_combo.isEnabled()
    assert not page.create_button.isEnabled()
    assert "another commit" in page.create_button.toolTip()
    assert "statistics" in page.pipeline_label.text() and "float32" in page.probability_label.text()
    assert "1 runs" in page.pressure_label.text()


@pytest.fixture
def owner(local_registry, qtbot):  # noqa: F811
    def forbidden(*a, **k):
        pytest.fail("No real producer, worker or probe may run here")

    value = ui.NativeJobsWorkspace(
        registry=local_registry,
        local_machine="Main",
        reader=snapshot,
        job_creator=forbidden,
        local_launcher=forbidden,
        remote_launcher=forbidden,
        inventory_provider=lambda: {},
        resource_factory=jobs_resource_factory(),
    )
    value.new_analysis.facts_reader = lambda row: dict(FACTS)
    qtbot.addWidget(value, before_close_func=lambda widget: qtbot.waitUntil(widget.can_close, timeout=30000))
    yield value
    qtbot.waitUntil(value.can_close, timeout=30000)


def jobs_resource_factory():
    from darsia.fluidflower.resource_advisor_panel import ResourceAdvisorPanel

    return ResourceAdvisorPanel


def _ancestors(widget):
    parent = widget.parentWidget()
    while parent is not None:
        yield parent
        parent = parent.parentWidget()


def _registered_plan(owner, tmp_path):
    root = series(tmp_path / "s", "literal")
    path = plan(tmp_path / "plan.toml")
    owner.registry.register_plan(root, "literal", path)
    owner.set_series("literal", root)
    return root


def test_the_page_offers_the_dialogs_profile_choice_estimate_and_choices(qtbot):
    plan_row = dict(path=r"\\server\share\fixed.toml", sha256="a" * 64, size_bytes=15, label="Frozen")
    free = dict(free_bytes=1_234_000_000_000, series_id="unit")
    panel = jobs.JobsPanel(local_machine="Main")
    qtbot.addWidget(panel)
    page = jobs.NewAnalysisPage(panel, facts_reader=lambda row: dict(FACTS))
    qtbot.addWidget(page)
    dialog = jobs.StartAnalysisDialog([plan_row], free_space=free)
    qtbot.addWidget(dialog)
    page.set_plans([plan_row], free_space=free)
    qtbot.waitUntil(lambda: page.facts is not None and page.can_close(), timeout=30000)
    for form in (page.form, dialog):
        combo = form.profile_combo
        assert [combo.itemData(i) for i in range(combo.count())] == ["full", "lean"]
        assert combo.currentData() == jobs.DEFAULT_STORAGE_PROFILE
    labels = [page.form.profile_combo.itemText(i) for i in range(2)]
    assert labels == [dialog.profile_combo.itemText(i) for i in range(2)]
    for profile in ("full", "lean"):
        for combo in (page.form.profile_combo, dialog.profile_combo):
            combo.setCurrentIndex(combo.findData(profile))
        assert page.form.estimate_label.text() == dialog.estimate_label.text()
        assert page.form.estimate_label.text() == jobs.choice_estimate_text(profile, free)
    for form in (page.form, dialog):
        form.job_edit.setText(r"\\server\share\jobs\new")
        form.name_edit.setText("Name")
    assert page.form.choices() == dialog.choices()
    assert page.form.choices()["storage_profile"] == "lean"
    assert "1.1 GB" in jobs.LEAN_NOTE and "13.6 GB" in jobs.LEAN_NOTE
    qtbot.waitUntil(lambda: page.can_close() and panel.can_close(), timeout=30000)


def test_dialog_and_page_are_freed_by_their_last_reference(qtbot):
    """Q160: no reference cycle may keep a parentless dialog or page alive.

    Such a widget is otherwise left to the cyclic garbage collector, which can
    run in a Jobs worker thread and destroy it there; that hung a full suite.
    """
    import gc
    import weakref

    plan_row = dict(path=r"\\server\share\fixed.toml", sha256="a" * 64, size_bytes=15, label="Frozen")
    panel = jobs.JobsPanel(local_machine="Main")
    qtbot.addWidget(panel)
    enabled = gc.isenabled()
    gc.disable()  # Only reference counting may free them here.
    try:
        dialog = jobs.StartAnalysisDialog([plan_row])
        page = jobs.NewAnalysisPage(panel, facts_reader=lambda row: dict(FACTS))
        refs = [weakref.ref(item) for item in (dialog, dialog.form, page, page.form)]
        del dialog, page
        assert [ref() is None for ref in refs] == [True, True, True, True]
    finally:
        if enabled:
            gc.enable()
    assert panel.can_close()


def test_computed_at_is_native_with_the_coarse_grids_disabled(qtbot):
    combo = jobs.computing_resolution_combo()
    qtbot.addWidget(combo)
    assert [combo.itemData(i) for i in range(combo.count())] == [None, 0.5, 1.0]
    assert combo.currentIndex() == 0 and combo.currentData() is None
    assert combo.model().item(0).isEnabled()
    for index in (1, 2):
        assert not combo.model().item(index).isEnabled()
        assert jobs.NOT_AVAILABLE_YET in combo.itemText(index)


def test_stored_is_never_finer_than_computed():
    assert jobs.stored_not_finer(None, None) and jobs.stored_not_finer(0.25, None)
    assert jobs.stored_not_finer(1.0, 1.0) and jobs.stored_not_finer(2.0, 0.5)
    assert not jobs.stored_not_finer(0.5, 1.0)
    assert not jobs.stored_not_finer(None, 0.5)
    assert jobs.stored_at_text([None, None]) == "native, bound in the adopted plan for all 2 runs (read-only)"
    assert jobs.stored_at_text([1.0]) == "1 mm, bound in the adopted plan for all 1 runs (read-only)"
    assert jobs.stored_at_text([None, 0.5]).startswith("Differs per run: native, 0.5 mm")


def test_stored_choices_are_never_finer_than_computed():
    assert jobs.stored_resolution_choices(None) == [None, 0.25, 0.5, 1.0, 2.0]
    assert jobs.stored_resolution_choices(0.5) == [0.5, 1.0, 2.0]
    assert jobs.stored_resolution_choices(1.0) == [1.0, 2.0]


def test_stored_at_lists_allowed_sizes_with_the_plans_size_read_only(owner, tmp_path, qtbot):
    _registered_plan(owner, tmp_path)
    owner.show_view("new_analysis")
    page = owner.new_analysis
    qtbot.waitUntil(lambda: page.facts is not None, timeout=30000)
    combo = page.stored_combo
    assert not combo.isEnabled()
    assert [combo.itemText(i) for i in range(combo.count())] == ["native", "0.25 mm", "0.5 mm", "1 mm", "2 mm"]
    assert combo.currentText() == "native"
    page.computed_combo.setCurrentIndex(page.computed_combo.findData(1.0))
    assert [combo.itemText(i) for i in range(combo.count())] == ["1 mm", "2 mm"]
    assert combo.currentIndex() == -1 and not page.stored_allowed()
    page.facts = dict(FACTS, storage_mm=[2.0, 2.0])
    page._show_facts()
    assert combo.currentText() == "2 mm" and page.stored_allowed()
    page.computed_combo.setCurrentIndex(0)
    assert combo.currentText() == "2 mm"


def test_read_plan_facts_uses_the_pipeline_reader_and_names_each_runs_storage():
    calls = []

    def reader(path, *, expected_sha256):
        calls.append((path, expected_sha256))
        return (
            dict(job=dict(precision="float32")),
            [dict(run="ac52", frames=[1, 2]), dict(run="ac53", frames=[3], storage=dict(resolution_mm=1.0)),
             dict(run="ac54", frames=[], storage=dict(resolution_mm="native"))],
            dict(path=str(path)),
        )

    facts = jobs.read_plan_facts(dict(path="plan.toml", sha256="b" * 64), reader=reader)
    assert calls == [(Path("plan.toml"), "b" * 64)]
    assert facts == dict(runs=["ac52", "ac53", "ac54"], frames=3, precision="float32",
                         storage_mm=[None, 1.0, None])


def test_page_shows_the_plans_facts_and_blocks_create_only_if_stored_is_finer(owner, tmp_path, qtbot):
    _registered_plan(owner, tmp_path)
    owner.show_view("new_analysis")
    page = owner.new_analysis
    assert owner.views.currentWidget() is page
    qtbot.waitUntil(lambda: page.facts is not None, timeout=30000)
    assert page.selection_label.text() == "2 runs, 78 images, fixed by the adopted plan"
    assert page.stored_label.text().startswith("native, bound in the adopted plan")
    assert page.precision_label.text().startswith("float32")
    page.form.job_edit.setText(str(tmp_path / "fresh-job"))
    assert page.create_button.isEnabled()
    page.facts = dict(FACTS, storage_mm=[0.5, 0.5])
    page.computed_combo.setCurrentIndex(page.computed_combo.findData(1.0))
    assert not page.stored_allowed() and not page.create_button.isEnabled()
    assert "not allowed" in page.stored_label.text()
    page.computed_combo.setCurrentIndex(0)
    assert page.stored_allowed() and page.create_button.isEnabled()


def test_create_job_files_then_start_with_the_lent_machine_controls(owner, tmp_path, qtbot):
    _registered_plan(owner, tmp_path)
    panel, page = owner.panel, owner.new_analysis
    created, started = [], []

    def creator(plan_row, root, **kwargs):
        created.append(kwargs)
        job(Path(root))
        return dict(root=str(root), name="Created", series_label="literal", active=True, worker_started=False)

    owner._creator = creator
    panel.start_selected_machine = lambda: started.append(panel.current["root"])
    owner.show_view("new_analysis")
    qtbot.waitUntil(lambda: page.facts is not None and page.can_close(), timeout=30000)
    # The job toolbar lends the same machine, backend and workers objects.
    for widget in (panel.target_combo, panel.backend_combo, panel.workers_spin):
        assert page.target_layout.indexOf(widget) >= 0
    assert page.machine_box.isAncestorOf(panel.resource_panel)
    assert not page.start_button.isEnabled()
    page.form.profile_combo.setCurrentIndex(page.form.profile_combo.findData("lean"))
    page.form.job_edit.setText(str(tmp_path / "fresh-job"))
    page.create_button.click()
    qtbot.waitUntil(lambda: page.created_root is not None, timeout=30000)
    qtbot.waitUntil(lambda: page.start_button.isEnabled(), timeout=30000)
    assert created[0]["storage_profile"] == "lean"
    assert Path(page.created_root) == tmp_path / "fresh-job"
    assert page.machine_box.isEnabled()
    page.start_button.click()
    assert [Path(root) for root in started] == [tmp_path / "fresh-job"]
    owner.show_view("jobs")
    for index, widget in enumerate((panel.target_combo, panel.backend_combo, panel.workers_spin)):
        assert panel._controls.indexOf(widget) == index
    assert panel.resource_slot.indexOf(panel.resource_panel) >= 0


def test_start_analysis_asks_for_the_page_instead_of_a_dialog(owner, tmp_path, qtbot, monkeypatch):
    _registered_plan(owner, tmp_path)
    monkeypatch.setattr(jobs, "StartAnalysisDialog", lambda *a, **k: pytest.fail("No dialog in the app"))
    requests = []
    owner.new_analysis_requested.connect(lambda: requests.append("new analysis"))
    owner.panel.create_button.click()
    owner.start_analysis()
    assert requests == ["new analysis", "new analysis"]


def test_no_scroll_area_inside_another_on_either_view(owner, tmp_path, qtbot):
    _registered_plan(owner, tmp_path)
    for view in ("new_analysis", "jobs", "new_analysis"):
        owner.show_view(view)
        for area in owner.findChildren(QScrollArea):
            assert not any(isinstance(parent, QScrollArea) for parent in _ancestors(area)), (view, area)
    with pytest.raises(ValueError):
        owner.show_view("elsewhere")


@pytest.mark.parametrize("outcome", ["ready", "error"])
def test_create_waits_for_selected_plan_facts_and_explains_a_failed_read(owner, tmp_path, qtbot, monkeypatch, outcome):
    """Clicking Create during a blocked or failed read cannot call its producer."""
    page, started, release, created = owner.new_analysis, Event(), Event(), []

    def read(_row):
        started.set()
        if not release.wait(10):
            raise RuntimeError("The private plan read was not released")
        if outcome == "error":
            raise OSError("The selected plan location is unavailable")
        return dict(FACTS)

    page.facts_reader = read
    monkeypatch.setattr(owner.panel, "create_job", lambda **kwargs: created.append(kwargs))
    _registered_plan(owner, tmp_path)
    owner.show_view("new_analysis")
    try:
        qtbot.waitUntil(started.is_set, timeout=10000)
        page.form.job_edit.setText(str(tmp_path / "fresh-job"))
        assert not page.create_button.isEnabled()
        assert "selected execution plan is read" in page.create_button.toolTip()
        assert "Reading the adopted plan" in page.selection_label.text()
        page.create_button.click()
        assert not created
    finally:
        release.set()
    qtbot.waitUntil(page.can_close, timeout=10000)
    if outcome == "ready":
        assert page.facts == FACTS and page.create_button.isEnabled()
        page.create_button.click()
        assert len(created) == 1 and created[0]["plan"] == page.form.plan_combo.currentData()
    else:
        assert page.facts is None and not page.create_button.isEnabled()
        assert "selected plan location is unavailable" in page.create_button.toolTip()
        assert "Not read: OSError" in page.selection_label.text()
        page.create_button.click()
        assert not created


def test_previous_plan_completion_cannot_enable_create_for_the_new_selection(owner, tmp_path, qtbot):
    """Retain one background reader while its old selection becomes stale."""
    page = owner.new_analysis
    first_started, second_started, release_first, release_second = (Event() for _ in range(4))
    second = dict(path=str(tmp_path / "another-plan.toml"), sha256="b" * 64,
                  size_bytes=11, label="Another saved plan")
    chosen_facts = dict(FACTS, runs=["chosen-run"], frames=3, storage_mm=[None])

    def read(row):
        if row["path"] == second["path"]:
            second_started.set()
            if not release_second.wait(10):
                raise RuntimeError("The selected private plan read was not released")
            return chosen_facts
        first_started.set()
        if not release_first.wait(10):
            raise RuntimeError("The former private plan read was not released")
        return dict(FACTS)

    page.facts_reader = read
    _registered_plan(owner, tmp_path)
    owner.show_view("new_analysis")
    try:
        qtbot.waitUntil(first_started.is_set, timeout=10000)
        original = page.form.plan_combo.currentData()
        page.set_plans([original, second])
        page.form.plan_combo.setCurrentIndex(1)
        page.form.job_edit.setText(str(tmp_path / "fresh-job"))
        assert not page.create_button.isEnabled()
        release_first.set()
        qtbot.waitUntil(second_started.is_set, timeout=10000)
        assert page.facts is None and not page.create_button.isEnabled()
        assert "selected execution plan is read" in page.create_button.toolTip()
        assert page.form.plan_combo.currentData() == second
    finally:
        release_first.set()
        release_second.set()
    qtbot.waitUntil(lambda: page.can_close() and page.facts is not None, timeout=10000)
    assert page.facts == chosen_facts and page.create_button.isEnabled()
    assert page.selection_label.text() == "1 runs, 3 images, fixed by the adopted plan"


def test_job_folder_is_named_in_the_series_shared_jobs_folder(qtbot, tmp_path, monkeypatch):
    import re
    from darsia.fluidflower import series_paths
    panel = jobs.JobsPanel()
    qtbot.addWidget(panel)
    page = jobs.NewAnalysisPage(panel)
    qtbot.addWidget(page)
    edit = page.form.job_edit
    assert edit.isReadOnly() and not edit.text()
    folders = {"shared": Path("//server/share/series/jobs"), "local": tmp_path / "jobs"}
    chosen = ["shared"]
    monkeypatch.setattr(series_paths, "series_folder_path", lambda root, key: folders[chosen[0]])
    panel.series_root = tmp_path
    page.suggest_job_folder()
    first = edit.text()
    parent, name = first.rsplit("\\", 1)
    assert parent == str(folders["shared"]) and re.fullmatch("production-[0-9a-f]{32}", name)
    page.suggest_job_folder()
    assert edit.text() == first  # The shown fresh folder is kept.
    page.suggest_job_folder(fresh=True)
    assert edit.text() != first and edit.text().startswith(str(folders["shared"]))
    chosen[0] = "local"  # No shared folder: nothing is invented.
    page.suggest_job_folder()
    assert not edit.text() and "Jobs > Shared job folder" in edit.placeholderText()
