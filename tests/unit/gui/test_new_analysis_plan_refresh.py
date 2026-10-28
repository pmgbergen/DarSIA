"""Page entry rechecks selected-plan admission without creating or launching jobs."""

from copy import deepcopy
from threading import Event

from darsia.fluidflower import jobs_panel as ui
from darsia.fluidflower import staged_jobs


class _ObservedPage(ui.NewAnalysisPage):
    """Retain actual Create states, including transient asynchronous updates."""

    def __init__(self, panel, **choices):
        """Start observing before the original page initializes its controls."""
        self.create_states = []
        super().__init__(panel, **choices)

    def sync_actions(self):
        """Record the original action state after each real synchronization."""
        super().sync_actions()
        self.create_states.append(self.create_button.isEnabled())


def _page(qtbot, tmp_path, stores):
    """Bind synthetic plan choices to the existing page-entry and call runner."""
    selected = dict(path=str(tmp_path / "selected.toml"), sha256="a" * 64,
        size_bytes=12, label="Selected plan")
    offered = [selected]
    reads = []

    def read(row):
        reads.append(deepcopy(row))
        return dict(runs=["run-one"], frames=2, precision="float32", storage_mm=[None],
            pipeline="n1", route_blocker="")

    def forbidden(*args, **choices):
        raise AssertionError("Admission display must not create jobs or start workers")

    panel = ui.JobsPanel(local_machine="fixture", local_stores_path_provider=lambda: stores["path"],
        job_creator=forbidden, local_launcher=forbidden, remote_launcher=forbidden)
    page = _ObservedPage(panel, parent=panel, facts_reader=read)
    panel.attach_new_analysis(page)
    qtbot.addWidget(panel, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    panel.set_context(series_root=tmp_path / "series", series_id="campaign",
        plans_provider=lambda: deepcopy(offered), selection_provider=lambda runs: [])
    page.form.name_edit.setText("Analysis")
    page.form.job_edit.setText(str(tmp_path / "fresh-job"))
    return panel, page, selected, offered, reads


def test_page_reentry_rechecks_corrected_stores_for_the_same_plan(qtbot, tmp_path, monkeypatch):
    """An unchanged plan loses its stale refusal after the selected stores are corrected."""
    stores = dict(path=tmp_path / "original-stores.toml", ready=False)
    panel, page, selected, offered, reads = _page(qtbot, tmp_path, stores)
    admissions = []

    def inspect(plan, *, series_root, stores_path):
        admissions.append((deepcopy(plan), series_root, stores_path))
        return dict(ready=stores["ready"], reason="Local store mapping requires correction")

    monkeypatch.setattr(staged_jobs, "inspect_staged_resources", inspect)
    panel.prepare_new_analysis()
    qtbot.waitUntil(lambda: page.can_close() and page.facts is not None, timeout=10000)
    assert not page.create_button.isEnabled()
    assert "Local store mapping requires correction" in page.create_button.toolTip()
    choices = page.form.choices()
    stores.update(path=tmp_path / "corrected-stores.toml", ready=True)
    offered.insert(0, dict(path=str(tmp_path / "other.toml"), sha256="b" * 64,
        size_bytes=15, label="Another plan"))
    panel.prepare_new_analysis()
    qtbot.waitUntil(lambda: page.can_close() and page.facts is not None, timeout=10000)
    assert page.create_button.isEnabled() and not page.create_button.toolTip()
    assert page.form.plan_combo.currentData() == selected
    assert page.form.choices() == choices
    assert reads == [selected, selected]
    assert admissions == [(selected, panel.series_root, tmp_path / "original-stores.toml"),
        (selected, panel.series_root, tmp_path / "corrected-stores.toml")]


def test_same_plan_old_success_cannot_enable_create_during_a_new_admission_read(qtbot, tmp_path, monkeypatch):
    """A refreshed same-plan request rejects even a transient older success."""
    stores = dict(path=tmp_path / "original-stores.toml")
    panel, page, selected, _offered, reads = _page(qtbot, tmp_path, stores)
    first_started, second_started, release_first, release_second = (Event() for _ in range(4))
    admissions = []

    def inspect(plan, *, series_root, stores_path):
        admissions.append(stores_path)
        if len(admissions) == 1:
            first_started.set()
            if not release_first.wait(10):
                raise RuntimeError("The older synthetic admission read was not released")
            return dict(ready=True, reason="Older admission")
        second_started.set()
        if not release_second.wait(10):
            raise RuntimeError("The fresh synthetic admission read was not released")
        return dict(ready=True, reason="Fresh admission")

    monkeypatch.setattr(staged_jobs, "inspect_staged_resources", inspect)
    try:
        panel.prepare_new_analysis()
        qtbot.waitUntil(first_started.is_set, timeout=10000)
        stores["path"] = tmp_path / "fresh-stores.toml"
        panel.prepare_new_analysis()
        assert page.form.plan_combo.currentData() == selected
        assert page.facts is None and not page.create_button.isEnabled()
        page.create_states.clear()
        release_first.set()
        qtbot.waitUntil(second_started.is_set, timeout=10000)
        assert page.facts is None and not page.create_button.isEnabled()
        assert not any(page.create_states)
        release_second.set()
        qtbot.waitUntil(lambda: page.can_close() and page.facts is not None, timeout=10000)
        assert page.create_button.isEnabled()
        assert page.facts["resource_admission"]["reason"] == "Fresh admission"
        assert reads == [selected, selected]
        assert admissions == [tmp_path / "original-stores.toml", tmp_path / "fresh-stores.toml"]
    finally:
        release_first.set()
        release_second.set()
        qtbot.waitUntil(panel.can_close, timeout=10000)
