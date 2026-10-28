"""Real local Qt owner/providers; no native app/job/worker/SSH/probe starts."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest
from PySide6.QtWidgets import QWidget, QVBoxLayout

from darsia.fluidflower import job_launch_state as launch
from darsia.fluidflower import jobs_metadata as metadata
from darsia.fluidflower import remote_worker_launch as remote
from darsia.fluidflower import jobs_panel as jobs
from darsia.fluidflower import native_jobs_registry as registry
from darsia.fluidflower import native_jobs_workspace as ui


def put(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def series(root, identifier):
    root.mkdir()
    (root / "series.toml").write_text(
        f'schema_version=1\nseries_id="{identifier}"\n[runs]\nids=["ac52"]\n',
        encoding="utf-8",
    )
    return root


def job(root):
    put(
        root / "job.json",
        dict(expected_runs=["ac52"], precision="float32", commit="a" * 40),
    )
    return root


def plan(path, *, adopted=True, precision="float32"):
    path.write_text(
        f'format="darsia-series-v1"\nadopted_for_execution={str(adopted).lower()}\njob_id="Existing"\n[job]\nprecision="{precision}"\n',
        encoding="utf-8",
    )
    return path


def snapshot(row):
    return dict(
        root=row["root"],
        name=row.get("name", "Existing"),
        series=row.get("series_label", "UI only"),
        state="Waiting",
        job_facts=dict(precision="float32", commit="a" * 40),
        run_count=1,
        completed_runs=0,
        committed=0,
        total_frames=2,
        remaining=2,
        runs=[],
        workers=[],
        machines=[],
        registered_machines=[],
        logs=[],
        errors=[],
        launch_requests=[],
    )


@pytest.fixture
def local_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "require_unc", lambda value: None)
    return registry.JobsUIRegistry(tmp_path / "local_QStandardPaths_fixture")


@pytest.fixture
def workspace(local_registry, tmp_path, qtbot):
    def forbidden(*a, **k):
        pytest.fail(
            "No real producer/worker/probe/SSH may run in owner navigation tests"
        )

    value = ui.NativeJobsWorkspace(
        registry=local_registry,
        local_machine="Main",
        reader=snapshot,
        job_creator=forbidden,
        local_launcher=forbidden,
        remote_launcher=forbidden,
        inventory_provider=forbidden,
    )
    qtbot.addWidget(value)
    yield value
    qtbot.waitUntil(value.can_close, timeout=5000)


def test_registry_constructor_and_browse_do_not_write_or_discover(
    local_registry, tmp_path
):
    root = series(tmp_path / "s", "literal")
    assert local_registry.context(root, "literal")["jobs"] == []
    assert (
        local_registry.jobs(root, "literal") == []
        and local_registry.plans(root, "literal") == []
    )
    assert not local_registry.root.exists()


def test_selecting_shared_folder_lists_published_series_job_without_import(workspace, tmp_path, qtbot):
    from darsia.fluidflower.shared_job_folder import SharedJobFolder

    source = series(tmp_path / "series", "own")
    folder = tmp_path / "shared"
    folder.mkdir()
    published = folder / "published-job"
    put(published / "job.json", dict(pipeline="native-phase-n1", series_id="own",
        expected_runs=["ac52"], precision="float32", commit="a" * 40))
    SharedJobFolder(folder).publish(published, master_machine="Main")
    workspace.set_series("own", source)
    workspace.shared_jobs.folder.setText(str(folder))
    qtbot.waitUntil(lambda: any(row["root"] == str(published)
                               for row in workspace.panel.snapshots), timeout=10000)
    row = next(index for index, item in enumerate(workspace.panel.snapshots)
               if item["root"] == str(published))
    assert workspace.panel.job_table.rowCount() == 1
    workspace._job_clicked(row, 0)
    assert workspace.registry.jobs(source, "own")[0]["active"] is True
    assert workspace.shared_jobs._service is None
    assert workspace.panel._session_requests == []


def test_explicit_job_plan_and_active_choice_preserve_sources_unknown_fields(
    local_registry, tmp_path
):
    root = series(tmp_path / "s", "literal")
    one = job(tmp_path / "one")
    two = job(tmp_path / "two")
    saved = plan(tmp_path / "adopted.toml")
    original = {
        (one / "job.json"): (one / "job.json").read_bytes(),
        (two / "job.json"): (two / "job.json").read_bytes(),
        saved: saved.read_bytes(),
        root / "series.toml": (root / "series.toml").read_bytes(),
    }
    local_registry.register_job(root, "literal", one)
    local_registry.register_plan(root, "literal", saved)
    path = local_registry._file(root, "literal")
    doc = registry.read_json(path)
    doc["unknown"] = {"keep": [1, "Ω", False]}
    put(path, doc)
    local_registry.register_job(root, "literal", two)
    local_registry.set_active(root, "literal", one)
    state = registry.JobsUIRegistry(local_registry.root).context(root, "literal")
    assert (
        len(state["jobs"]) == 2
        and len(state["plans"]) == 1
        and state["active_job"] == str(one)
    )
    assert state["unknown"] == doc["unknown"]
    assert all(path.read_bytes() == raw for path, raw in original.items())
    assert [row["active"] for row in local_registry.jobs(root, "literal")] == [
        True,
        False,
    ]


@pytest.mark.parametrize("adopted,precision", [(False, "float32"), (True, "float64")])
def test_invalid_or_unadopted_plan_refuses_without_local_mutation(
    local_registry, tmp_path, adopted, precision
):
    root = series(tmp_path / "s", "literal")
    saved = plan(tmp_path / "bad.toml", adopted=adopted, precision=precision)
    with pytest.raises(ValueError):
        local_registry.register_plan(root, "literal", saved)
    assert not local_registry.root.exists()


def test_changed_registered_sources_refuse_without_replacement(
    local_registry, tmp_path
):
    root = series(tmp_path / "s", "literal")
    one = job(tmp_path / "one")
    local_registry.register_job(root, "literal", one)
    before = local_registry._file(root, "literal").read_bytes()
    put(one / "job.json", dict(expected_runs=["ac19"], precision="float32"))
    with pytest.raises(ValueError, match="SHA changed"):
        local_registry.jobs(root, "literal")
    with pytest.raises(ValueError, match="immutable job"):
        local_registry.register_job(root, "literal", one)
    assert local_registry._file(root, "literal").read_bytes() == before


def test_registry_race_preserves_other_writer_and_pending_bytes(
    local_registry, tmp_path, monkeypatch
):
    root = series(tmp_path / "s", "literal")
    one = job(tmp_path / "one")
    local_registry.register_job(root, "literal", one)
    original = local_registry.context
    path = local_registry._file(root, "literal")

    def raced(*args):
        doc = original(*args)
        put(path, dict(doc, external_writer="preserve"))
        return doc

    monkeypatch.setattr(local_registry, "context", raced)
    with pytest.raises(RuntimeError, match="changed"):
        local_registry.set_active(root, "literal", one)
    assert registry.read_json(path)["external_writer"] == "preserve"
    assert list(local_registry.root.glob("*.partial"))


def test_workspace_uses_literal_current_series_id_and_fresh_b4_provider_no_model_load(
    workspace, tmp_path, qtbot
):
    a = series(tmp_path / "a", "literal-A")
    b = series(tmp_path / "b", "literal-B")
    calls = []
    workspace.protocol_reader = lambda root, run: (
        calls.append(("protocol", root, run)) or dict(run=run)
    )
    workspace.selection_resolver = lambda root, run, protocols, **kw: (
        calls.append(("resolve", root, run, kw))
        or dict(series_id="literal-A" if root == a else "literal-B", run_id=run)
    )
    workspace.set_series("Displayed-alias", a, series_label="Chosen A")
    assert (
        workspace.series_id == "literal-A"
        and calls == []
        and not workspace.registry.root.exists()
    )
    captured = workspace.panel.selection_provider
    workspace.set_series("Other-alias", b, series_label="Chosen B")
    assert workspace.panel.selection_provider(["ac52"])[0]["series_id"] == "literal-B"
    assert captured(["ac19"])[0]["series_id"] == "literal-A"
    assert calls[0][1] == b and calls[2][1] == a
    assert calls[1][3] == dict(check_files=False)
    assert workspace.panel.current is None and workspace.can_close()


def test_explicit_registration_and_click_restore_active_job_with_no_navigation_writes(
    workspace, tmp_path, qtbot
):
    root = series(tmp_path / "s", "literal")
    one = job(tmp_path / "one")
    two = job(tmp_path / "two")
    workspace.set_series("literal", root)
    workspace.register_existing_job(one)
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    workspace.register_existing_job(two)
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    saved = workspace.registry._file(root, "literal")
    before = saved.read_bytes()
    workspace.panel.refresh()
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    assert saved.read_bytes() == before and workspace.panel.current["root"] == str(two)
    workspace.panel.job_table.cellClicked.emit(0, 0)
    assert workspace.registry.context(root, "literal")["active_job"] == str(one)
    workspace.refresh_registered()
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    assert workspace.panel.current["root"] == str(one)


def test_file_dialog_cancels_write_nothing_and_mass_button_only_routes(
    workspace, tmp_path, qtbot, monkeypatch
):
    root = series(tmp_path / "s", "literal")
    workspace.set_series("literal", root)
    monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    workspace.open_existing_job()
    workspace.import_execution_plan()
    assert not workspace.registry.root.exists()
    page = QWidget()
    QVBoxLayout(page)
    qtbot.addWidget(page)
    calls = []
    button = workspace.install_mass_action(
        page, lambda: calls.append("open-job-dialog")
    )
    assert workspace.install_mass_action(page) is button
    button.click()
    assert calls == ["open-job-dialog"] and not workspace.registry.root.exists()


def test_created_job_is_recorded_for_captured_old_context_not_new_selection(
    workspace, tmp_path, qtbot
):
    a = series(tmp_path / "a", "literal-A")
    b = series(tmp_path / "b", "literal-B")
    target = tmp_path / "created"
    entered = threading.Event()
    release = threading.Event()

    def creator(plan, root, **kwargs):
        assert kwargs["series_root"] == a and kwargs["series_id"] == "literal-A"
        entered.set()
        assert release.wait(3)
        job(Path(root))
        return dict(
            root=str(root),
            name="Created",
            series_label="A",
            active=True,
            worker_started=False,
        )

    workspace._creator = creator
    workspace.set_series("literal-A", a)
    workspace.panel.create_job(dict(path="metadata-fixture"), str(target), "Created")
    qtbot.waitUntil(entered.is_set, timeout=5000)
    workspace.set_series("literal-B", b)
    release.set()
    qtbot.waitUntil(workspace.can_close, timeout=5000)
    assert workspace.registry.context(a, "literal-A")["jobs"][0]["root"] == str(target)
    assert workspace.registry.context(b, "literal-B")["jobs"] == []
    assert (
        workspace.panel.current is None and "literal-A" in workspace.status_label.text()
    )


def test_local_alias_registration_does_not_duplicate_b6_machine_and_missing_rss_stays_unknown(
    workspace,
):
    workspace.panel.current = dict(
        registered_machines=[
            dict(machine="MAIN", source=dict(path="only-metadata")),
            dict(machine="Second", source=dict(path="remote-metadata")),
        ]
    )
    rows = workspace._machines_provider()
    assert len(rows) == 2 and rows[0]["machine"] == "Main"
    assert all(row["inventory"] == {} and row["history"] == [] for row in rows)
    workspace._hardware_ready(
        dict(
            host="actual-local-fixture",
            cpu=dict(logical_cores=4),
            memory=dict(available_bytes=123),
        )
    )
    assert (
        workspace._machines_provider()[0]["inventory"]["host"] == "actual-local-fixture"
    )
    assert workspace.resources is None and workspace.panel.resource_panel is None


class OptionalResourceStub(QWidget):
    from PySide6.QtCore import Signal

    hardware_inspected = Signal(object)
    workers_selected = Signal(object)

    def __init__(self):
        super().__init__()
        self.contexts = []
        self.refreshes = 0
        self.ready = True

    def set_context(self, *args, **kwargs):
        self.contexts.append((args, kwargs))

    def refresh_known_machines(self):
        self.refreshes += 1

    def can_close(self):
        return self.ready


def test_b5_default_owner_has_no_resource_dependency_and_safe_refresh(workspace):
    assert workspace.resources is None and workspace.panel.resource_panel is None
    workspace._hardware_ready(dict(host="explicit fixture"))
    assert workspace._machines_provider()[0]["inventory"] == dict(
        host="explicit fixture"
    )
    assert workspace.can_close()


def test_optional_resource_factory_retains_worker_choice_facts_and_close_guard(
    local_registry, qtbot
):
    resources = OptionalResourceStub()
    captured = []

    def factory(**kwargs):
        captured.append(kwargs)
        return resources

    value = ui.NativeJobsWorkspace(
        registry=local_registry,
        local_machine="Main",
        reader=snapshot,
        measurement_registry="explicit local registry",
        inventory_provider="explicit provider",
        resource_factory=factory,
    )
    qtbot.addWidget(value)
    assert captured == [
        dict(
            measurement_registry="explicit local registry",
            inventory_provider="explicit provider",
        )
    ]
    assert value.resources is resources and value.panel.resource_panel is resources
    assert resources.contexts
    resources.hardware_inspected.emit(
        dict(host="Main facts", memory={"available_bytes": 123})
    )
    assert value._machines_provider()[0]["inventory"]["host"] == "Main facts"
    assert resources.refreshes == 1
    resources.workers_selected.emit(dict(machine="MAIN", workers=3, backend="cpu"))
    assert value.panel.workers_spin.value() == 3
    resources.ready = False
    assert not value.can_close()
    resources.ready = True
    assert value.can_close()


def test_optional_resource_cannot_be_replaced_silently(workspace, qtbot):
    first = OptionalResourceStub()
    second = OptionalResourceStub()
    qtbot.addWidget(second)
    workspace.attach_resources(first)
    with pytest.raises(ValueError, match="already attached"):
        workspace.attach_resources(second)
    assert workspace.resources is first and workspace.panel.resource_panel is first


def test_a_cleared_refusal_leaves_the_status_line(workspace):
    refusal = "A control publication is still finishing."
    workspace.panel.error_reported.emit(refusal)
    assert workspace.status_label.text() == refusal
    workspace.panel.error_cleared.emit(refusal)
    assert workspace.status_label.text() == ""
    workspace.status_label.setText("Shared job folder: not reachable")
    workspace.panel.error_cleared.emit(refusal)  # Another message stays.
    assert workspace.status_label.text() == "Shared job folder: not reachable"
