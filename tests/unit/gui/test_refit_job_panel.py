"""The Mass calibration page's refit job, against the real refit job API.

Offscreen Qt with a synthetic series, phase-one job and rules in tmp_path. The
shared-folder admission is the unit transport, the job identity is fixed, and
the Jobs page's launcher is replaced, so no worker process is ever started.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QMessageBox, QScrollArea

from darsia.fluidflower import job_filesystem, jobs_panel, store_map
from darsia.fluidflower import refit_job_panel as refit
from darsia.fluidflower.mass.n1_refit import RULES_FORMAT
from darsia.fluidflower.mass_calibration_display import (
    MODE_NOTICE,
    REFIT_MODE_NOTICE,
    MassCalibrationDisplay,
)
from darsia.fluidflower.pipeline.job_queue import name
from darsia.fluidflower.series_method import MethodSelection, save_series_method

from .test_native_series_method import method_window  # noqa: F401

FIT = ["r1", "r2"]
BASELINE = dict(candidate="N1", gain=24.0, beta=0.6)
MACHINE = "local-test-machine"
YES, NO = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No


def _bound(path):
    data = Path(path).read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def _files(*roots):
    return {str(path): path.read_bytes() for root in roots for path in sorted(Path(root).rglob("*"))
            if path.is_file()}


def _rules(**changes):
    """Rules shaped like the production ones, with small grids."""
    value = dict(
        format=RULES_FORMAT, stage1_window_h=[.5, 2.375], stage2_start_h=2.5, post_bin_edges_h=[8., 48.],
        width_limit=.05, anchor_target_h=2.247, anchor_tolerance_h=.13, baseline=dict(BASELINE),
        gain_starts=[1., 20., 24.], beta_starts=[.6, 0., 1., 3.], require_frame_list_evidence=False,
        gain_grid=dict(spacing="geometric", wide=[[1e-2, 1e3, 61]],
                       local=dict(factor_low=.5, factor_high=2., floor=1e-8, points=41)),
        beta_grid=dict(spacing="linear", wide=[[0., 4., 81], [4., 16., 13]], extra=[128.],
                       local=dict(half_width=.25, floor=0., points=41)),
    )
    value.update(changes)
    return value


def _identity(repository, *, calibration_sha256, precision="float32"):
    """A fixed job identity in place of this checkout's commit and lock."""
    return dict(commit="c" * 40, calibration_sha256=calibration_sha256, precision=precision,
                created_utc="2026-10-21T00:00:00+00:00",
                environment_lock=dict(path="uv.lock", sha256="d" * 64))


def _no_launch(*_args, **_kwargs):
    pytest.fail("No worker is started without Start refit job and its confirmation")


@pytest.fixture
def series(tmp_path, monkeypatch):
    """Five runs (two fit, one held out, one descriptive, one excluded) and a finished phase-one job."""
    admitted = job_filesystem.SharedFolderAdmission(True, "smb", "unit transport", "no filesystem qualification")
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path: admitted)
    monkeypatch.setattr(jobs_panel, "launch_local_worker", _no_launch)
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version = 1\nseries_id = "TEST"\nname = "TEST"\n\n[roots]\nseries = "."\n\n'
        '[runs]\nids = ["r1", "r2", "r3", "r4", "r5"]\n\n[runs.excluded]\nr5 = "outlier"\n',
        encoding="utf-8")
    baseline = root / "calibration" / "baseline.json"
    baseline.parent.mkdir()
    baseline.write_text(json.dumps(dict(parameters=BASELINE, calibration_runs=FIT, heldout_runs=["r3"],
                                        descriptive_runs=["r4"])), encoding="utf-8")
    save_series_method(root, MethodSelection(active="phase_model", model_version="v1",
                                             mass_calibration="calibration/baseline.json"))
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps(_rules()), encoding="utf-8")
    phase_job = tmp_path / "jobs" / "phase-one"
    for run in FIT:
        attempt = root / "results" / "phase-one" / run / ".attempts" / "a1"
        for index in (1, 2, 3):
            frame = attempt / "frames" / f"{run}_f{index}"
            frame.mkdir(parents=True)
            row = dict(run=run, frame_index=index, selected_target_h=.5 * index, elapsed_h=.5 * index)
            (frame / "native_statistics.json.gz").write_bytes(gzip.compress(json.dumps(row).encode()))
        manifest = attempt / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        completion = phase_job / "results" / run / "completed.json"
        completion.parent.mkdir(parents=True)
        completion.write_text(json.dumps(dict(complete=True, run=run, manifest=_bound(manifest))),
                              encoding="utf-8")
    roles = dict(fit=FIT, heldout=["r3"], descriptive=["r4"], excluded=["r5"])
    return SimpleNamespace(root=root, baseline=baseline, rules=rules, phase_job=phase_job,
                           jobs=tmp_path / "jobs", roles=roles, tmp=tmp_path)


def _create(series, job_id="refit-test"):
    """The queued job through the page's own create function and the real API."""
    refit.create_job(series.root, roles=series.roles, rules=_rules(), baseline=_bound(series.baseline),
                     phase_job=series.phase_job, jobs_root=series.jobs, job_id=job_id,
                     job_identity=_identity)
    return series.jobs / job_id


def _panel(qtbot, series):
    panel = refit.RefitJobPanel(job_identity=_identity, machine=MACHINE)
    qtbot.addWidget(panel)
    panel.set_series(series.root)
    return panel


def _idle(qtbot, panel):
    qtbot.waitUntil(lambda: panel._thread is None, timeout=20000)


def _rows(panel):
    table = panel.tasks
    return {table.item(row, 0).text(): tuple(table.item(row, column).text() for column in range(1, 5))
            for row in range(table.rowCount())}


def test_exact_gui_input_choices_are_restored_without_changing_series(qtbot, series, monkeypatch):
    from PySide6.QtCore import QSettings
    from darsia.fluidflower.pipeline import local_overflow_refit
    settings = QSettings(str(series.tmp/'local.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(refit.RefitJobPanel, '_input_settings', staticmethod(lambda: settings))
    monkeypatch.setattr(local_overflow_refit, 'read_request', lambda *args, **kwargs: {})
    before = (series.root/'series.toml').read_bytes()
    first = _panel(qtbot, series)
    assert first.load_rules(series.rules)
    source = series.tmp/'overflow.json'
    source.write_text('{}', encoding='utf-8')
    first.set_overflow_enabled(True, source)
    second = _panel(qtbot, series)
    assert second.rules == first.rules
    assert second.f_check.isChecked()
    assert second.overflow_inputs == _bound(source)
    assert (series.root/'series.toml').read_bytes() == before


@pytest.mark.parametrize("rules_choice", ["", "   ", "directory", "missing.json"])
def test_overflow_toggle_remembers_no_rules_for_an_incomplete_file_choice(qtbot, series, monkeypatch, rules_choice):
    from PySide6.QtCore import QSettings
    from darsia.fluidflower.pipeline import job_queue
    settings = QSettings(str(series.tmp/'incomplete.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(refit.RefitJobPanel, '_input_settings', staticmethod(lambda: settings))
    panel = _panel(qtbot, series)
    assert panel.load_rules(series.rules)
    chosen = str(series.tmp) if rules_choice == "directory" else (
        str(series.tmp/rules_choice) if rules_choice.strip() else rules_choice)
    panel.rules_edit.setText(chosen)
    monkeypatch.setattr(job_queue, 'descriptor', lambda *_a, **_k: pytest.fail('No rules file was selected'))
    panel.f_check.setChecked(True)
    saved = json.loads(settings.value(panel._input_settings_key(), '', type=str))
    assert saved['rules'] is None and saved['overflow_enabled'] is True
    assert panel._create_blocker() == 'Choose the reviewed overflow method and scale-source inputs for the free-gas target.'


def test_remembered_rules_are_not_reused_after_source_change(qtbot, series, monkeypatch):
    from PySide6.QtCore import QSettings
    settings = QSettings(str(series.tmp/'local.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(refit.RefitJobPanel, '_input_settings', staticmethod(lambda: settings))
    first = _panel(qtbot, series)
    assert first.load_rules(series.rules)
    changed = _rules()
    changed['width_limit'] = .09
    series.rules.write_text(json.dumps(changed), encoding='utf-8')
    second = _panel(qtbot, series)
    assert second.rules is None
    assert 'previously chosen rules cannot be used' in second.rules_label.text()


def test_moved_overflow_sources_are_chosen_and_verified_in_the_embedded_gui(qtbot, series, monkeypatch):
    from darsia.fluidflower import calibration_source_locations
    from darsia.fluidflower.pipeline import local_overflow_refit
    source = series.tmp/'inputs.json'
    source.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(local_overflow_refit, 'read_request',
                        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError('old folder')))
    monkeypatch.setattr(refit.QFileDialog, 'getOpenFileName', lambda *a, **k: (str(source), ''))
    monkeypatch.setattr(calibration_source_locations, 'overflow_source_folder',
                        lambda *args, **kwargs: str(series.tmp/'old'))
    calls = []
    def relocate(root, bound, old, new):
        calls.append((root, bound, old, new))
        return dict(request=bound, verified_sources=2)
    monkeypatch.setattr(calibration_source_locations, 'relocate_overflow_sources', relocate)
    panel = _panel(qtbot, series)
    panel.choose_overflow_inputs()
    assert panel.overflow_inputs is None
    assert not panel.source_folder_panel.isHidden()
    panel.source_new_folder.setText(str(series.tmp/'moved'))
    assert panel._use_source_folder()
    _idle(qtbot, panel)
    assert len(calls) == 1
    assert calls[0][0:2] == (series.root, _bound(source))
    assert panel.overflow_inputs == _bound(source)
    assert panel.f_check.isChecked()
    assert 'Verified 2 unchanged sources' in panel.source_folder_status.text()


def test_roles_follow_the_review_list_and_name_runs_without_a_role(series):
    roles, summary = refit.series_refit_roles(series.root)
    assert roles == series.roles
    assert "2 fit" in summary and "1 excluded" in summary
    text = (series.root / "series.toml").read_text(encoding="utf-8")
    assert text.count('"r5",]') == 1  # The run list as the method writer leaves it.
    (series.root / "series.toml").write_text(text.replace('"r5",]', '"r5", "r6",]'), encoding="utf-8")
    roles, summary = refit.series_refit_roles(series.root)
    assert "r6" not in sum(roles.values(), [])
    assert "Left out of mass calibration, without a role: r6." in summary


def test_the_baseline_is_the_series_mass_calibration_bound_by_its_sha(series, tmp_path):
    assert refit.baseline_calibration(series.root) == _bound(series.baseline)
    other = tmp_path / "other"
    other.mkdir()
    (other / "series.toml").write_text('schema_version = 1\nseries_id = "OTHER"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="references no baseline mass calibration"):
        refit.baseline_calibration(other)


def test_rules_are_checked_against_the_roles_and_the_baseline(series, tmp_path):
    baseline = _bound(series.baseline)
    rules, summary = refit.read_rules(series.rules, roles=series.roles, baseline=baseline)
    assert rules == _rules()
    assert summary.startswith("Stage 1 0.5-2.375 h · stage 2 from 2.5 h · post-injection bins split at 8, 48 h")
    assert summary.endswith("baseline gain 24, beta 0.6.")
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(dict(format="other")), encoding="utf-8")
    with pytest.raises(ValueError, match=RULES_FORMAT):
        refit.read_rules(wrong, roles=series.roles, baseline=baseline)
    wrong.write_text(json.dumps(_rules(baseline=dict(BASELINE, gain=25.0))), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline differs"):
        refit.read_rules(wrong, roles=series.roles, baseline=baseline)
    with pytest.raises(ValueError, match="cannot also be held out"):
        refit.read_rules(series.rules, roles=dict(series.roles, heldout=["r1"]), baseline=baseline)


def test_phase_job_readiness_names_the_fit_runs_without_a_published_result(series):
    ready, problems = refit.phase_job_readiness(series.phase_job, FIT + ["r9"])
    assert ready == FIT
    assert len(problems) == 1 and problems[0].startswith("r9: ")


def test_a_default_job_name_is_a_valid_job_name():
    moment = datetime(2026, 10, 21, 1, 2, 3, tzinfo=timezone.utc)
    assert refit.default_job_id(moment) == "refit-20261021T010203Z"
    job_id = refit.default_job_id()
    name(job_id)
    assert job_id.startswith("refit-") and job_id.endswith("Z")


def test_status_texts_name_each_task_state_step_and_result():
    status = dict(job_root="x/refit-a", complete=False, stop_required=None, tasks={
        "refit": dict(state="completed", step="complete", parameters=dict(candidate="N1", gain=23.5, beta=.65),
                      stop_required=False),
        "loro-r1": dict(state="running", step="statistics_read", parameters=None, stop_required=None),
        "loro-r2": dict(state="waiting", step=None, parameters=None, stop_required=None)})
    assert refit.status_rows(status) == [
        ("Mass calibration", "Completed", "complete", "gain 23.5 · beta 0.65 · candidate N1", "No"),
        ("loro-r1", "Running", "statistics_read", "—", "—"),
        ("loro-r2", "Waiting", "—", "—", "—")]
    assert refit.status_summary(status) == "3 tasks: 1 completed, 1 running, 1 waiting."
    assert "grid check" in refit.status_summary(dict(status, stop_required=True))
    document = dict(parameters=dict(gain=23.5, beta=.65), calibration_runs=FIT)
    text = refit.frozen_text((document, dict(path="f.json", sha256="a" * 64, size_bytes=10)))
    assert "SHA-256 " + "a" * 64 in text and "fitted on 2 runs" in text and "not adopted" in text
    assert refit.frozen_text(None) == "" and "could not be read: gone" in refit.frozen_text(None, "gone")


def test_the_panel_creates_the_current_local_refit_job_and_starts_nothing(qtbot, series, monkeypatch):
    from darsia.fluidflower import execution
    from darsia.fluidflower.mass.n1_statistics import CONTRACT
    from darsia.fluidflower.pipeline import local_refit
    questions = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **_kwargs: questions.append(args[2]) or YES)
    monkeypatch.setattr(execution, 'detect_host_resources',
                        lambda: dict(memory=dict(available_bytes=1024**4)))
    monkeypatch.setattr(local_refit.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1024**4))
    for path in series.root.glob('results/phase-one/*/.attempts/a1/frames/*/native_statistics.json.gz'):
        row = json.loads(gzip.decompress(path.read_bytes()))
        row.update(contract=CONTRACT,
                   paired_estimates={key: [] for key in ('own_native_relative', 'column_native_relative',
                        'solubility_ratio', 'kind', 'aqueous_capacity_g')},
                   main_aqueous_histogram=dict(native_relative=[.01], capacity_g=[1.]))
        path.write_bytes(gzip.compress(json.dumps(row).encode()))
    panel = _panel(qtbot, series)
    panel._identity = None  # Actual local source identity, rather than the historical queued fixture.
    assert panel.roles == series.roles
    assert not panel.create_button.isEnabled()
    assert panel.load_rules(series.rules)
    panel.check_phase_job(series.phase_job)
    _idle(qtbot, panel)
    assert panel.phase_ready and panel.phase_label.text().startswith("2 of 2 fit runs")
    panel.jobs_edit.setText(str(series.jobs))
    panel.name_edit.setText("refit-test")
    assert panel.create_button.isEnabled()
    assert panel.create()
    qtbot.waitUntil(lambda: panel._thread is None and panel.snapshot is not None, timeout=20000)
    assert questions == []  # Creation prepares metadata; calculation still requires Start.
    job_root = series.jobs / "refit-test"
    assert panel.job_root == job_root
    job = json.loads((job_root / "job.json").read_text(encoding="utf-8"))
    assert job["pipeline"] == "n1-calibration-refit"
    assert job['format'] == local_refit.FORMAT
    assert job["expected_runs"] == ["refit", "loro-r1", "loro-r2"]
    assert job["calibration_sha256"] == _bound(series.baseline)["sha256"]
    assert set(job['plans']) == {'refit', 'loro-r1', 'loro-r2'}
    assert all(Path(bound['path']).is_file() for bound in job['plans'].values())
    assert not (job_root/'launch.json').exists()
    assert _rows(panel) == {task: ("Waiting", "—", "—", "—") for task in ("Mass calibration", "loro-r1", "loro-r2")}
    assert panel.job_label.text() == "Mass calibration job refit-test: 3 tasks: 3 waiting."
    assert panel.start_button.isEnabled() and panel.result_label.text() == ""
    assert panel.name_edit.text() == "refit-test"


def test_start_requires_a_read_and_retains_an_unresolved_launch_on_reopen(qtbot, series, monkeypatch):
    from darsia.fluidflower.job_launch_state import begin_launch_request
    job_root = _create(series)
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: pytest.fail("not asked"))
    panel = _panel(qtbot, series)
    assert not panel.start_button.isEnabled() and not panel.start()
    begin_launch_request(job_root, machine=MACHINE, backend="cpu", workers=1, kind="local")
    panel.open_job(job_root)
    _idle(qtbot, panel)
    assert refit.START_BLOCKER == ""
    assert not panel.start_button.isEnabled()
    assert "worker launch is already recorded" in panel.start_button.toolTip()
    assert panel.start_reason.text() == panel.start_button.toolTip() and not panel.start_reason.isHidden()
    assert not panel.start()  # The launcher stub fails the test if anything is started.
    assert panel.refresh_button.isEnabled() and panel.snapshot is not None
    panel.open_job(job_root)
    _idle(qtbot, panel)
    assert not panel.start_button.isEnabled() and not panel.start()


def test_create_is_refused_without_a_ready_phase_one_job(qtbot, series, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: pytest.fail("not asked"))
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **_kwargs: shown.append(args[2]))
    panel = _panel(qtbot, series)
    panel.name_edit.setText("not-ready")
    panel.load_rules(series.rules)
    panel.jobs_edit.setText(str(series.jobs))
    assert not panel.create()
    assert shown == ["Choose a phase-map generation job with completed results for every fit run."]
    assert not (series.jobs / panel.name_edit.text()).exists()


def test_the_follow_view_shows_steps_and_the_frozen_result_with_its_sha(qtbot, series):
    job_root = _create(series)
    # A finite historical inline-completion view, rather than a fabricated
    # closed alias publication. The alias path has a real queue/index test.
    legacy = json.loads((job_root / "job.json").read_text(encoding="utf8"))
    legacy.pop("format")
    (job_root / "job.json").write_text(json.dumps(legacy), encoding="utf8")
    (job_root / "queue" / "loro-r1.json").unlink()
    lease = job_root / "leases" / "loro-r1.attempt"
    lease.mkdir(parents=True)
    (lease / "progress_000001.json").write_text(
        json.dumps(dict(written_unix_ns=1, event=dict(step="statistics_read"))), encoding="utf-8")
    frozen = job_root / "results" / "refit" / "attempt" / "frozen_calibration.json"
    frozen.parent.mkdir(parents=True)
    parameters = dict(candidate="N1", gain=23.5, beta=.65)
    frozen.write_text(json.dumps(dict(parameters=parameters, calibration_runs=FIT)), encoding="utf-8")
    (job_root / "results" / "refit" / "completed.json").write_text(json.dumps(dict(
        complete=True, run="refit", parameters=parameters, stop_required=False,
        frozen_calibration=_bound(frozen))), encoding="utf-8")
    panel = _panel(qtbot, series)
    assert panel.open_job(job_root)
    qtbot.waitUntil(lambda: panel._thread is None and panel.snapshot is not None, timeout=20000)
    rows = _rows(panel)
    assert rows["Mass calibration"] == ("Completed", "—", "gain 23.5 · beta 0.65 · candidate N1", "No")
    assert rows["loro-r1"][:2] == ("Running", "statistics_read")
    assert rows["loro-r2"][0] == "Waiting"
    assert "SHA-256 " + _bound(frozen)["sha256"] in panel.result_label.text()
    assert "not adopted" in panel.result_label.text()
    frozen.write_text("{}", encoding="utf-8")
    panel.refresh()
    qtbot.waitUntil(lambda: "could not be read" in panel.result_label.text(), timeout=20000)
    _idle(qtbot, panel)


def test_another_job_folder_is_refused_when_it_is_read(qtbot, series):
    panel = _panel(qtbot, series)
    panel.open_job(series.phase_job)
    _idle(qtbot, panel)
    assert panel.snapshot is None
    assert panel.job_label.text().startswith("That did not work:")


def test_once_admitted_start_asks_first_then_uses_the_jobs_launcher(qtbot, series, monkeypatch):
    """The admitted route uses the original launcher and its persistent reservation."""
    job_root = _create(series)
    stores = series.tmp / "stores.toml"
    stores.write_text("# local test settings\n", encoding="utf-8")
    monkeypatch.setattr(store_map, "default_store_map_path", lambda **_kwargs: stores)
    launches = []

    def launch(root, **options):
        launches.append((Path(root), options))
        return dict(started=True, console="ui/launches/x/console.log")

    monkeypatch.setattr(jobs_panel, "launch_local_worker", launch)
    answers, asked = iter([NO, YES]), []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args, **_kwargs: asked.append(args[2]) or next(answers))
    panel = _panel(qtbot, series)
    panel.open_job(job_root)
    _idle(qtbot, panel)
    assert panel.start_button.isEnabled() and panel.start_reason.isHidden()
    assert panel.notice.text() == refit.ADMITTED_NOTICE
    assert not panel.start()
    assert launches == []
    assert MACHINE in asked[0] and "Nothing is adopted" in asked[0]
    assert panel.start()
    from darsia.fluidflower.job_launch_state import read_launch_requests
    requests = read_launch_requests(job_root)
    assert len(requests) == 1 and requests[0]["released"] is False
    assert launches == [(job_root, dict(backend="cpu", workers=1, machine=MACHINE, python=sys.executable,
                                        cwd=None, stores=stores, launch_directory=requests[0]["launch_directory"]))]
    assert "ui/launches/x/console.log" in panel.job_label.text()
    _idle(qtbot, panel)
    assert not panel.start_button.isEnabled() and not panel.start()
    assert len(launches) == 1


def test_once_admitted_start_is_refused_while_this_machine_has_no_store_settings(qtbot, series, monkeypatch):
    job_root = _create(series)
    monkeypatch.setattr(store_map, "default_store_map_path", lambda **_kwargs: series.tmp / "absent.toml")
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: pytest.fail("not asked"))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **_kwargs: warnings.append(args[2]))
    panel = _panel(qtbot, series)
    panel.open_job(job_root)
    _idle(qtbot, panel)
    assert not panel.start()
    assert "Machine stores" in warnings[0]


def test_start_confirmation_cannot_follow_a_series_change_to_another_job(qtbot, series, monkeypatch):
    job_root = _create(series)
    stores = series.tmp / "stores.toml"
    stores.write_text("# finite local settings\n", encoding="utf8")
    monkeypatch.setattr(store_map, "default_store_map_path", lambda **kwargs: stores)
    panel = _panel(qtbot, series)
    panel.open_job(job_root)
    _idle(qtbot, panel)
    assert panel.start_button.isEnabled()
    def change_series(*args, **kwargs):
        panel.set_series(None)  # The real reset runs inside the final question's event loop.
        return YES
    monkeypatch.setattr(QMessageBox, "question", change_series)
    assert not panel.start()
    assert panel.job_root is None and not panel.start_button.isEnabled()
    assert not list((job_root / "ui" / "launches").glob("*.request.json"))


def test_the_choosers_hand_the_chosen_paths_to_their_checks(qtbot, series, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    panel = _panel(qtbot, series)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args, **_kwargs: (str(series.rules), "JSON"))
    folders = iter([str(series.phase_job), str(series.jobs), str(series.phase_job)])
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: next(folders))
    panel.choose_rules()
    assert panel.rules == _rules() and panel.rules_edit.text() == str(series.rules)
    panel.choose_phase_job()
    _idle(qtbot, panel)
    assert panel.phase_ready and panel.phase_edit.text() == str(series.phase_job)
    panel.choose_jobs_folder()
    assert panel.jobs_edit.text() == str(series.jobs)
    panel.choose_job()
    _idle(qtbot, panel)
    assert panel.job_root == series.phase_job and panel.snapshot is None


def test_a_result_read_for_another_series_is_dropped(qtbot, series, tmp_path):
    job_root = _create(series)
    panel = _panel(qtbot, series)
    panel.open_job(job_root)
    other = tmp_path / "other"
    other.mkdir()
    (other / "series.toml").write_text('schema_version = 1\nseries_id = "OTHER"\n[runs]\nids = []\n',
                                       encoding="utf-8")
    panel.set_series(other)
    _idle(qtbot, panel)
    assert panel.job_root is None and panel.snapshot is None and panel.tasks.rowCount() == 0


def test_the_gui_checkout_is_the_source_tree_this_code_runs_from():
    import darsia

    assert Path(darsia.__file__).resolve().is_relative_to(refit.gui_checkout())
    assert (refit.gui_checkout() / "src" / "darsia").is_dir()


def test_the_panel_describes_explicit_start_and_shows_the_overflow_target_choice(qtbot, series):
    panel = _panel(qtbot, series)
    assert panel.notice.text() == refit.NOTICE
    assert panel.notice.text().startswith("Create prepares local inputs.")
    assert "waits for queue admission" not in panel.notice.text()
    assert "Start runs the original mass fit on this computer." in panel.notice.text()
    assert panel.create_button.text() == "Create mass calibration job"
    assert panel.start_button.text() == "Start mass calibration"
    assert panel.open_button.text() == "Open mass calibration job…"
    assert panel.title() == "Mass calibration job — fit to the injected mass"
    assert panel.f_check.isEnabled() and not panel.f_check.isChecked()
    assert panel.f_check.text() == "Use measured overflow as the free CO₂-gas target"
    assert panel.loro_check.isChecked()


def test_reading_the_series_writes_nothing(qtbot, series):
    before = _files(series.root, series.jobs)
    panel = _panel(qtbot, series)
    panel.load_rules(series.rules)
    panel.check_phase_job(series.phase_job)
    _idle(qtbot, panel)
    assert _files(series.root, series.jobs) == before


def test_the_display_offers_the_refit_only_for_a_series_and_keeps_its_default(qtbot, series):
    display = MassCalibrationDisplay()
    qtbot.addWidget(display)
    assert display.refit_area.isHidden()
    assert not display.automatic_button.isEnabled() and not display.automatic_button.isChecked()
    display.set_refit_series(series.root)
    assert not display.refit_area.isHidden()
    assert display.automatic_button.isChecked() and not display.automatic_button.isEnabled()
    assert not display.manual_button.isEnabled() and not display.manual_button.isChecked()
    assert display.mode_notice.text() == REFIT_MODE_NOTICE
    areas = display.findChildren(QScrollArea)
    assert areas == [display.refit_area]  # The page's one scroll area, inside no other.
    assert display.refit_area.widget() is display.refit_panel
    display.set_refit_series(None)
    assert display.refit_area.isHidden()
    assert not display.automatic_button.isEnabled() and not display.automatic_button.isChecked()
    assert display.mode_notice.text() == MODE_NOTICE


def test_the_phase_map_mass_calibration_page_offers_the_refit_and_the_colour_page_does_not(
        qtbot, method_window):
    window = method_window
    root = window.phase_model_workspace.series_roots["series-a"]
    (root / "frozen.json").write_text(json.dumps(dict(parameters=BASELINE)), encoding="utf-8")
    save_series_method(root, MethodSelection(active="phase_model", model_version="v1",
                                             mass_calibration="frozen.json"))
    window._refresh_method_state(force=True)
    before = _files(root)
    window._select_row(("phase_model", "mass_from_phase"))
    display = window.mass_calibration_display
    qtbot.waitUntil(lambda: display.refit_panel.series_root == root, timeout=10000)
    assert not display.sections['job'].isHidden()
    assert display.refit_panel.series_root == root
    assert display.refit_panel.baseline == _bound(root / "frozen.json")
    window._select_row(("calibration", "mass"))
    assert not display.sections['job'].isHidden()  # Both legacy routes share the active method.
    window._show_mass_calibration_values("colour")
    assert not display.sections['job'].isEnabled()  # Visible navigation retains its position.
    qtbot.waitUntil(lambda: display.pressure_complete, timeout=10000)
    assert _files(root) == before

def test_individual_choice_is_visible_synced_and_passed_to_original_waiter(qtbot, series, monkeypatch):
    from darsia.fluidflower.calibration_preparation import CalibrationPreparation
    panel = _panel(qtbot, series)
    assert panel.load_rules(series.rules)
    _idle(qtbot, panel)
    window = SimpleNamespace(_select_row=lambda *a:None,show_phase_model_workflow=lambda:None,
        open_image_selection_plan=lambda:None)
    view = CalibrationPreparation(window,panel)
    qtbot.addWidget(view)
    assert panel.calibration_scope_combo.currentData() == 'shared'
    assert view.scope.currentData() == 'shared'
    view.scope.setCurrentIndex(view.scope.findData('individual'))
    assert panel.calibration_scope_combo.currentData() == 'individual'
    assert not panel.loro_check.isEnabled()
    assert 'own gain/beta' in panel.fit_images.text()
    monkeypatch.setattr(panel,'_poll_phase_one',lambda:None)
    panel.name_edit.setText('individual-fit')
    all_active = ["r1", "r2", "r3", "r4"]
    assert panel.roles == dict(fit=all_active, heldout=[], descriptive=[], excluded=["r5"])
    (series.phase_job/'job.json').write_text(json.dumps(dict(expected_runs=all_active)))
    assert panel.start_after_phase_one(series.phase_job, all_active)
    assert panel._automatic['arguments']['calibration_scope']=='individual'
    assert panel._automatic['arguments']['loro'] is False
    assert panel._automatic['arguments']['roles'] == panel.roles
    panel._automatic=None
    panel._phase_wait_timer.stop()

def test_switching_scope_cannot_start_a_job_prepared_for_another_scope(qtbot, series):
    panel=_panel(qtbot,series)
    panel.job_root=series.jobs/'prepared'
    panel.name_edit.setText('prepared')
    panel.snapshot=dict(start=dict(blocker=''),local_variant_f=False)
    panel.calibration_scope_combo.setCurrentIndex(1)
    assert 'different shared/individual' in panel._start_blocker()
