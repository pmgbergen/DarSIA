"""A worker service that stops before it has started is started once more (N64, N109).

The supervisor runs the unchanged worker command and imports no DarSIA code.
Tiny synthetic logs and commands only.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from darsia_service import prestart


def started_line(log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("a", encoding="utf-8") as stream:
        stream.write('2026-10-19T17:00:00+00:00 {"kind": "started", "message": "fixture"}\n')


def failing_start(log, text):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("a", encoding="utf-8") as stream:
        stream.write("Traceback (most recent call last):\n" + text + "\n")


def records(log):
    folder = Path(log).parent / "prestart"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]


def final_lines(log):
    return [json.loads(line.partition(" ")[2]) for line in Path(log).read_text(encoding="utf-8").splitlines()
            if line.partition(" ")[2].startswith("{")]


def test_a_start_that_failed_before_started_is_retried_once_and_recorded(tmp_path):
    log = tmp_path / "logs/main-cpu.log"
    calls, waits = [], []

    def run(command):
        calls.append(command)
        if len(calls) == 1:
            failing_start(log, "PermissionError: [WinError 32] the log is in use")
            return 1
        started_line(log)
        return 0

    command = ["python", "-m", "darsia.fluidflower.worker", "--job", str(tmp_path), "--watchdog"]
    assert prestart.supervise(command, service_log=log, delay=7, run=run, sleep=waits.append,
                              environment={}) == 0
    assert calls == [command, command] and waits == [7]
    [record] = records(log)
    assert record["format"] == prestart.RECORD_FORMAT and record["attempt"] == 1
    assert record["retried"] is True and record["service_started"] is False
    assert record["error"] == "PermissionError: [WinError 32] the log is in use"
    assert record["command"] == command and record["exit_code"] == 1
    assert [event["kind"] for event in final_lines(log)] == ["started"]  # No release line.


def test_the_last_failed_start_is_recorded_with_its_lineage_and_releases_the_launch(tmp_path):
    log = tmp_path / "logs/main-cpu.log"
    errors = iter(["CachePolicyError: first", "CachePolicyError: second"])

    def run(command):
        failing_start(log, next(errors))
        return 1

    assert prestart.supervise(["worker"], service_log=log, delay=0, run=run,
                              sleep=lambda seconds: None, environment={}) == 1
    first, second = records(log)
    assert (first["attempt"], first["retried"]) == (1, True)
    assert (second["attempt"], second["retried"]) == (2, False)
    assert second["previous_record"]["path"].endswith("-attempt-1.json")
    assert len(second["previous_record"]["sha256"]) == 64
    [release] = final_lines(log)
    assert release["kind"] == "startup_failure" and release["source"] == "darsia_service.prestart"
    assert release["attempts"] == 2 and release["error"] == "CachePolicyError: second"


def test_the_same_error_twice_stops_even_when_more_starts_are_allowed(tmp_path):
    log = tmp_path / "logs/main-cpu.log"
    calls = []

    def run(command):
        calls.append(command)
        failing_start(log, "ValueError: Store-map machine ID differs from --machine")
        return 1

    assert prestart.supervise(["worker"], service_log=log, attempts=3, delay=0, run=run,
                              sleep=lambda seconds: None, environment={}) == 1
    assert len(calls) == 2
    assert records(log)[-1]["same_error_repeated"] is True


@pytest.mark.parametrize("command,environment,reason", [
    (["worker", "--once"], {}, "--once"),
    (["worker"], {"DARSIA_CACHE_FIRST_QUALIFICATION": "bound request"}, "first-qualification"),
])
def test_once_only_and_first_qualification_starts_are_never_retried(tmp_path, command, environment, reason):
    log = tmp_path / "logs/main-cpu.log"
    calls = []

    def run(arguments):
        calls.append(arguments)
        return 1

    assert prestart.supervise(command, service_log=log, delay=0, run=run,
                              sleep=lambda seconds: pytest.fail("waited"), environment=environment) == 1
    [record] = records(log)
    assert len(calls) == 1 and reason in record["retry_refused"]
    assert record["error"] == "exit code 1 before the service wrote its log"
    assert final_lines(log)[-1]["kind"] == "startup_failure"


def test_a_stop_after_started_or_a_clean_exit_is_never_a_prestart_failure(tmp_path):
    log = tmp_path / "logs/main-cpu.log"

    def run(command):
        started_line(log)
        failing_start(log, "RuntimeError: later failure")
        return 1

    assert prestart.supervise(["worker"], service_log=log, run=run, environment={}) == 1
    assert prestart.supervise(["worker"], service_log=log, run=lambda command: 0, environment={}) == 0
    assert records(log) == []
    assert "startup_failure" not in [event["kind"] for event in final_lines(log)]


def test_service_events_and_failure_summary_read_only_new_log_bytes(tmp_path):
    log = tmp_path / "service.log"
    assert prestart.service_events(log, 0) == []
    assert prestart.failure_summary(log, 0, 2) == "exit code 2 before the service wrote its log"
    started_line(log)
    offset = log.stat().st_size
    failing_start(log, "OSError: [WinError 64] The specified network name is no longer available")
    assert prestart.service_events(log, 0) == [{"kind": "started", "message": "fixture"}]
    assert prestart.service_events(log, offset) == []
    assert prestart.failure_summary(log, offset, 1).startswith("OSError: [WinError 64]")


def test_retry_refusal_names_only_the_modes_that_forbid_a_retry():
    assert prestart.retry_refusal(["worker", "--watchdog"], {}) is None
    assert "--once" in prestart.retry_refusal(["worker", "--once"], {})
    assert "first-qualification" in prestart.retry_refusal(
        ["worker"], {"DARSIA_CACHE_FIRST_QUALIFICATION": "x"})


def test_command_line_runs_the_real_command_and_needs_the_separator(tmp_path):
    log = tmp_path / "logs/fixture-cpu.log"
    with pytest.raises(SystemExit):
        prestart.main(["--service-log", str(log), sys.executable, "-c", "pass"])
    code = prestart.main(["--service-log", str(log), "--delay", "0", "--",
                          sys.executable, "-c", "import sys; sys.exit(3)"])
    assert code == 3 and [record["attempt"] for record in records(log)] == [1, 2]
    assert final_lines(log)[-1]["kind"] == "startup_failure"


def test_the_supervisor_imports_no_darsia_numerics_or_qt():
    source = Path(prestart.__file__).resolve().parents[1]
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); import darsia_service.prestart; "
             "print(sorted(m for m in sys.modules if m.split('.')[0] in "
             "{'darsia', 'darsia_cache', 'numpy', 'numba', 'PIL', 'PySide6'}))")
    output = subprocess.check_output([sys.executable, "-I", "-c", probe, str(source)], text=True)
    assert output.strip() == "[]"


def test_both_launchers_start_the_worker_through_the_supervisor():
    from darsia.fluidflower import remote_worker_launch as remote
    target = remote.RemoteWorkerTarget(machine="Second", ssh_alias="second",
        python=r"C:\existing\python.exe", checkout=r"C:\existing\repo",
        ssh_executable=r"C:\Windows\System32\OpenSSH\ssh.exe", default_shell="windows-powershell")
    launch = remote.build_remote_worker_launch(r"\\NAS\share\jobs\explicit", target=target,
                                               backend="cpu", workers=1)
    assert launch.worker_argv[:4] == (target.python, "-B", "-m", "darsia.fluidflower.worker")
    expected = ("& 'C:\\existing\\python.exe' '-B' '-m' 'darsia_service.prestart' '--service-log' "
                "'\\\\NAS\\share\\jobs\\explicit\\logs\\second-cpu.log' '--' ")
    assert launch.remote_command.startswith(expected)
    assert launch.remote_command.endswith(" '--watchdog'")
