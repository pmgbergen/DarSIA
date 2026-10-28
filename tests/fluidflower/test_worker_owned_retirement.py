"""Exact-incarnation retirement tests; all processes and waits are doubles.

No real process is opened, signalled, waited for or started by this file.
The doubles distinguish owned epochs from reused PIDs and foreign ancestors.
"""

from copy import deepcopy

import psutil
import pytest

from darsia.fluidflower.pipeline import worker


class _Process:
    def __init__(self, world, pid, epoch, *, stubborn=False, unkillable=False):
        self.world, self.pid, self.epoch = world, pid, epoch
        self.stubborn, self.unkillable = stubborn, unkillable
        self.alive = True
        self.descendants = []
        self.identity_error = None
        self.vanish_on_terminate = False

    def create_time(self):
        if self.identity_error is not None:
            raise self.identity_error
        if not self.alive:
            raise psutil.NoSuchProcess(self.pid)
        return self.epoch

    def cmdline(self):
        return ["authored-owned-worker", str(self.pid)]

    def children(self, recursive=False):
        assert recursive is True
        self.world.events.append(("children", self.pid))
        return list(self.descendants)

    def terminate(self):
        self.world.events.append(("terminate", self.pid))
        if self.vanish_on_terminate:
            self.alive = False
            raise psutil.NoSuchProcess(self.pid)
        if not self.stubborn:
            self.alive = False

    def kill(self):
        self.world.events.append(("kill", self.pid))
        if not self.unkillable:
            self.alive = False


class _World:
    def __init__(self, monkeypatch):
        self.processes, self.events, self.waits = {}, [], []
        self.after_first_wait = None
        monkeypatch.setattr(psutil, "Process", self.open_process)
        monkeypatch.setattr(psutil, "wait_procs", self.wait_procs)

    def add(self, pid, epoch, **kwargs):
        result = _Process(self, pid, epoch, **kwargs)
        self.processes[pid] = result
        return result

    def open_process(self, pid):
        self.events.append(("open", pid))
        process = self.processes.get(pid)
        if process is None or not process.alive:
            raise psutil.NoSuchProcess(pid)
        return process

    def wait_procs(self, processes, timeout):
        self.waits.append((tuple(p.pid for p in processes), timeout))
        gone = [p for p in processes if not p.alive]
        pending = [p for p in processes if p.alive]
        if len(self.waits) == 1 and self.after_first_wait is not None:
            self.after_first_wait()
        return gone, pending


def _identity(process):
    return dict(pid=process.pid, process_create_time=str(process.epoch),
        command_line=["authored-owned-worker", str(process.pid)])


def test_retire_exact_descendants_child_first_without_touching_foreign_ancestors(monkeypatch):
    world = _World(monkeypatch)
    root = world.add(101, 1000.25)
    child = world.add(102, 1000.5)
    grandchild = world.add(103, 1000.75)
    ipc_child = world.add(104, 1000.875)
    ancestor = world.add(900, 999.0)
    unrelated = world.add(901, 1001.0)
    root.descendants = [child, grandchild]
    root_identity = _identity(root)
    observed = [_identity(ipc_child), _identity(child)]
    original = deepcopy((root_identity, observed))

    result = worker.retire_owned_processes(root_identity, observed)

    assert set((i["pid"], i["process_create_time"]) for i in result) == {
        (101, "1000.25"), (102, "1000.5"), (103, "1000.75"), (104, "1000.875")}
    termination = [pid for event, pid in world.events if event == "terminate"]
    assert termination.index(103) < termination.index(101)
    assert termination.index(102) < termination.index(101)
    assert sorted(termination) == [101, 102, 103, 104]
    assert not any(event == "kill" for event, _ in world.events)
    assert world.waits == [((103, 102, 104, 101), 10.0), ((), 10.0)]
    assert ancestor.alive and unrelated.alive
    assert not any(pid in {900, 901} for _, pid in world.events)
    assert (root_identity, observed) == original


def test_reused_root_pid_does_not_license_new_tree_or_foreign_process_termination(monkeypatch):
    world = _World(monkeypatch)
    replacement = world.add(101, 2000.0)
    new_child = world.add(105, 2000.25)
    replacement.descendants = [new_child]
    known_old_child = world.add(102, 1000.5)
    old_root = dict(pid=101, process_create_time="1000.25", command_line=["old worker"])
    result = worker.retire_owned_processes(old_root, [_identity(known_old_child)])
    assert result[0] == old_root
    assert replacement.alive and new_child.alive and not known_old_child.alive
    assert ("children", 101) not in world.events
    assert [pid for event, pid in world.events if event == "terminate"] == [102]
    assert not any(pid == 105 for _, pid in world.events)


def test_retirement_with_no_root_uses_only_exact_observed_ipc_identities(monkeypatch):
    world = _World(monkeypatch)
    known = world.add(102, 1000.5)
    reused = world.add(103, 2000.0)
    unrelated = world.add(901, 2001.0)
    observed = [_identity(known),
        dict(pid=103, process_create_time="1000.75", command_line=["old worker"])]
    result = worker.retire_owned_processes(None, observed)
    assert result == observed
    assert not known.alive and reused.alive and unrelated.alive
    assert [pid for event, pid in world.events if event == "terminate"] == [102]
    assert not any(event == "children" for event, _ in world.events)


def test_pending_owned_epoch_is_rechecked_before_kill_and_waited_again(monkeypatch):
    world = _World(monkeypatch)
    root = world.add(101, 1000.25, stubborn=True)
    result = worker.retire_owned_processes(_identity(root))
    assert result == [_identity(root)]
    assert [event for event, pid in world.events if pid == 101 and event in {"terminate", "kill"}] == [
        "terminate", "kill"]
    assert world.waits == [((101,), 10.0), ((101,), 10.0)]
    assert not root.alive


def test_pid_reuse_between_terminate_and_kill_preserves_replacement(monkeypatch):
    world = _World(monkeypatch)
    root = world.add(101, 1000.25, stubborn=True)
    replacement = []
    def reuse():
        root.alive = False
        replacement.append(world.add(101, 2000.0))
    world.after_first_wait = reuse
    result = worker.retire_owned_processes(_identity(root))
    assert result == [_identity(root)]
    assert replacement[0].alive
    assert ("kill", 101) not in world.events
    assert world.waits == [((101,), 10.0), ((101,), 10.0)]


def test_retirement_incomplete_guard_blocks_replacement_after_both_waits(monkeypatch):
    world = _World(monkeypatch)
    root = world.add(101, 1000.25, stubborn=True, unkillable=True)
    with pytest.raises(RuntimeError, match="retirement is incomplete; no replacement"):
        worker.retire_owned_processes(_identity(root))
    assert root.alive
    assert world.waits == [((101,), 10.0), ((101,), 10.0)]
    assert ("terminate", 101) in world.events and ("kill", 101) in world.events


def test_access_denied_is_not_treated_as_owned_incarnation_absence(monkeypatch):
    world = _World(monkeypatch)
    root = world.add(101, 1000.25)
    root.identity_error = psutil.AccessDenied(root.pid)
    with pytest.raises(psutil.AccessDenied):
        worker.retire_owned_processes(_identity(root))
    assert root.alive and world.waits == []
    assert not any(event in {"terminate", "kill"} for event, _ in world.events)


def test_process_disappearing_during_terminate_is_proved_absent_without_kill(monkeypatch):
    world = _World(monkeypatch)
    root = world.add(101, 1000.25)
    root.vanish_on_terminate = True
    result = worker.retire_owned_processes(_identity(root))
    assert result == [_identity(root)]
    assert not root.alive and ("kill", 101) not in world.events
    assert world.waits == [((), 10.0), ((), 10.0)]
