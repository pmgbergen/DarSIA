import json
import os
import subprocess
import sys

import pytest

from darsia.fluidflower import shared_admission_lock as subject
from darsia.fluidflower.shared_job_folder import SharedJobFolder


def test_active_owner_excludes_other_sessions_and_crash_releases_kernel_lock(tmp_path):
    folder = SharedJobFolder(tmp_path)
    marker, guard = subject._paths(folder, 'test-machine')
    code = '''
import os, sys
from darsia.fluidflower.shared_admission_lock import admission
from darsia.fluidflower.shared_job_folder import SharedJobFolder
with admission(SharedJobFolder(sys.argv[1]), 'test-machine', 'child'):
 print('locked', os.getpid(), flush=True)
 sys.stdin.readline()
 os._exit(7)
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        ready = child.stdout.readline().strip().split()
        assert ready[0] == 'locked'
        actual_pid = int(ready[1])  # A Windows venv launcher can retain its own parent PID.
        held = json.loads(marker.read_bytes())
        assert held['owner']['pid'] == actual_pid and held['owner']['process_create_time']
        with pytest.raises(ValueError, match='admission is active'):
            with subject.admission(folder, 'test-machine', 'parent'):
                pytest.fail('An active owner was overridden')
        with pytest.raises(ValueError, match='admission is active'):
            subject.recover_legacy(folder, 'test-machine', confirmed=True, session_reader=lambda: [])
        child.stdin.write('crash\n')
        child.stdin.flush()
        assert child.wait(timeout=10) == 7
        assert json.loads(marker.read_bytes())['state'] == 'active'
        with subject.admission(folder, 'test-machine', 'parent'):
            current = json.loads(marker.read_bytes())
            assert current['owner']['pid'] == os.getpid()
            assert current['state'] == 'active'
        assert json.loads(marker.read_bytes())['state'] == 'idle'
        assert guard.exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def test_legacy_recovery_requires_confirmation_and_no_other_possible_owner(tmp_path):
    folder = SharedJobFolder(tmp_path)
    marker, _ = subject._paths(folder, 'test-machine')
    token = b'a'*32
    marker.write_bytes(token)
    with pytest.raises(ValueError, match='explicit recovery'):
        with subject.admission(folder, 'test-machine', 'new'):
            pytest.fail('Ownerless legacy lock was automatically replaced')
    with pytest.raises(ValueError, match='Confirm'):
        subject.recover_legacy(folder, 'test-machine', confirmed=False)
    with pytest.raises(ValueError, match='still present: 12'):
        subject.recover_legacy(folder, 'test-machine', confirmed=True,
            session_reader=lambda: [dict(pid=12, process_create_time='34.5')])
    assert marker.read_bytes() == token
    result = subject.recover_legacy(folder, 'test-machine', confirmed=True, session_reader=lambda: [])
    assert result['recovered']
    with subject.admission(folder, 'test-machine', 'new'):
        assert json.loads(marker.read_bytes())['recovered_legacy']['sha256']


def test_racing_legacy_client_is_not_replaced(tmp_path, monkeypatch):
    from pathlib import Path
    folder = SharedJobFolder(tmp_path)
    marker, _ = subject._paths(folder, 'test-machine')
    original = Path.open
    def race(path, mode='r', *args, **kwargs):
        if path == marker and mode == 'xb':
            with original(path, 'wb') as stream:
                stream.write(b'f'*32)
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', race)
    with pytest.raises(ValueError, match='appeared during startup'):
        with subject.admission(folder, 'test-machine', 'new'):
            pytest.fail('Legacy client won the exclusive marker creation')
    assert marker.read_bytes() == b'f'*32
