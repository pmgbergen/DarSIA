"""Crash-safe shared admission exclusion and explicit legacy-lock recovery."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import uuid

from .pipeline.folder_capacity import _lock_identity
from .shared_job_folder import _atomic

FORMAT = 'darsia-shared-admission-owner-v1'


def _paths(folder, machine):
    from .pipeline.job_queue import name
    identity = hashlib.sha256(name(machine).casefold().encode()).hexdigest()
    parent = folder.metadata/'services'
    parent.mkdir(parents=True, exist_ok=True)
    return parent/(identity+'.lock'), parent/(identity+'.guard')


@contextmanager
def _kernel_guard(path):
    """Keep one stable inode locked; the OS releases it even after a crash."""
    try:
        stream = path.open('x+b')
    except FileExistsError:
        stream = path.open('r+b')
    with stream:
        if path.stat().st_size == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError('Another shared Jobs admission is active on this machine') from error
        try:
            identity = _lock_identity(stream, path)
            yield
            if _lock_identity(stream, path) != identity:
                raise ValueError('Shared admission kernel lock was replaced')
        finally:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read(path):
    try:
        with path.open('rb') as stream:
            raw = stream.read(8193)
    except FileNotFoundError:
        return None, b''
    if len(raw) > 8192:
        raise ValueError('Shared admission owner record exceeds its size limit')
    if re.fullmatch(rb'[0-9a-f]{32}', raw):
        return 'legacy', raw
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get('format') != FORMAT
            or value.get('state') not in {'active', 'idle'}):
        raise ValueError('Shared admission owner record is not recognized; preserved')
    return value, raw


def _owner(machine, session):
    import psutil
    return dict(machine=machine, host=socket.gethostname(), pid=os.getpid(),
                process_create_time=str(psutil.Process().create_time()),
                session_id=session, token=uuid.uuid4().hex)


@contextmanager
def admission(folder, machine, session):
    marker, guard = _paths(folder, machine)
    with _kernel_guard(guard):
        previous, _ = _read(marker)
        if previous == 'legacy':
            raise ValueError('Legacy shared Jobs admission lock needs explicit recovery in Jobs')
        record = dict(format=FORMAT, state='active', owner=_owner(machine, session),
                      updated_utc=datetime.now(timezone.utc).isoformat())
        if isinstance(previous, dict) and previous.get('recovered_legacy'):
            record['recovered_legacy'] = previous['recovered_legacy']
        if previous is None:
            # Older clients do not know the kernel guard. Exclusive marker
            # creation must win before we enter; never replace their new lock.
            try:
                with marker.open('xb') as stream:
                    stream.write(json.dumps(record, sort_keys=True).encode('utf-8'))
            except FileExistsError as error:
                raise ValueError('A shared Jobs admission appeared during startup; retry without replacing it') from error
        else:
            _atomic(marker, record)
        try:
            yield
        finally:
            current, _ = _read(marker)
            if current != record:
                raise ValueError('Shared admission owner changed; its record was preserved')
            _atomic(marker, dict(record, state='idle', updated_utc=datetime.now(timezone.utc).isoformat()))


def other_local_sessions():
    """Ownerless legacy files require absence of other possible Python GUIs."""
    import psutil
    rows = []
    for process in psutil.process_iter(['pid', 'name', 'create_time']):
        if process.pid == os.getpid():
            continue
        info = process.info
        name = (info.get('name') or '').casefold()
        if name.startswith('python') or 'darsia' in name:
            rows.append(dict(pid=process.pid, process_create_time=info.get('create_time'), name=name))
    return rows


def recover_legacy(folder, machine, *, confirmed, session='gui-recovery', session_reader=None):
    """GUI-only recovery; no active kernel holder or possible old GUI is bypassed."""
    if confirmed is not True:
        raise ValueError('Confirm that all other DarSIA/Python sessions on this machine are closed')
    marker, guard = _paths(folder, machine)
    with _kernel_guard(guard):
        previous, raw = _read(marker)
        if previous != 'legacy':
            return dict(recovered=False, reason='No legacy admission lock needs recovery')
        others = (session_reader or other_local_sessions)()
        if others:
            detail = ', '.join(str(row['pid']) for row in others)
            raise ValueError('Other local Python/DarSIA processes are still present: '+detail+
                             '. Close the other sessions before recovering this ownerless lock.')
        # The marker stays present throughout: old open("x") clients cannot
        # enter admission between this absence check and the atomic replacement.
        if marker.read_bytes() != raw:
            raise ValueError('Legacy admission lock changed during recovery; preserved')
        record = dict(format=FORMAT, state='idle', owner=_owner(machine, session),
                      updated_utc=datetime.now(timezone.utc).isoformat(),
                      recovered_legacy=dict(sha256=hashlib.sha256(raw).hexdigest(),
                                            other_local_sessions=[]))
        _atomic(marker, record)
        return dict(recovered=True, owner=record['owner'])
