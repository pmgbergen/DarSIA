"""Finite metadata-only series reads; immutable delivery, GUI-owned application."""
from dataclasses import dataclass, replace
from pathlib import Path
import pickle
import tomllib

from PySide6.QtCore import QObject, QThread, Signal, Slot


@dataclass(frozen=True)
class SeriesSelectionPacket:
    series_id: str
    root: Path | None
    payload: bytes

    def unpack(self):
        # These bytes are produced locally below, never read from an input file.
        return pickle.loads(self.payload)


def read_series_selection(series_id, root, label, status_rows=(), guard_path=''):
    """Read no images, scientific arrays or historical job directories."""
    from .native_darsia_app import run_configurations, _file_stamp
    from .series_native_context import read_series_native_binding
    from .series_reference_selection import reference_choice
    from .series_method import _selection, MethodSelection
    from .series_geometry import geometry_package_root
    from .setup_display import completed_current_native_setup_export
    from .row_scope import load_scope_context, _path
    from .rig_display import read_rig_display
    from .workflow_row_status import load_status_context, row_status
    from .model_index_names import model_reference_label
    from .published_dataset.layout import MANIFEST
    import toml

    root = Path(root) if root is not None else None
    native = read_series_native_binding(root)
    raw = (root / 'series.toml').read_bytes() if root is not None else None
    manifest = tomllib.loads(raw.decode('utf-8-sig')) if raw else {}
    choices = tuple(run_configurations(root)) if root is not None else ()
    reference = reference_choice(root, choices) if root is not None else None
    method = _selection(manifest) if raw else MethodSelection()
    configs = {}
    for _run, path, _note in choices:
        if path is None:
            continue
        try:
            before = _file_stamp(path)
            values = toml.loads(Path(path).read_text(encoding='utf-8'))
            after = _file_stamp(path)
            if before is None or before != after:
                raise ValueError('Configuration changed during series loading: ' + str(path))
            configs[str(path)] = (values, after, '')
        except (OSError, ValueError) as error:
            configs[str(path)] = (None, None, str(error))
    base = load_scope_context(root, series_id, series_label=label)
    scopes, rigs, statuses = {}, {}, {}
    entries = [('', None, '')] + list(choices)
    paths = manifest.get('paths') or {}
    run_files = (manifest.get('runs') or {}).get('config_files') or {}
    for run, path, _note in entries:
        run_path = (_path(root, run_files.get(run) or
            f"{paths.get('run_configs', 'config/runs')}/{run}.toml") if root and run else None)
        notes = list(base.notes)
        run_values = {}
        if run_path is not None:
            try:
                run_values = (configs[str(run_path)][0] if str(run_path) in configs
                    else tomllib.loads(run_path.read_text(encoding='utf-8-sig')))
            except (OSError, ValueError) as error:
                notes.append(str(error))
        values = configs.get(str(path), (None, None, ''))[0] or {}
        scope = replace(base, selected_run=run, run_path=run_path,
            merged_path=(_path(root, paths.get('merged_configs', 'config/merged')) / f'{run}.toml'
                if root and run else None), loaded_path=Path(path) if path else None,
            run=run_values or {}, loaded=values, notes=tuple(notes))
        scopes[run] = scope
        rigs[run] = read_rig_display(root, series_id, loaded_config=values if path else None)
        context = load_status_context(root, scope_context=scope, method_selection=method,
            selection_plan_reference=str(root / 'config/image_selection_plan.toml') if root else '')
        statuses[run] = (context, {key: row_status(context, *key) for key in status_rows})
    data = dict(native=native, raw=raw, choices=choices, reference=reference, method=method,
        configs=configs, scopes=scopes, rigs=rigs, statuses=statuses,
        completed=completed_current_native_setup_export(geometry_package_root(root)) if root else None,
        model_label=model_reference_label(root, method.model_version),
        dataset=bool(root and (root / MANIFEST).is_file()))
    if guard_path:
        before = _file_stamp(guard_path)
        guard_values = tomllib.loads(Path(guard_path).read_text(encoding='utf-8-sig'))
        if before is None or before != _file_stamp(guard_path):
            raise ValueError('The open configuration changed during series loading')
    else:
        guard_values = None
    data['guard'] = guard_path, guard_values
    return SeriesSelectionPacket(series_id, root, pickle.dumps(data))


class _Read(QObject):
    finished = Signal()

    def __init__(self, reader, args):
        super().__init__()
        self.reader, self.args = reader, args
        self.packet, self.error = None, ''

    @Slot()
    def run(self):
        try:
            self.packet = self.reader(*self.args)
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
        finally:
            self.finished.emit()


class SeriesSelectionLoader(QObject):
    ready = Signal(int, object, str)

    def __init__(self, parent=None, *, reader=read_series_selection):
        super().__init__(parent)
        self.reader, self.generation = reader, 0
        self._calls = {}

    def request(self, series_id, root, label, status_rows=(), guard_path=''):
        self.generation += 1
        generation = self.generation
        thread = QThread(self)
        worker = _Read(self.reader, (series_id, root, label, status_rows, guard_path))
        worker.moveToThread(thread)
        self._calls[generation] = thread, worker
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda: self._retire(generation))
        thread.start()
        return generation

    def _retire(self, generation):
        thread, worker = self._calls.pop(generation)
        packet, error = worker.packet, worker.error
        thread.deleteLater()
        self.ready.emit(generation, packet, error)

    def busy(self):
        return bool(self._calls)
