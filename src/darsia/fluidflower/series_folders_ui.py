"""Selected-series folder editing using the existing manifest and machine settings."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tomllib

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QFileDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .series_locations import _read, _path
from .series_creation import _relative_folder
from .series_method import _write_manifest, ManifestChangedError
from .series_paths import series_key, series_registry_path, _project_directory, _manifest_series
from .storage_paths import resolve_storage_path
from .store_map_ui import StoreSettingsDialog, _MetadataCall


def read_image_locations(root):
    """Read declared image roots and native folder settings without reading images."""
    root = Path(root)
    original, document = _read(root / 'series.toml')
    roots = document.get('roots') or {}
    aliases = {name: _path(root, value) for name, value in roots.items()
               if name == 'raw' or name == 'images' or name.startswith('images_')}
    if not aliases:
        # Older native projects can have folders without declared image aliases.
        from darsia.presets.workflows.config.utils import _get_section_from_toml, _config_path
        paths, runs = document.get('paths') or {}, document.get('runs') or {}
        common = resolve_storage_path(root / paths.get('common_config', 'config/common.toml'))
        folders = set()
        for run in runs.get('ids') or []:
            source = (runs.get('config_files') or {}).get(run, str(Path(paths.get('run_configs', 'config/runs')) / (run + '.toml')))
            selected = [p for p in (common, resolve_storage_path(_path(root, source))) if p.is_file()]
            section = _get_section_from_toml(selected, 'data')
            for value in [section.get('folder'), *(section.get('folders') or [])]:
                if value and value != 'No file chosen':
                    folders.add(str(resolve_storage_path(_config_path(value, selected))))
        aliases = {f'images_{index}': Path(value) for index, value in enumerate(sorted(folders), 1)}
    return dict(root=root, original=original, series_id=document['series_id'],
                images=[dict(alias=alias, folder=str(path)) for alias, path in aliases.items()])


def save_image_locations(snapshot, choices):
    """Relocate explicit image-root locators, retaining all original file identities."""
    root = snapshot['root']
    original, document = _read(root / 'series.toml')
    if original != snapshot['original']:
        prior = tomllib.loads(snapshot['original'].decode('utf-8-sig'))
        current = read_image_locations(root)
        source_paths = ('common_config', 'run_configs', 'merged_configs')
        same_sources = (current['images'] == snapshot['images']
            and prior.get('runs') == document.get('runs')
            and all((prior.get('paths') or {}).get(key) == (document.get('paths') or {}).get(key) for key in source_paths)
            and (prior.get('storage') or {}).get('path_aliases') == (document.get('storage') or {}).get('path_aliases'))
        if not same_sources:
            raise ManifestChangedError('Image locations changed; reload them before saving')
    if set(choices) != {row['alias'] for row in snapshot['images']}:
        raise ValueError('Image locations must retain the selected source roots')
    updated = deepcopy(document)
    for row in snapshot['images']:
        chosen = Path(choices[row['alias']]).expanduser()
        if not chosen.is_absolute() or not chosen.is_dir():
            raise ValueError('Choose an existing absolute image folder: ' + str(chosen))
        old = Path(row['folder'])
        if chosen != old:
            from .storage_paths import machine_stored_text
            updated.setdefault('storage', {}).setdefault('path_aliases', {})[
                machine_stored_text(old)] = machine_stored_text(chosen)
        updated.setdefault('roots', {})[row['alias']] = _relative_folder(chosen, root)
    from .storage_paths import _aliases, _resolve_with
    aliases = _aliases((updated.get('storage') or {}).get('path_aliases') or {}, root)
    rows = sorted(aliases.items(), key=lambda row:len(row[0]), reverse=True)
    for source in aliases:
        _resolve_with(source, rows)
    _write_manifest(root / 'series.toml', original, updated)
    from .storage_paths import refresh_storage_paths
    refresh_storage_paths(root)
    return read_image_locations(root)


def select_existing_setup(root, chosen):
    """Replace this computer's registration with an existing same-series setup."""
    root, chosen = Path(root), _project_directory(Path(chosen))
    expected = _manifest_series(root)
    selected = _manifest_series(chosen)
    if not expected or not selected or expected[0] != selected[0]:
        raise ValueError('Choose an existing DarSIA folder for the same series')
    registry = series_registry_path()
    original, document = _read(registry)
    updated = deepcopy(document)
    entries = [row for row in updated.get('series', []) if series_key(row['id']) == expected[0]]
    def registered(value):
        path = resolve_storage_path(value)
        return (path if path.is_absolute() else registry.parent / path).resolve()
    if len(entries) != 1 or registered(entries[0]['root']) != root.resolve():
        raise ManifestChangedError('The selected series registration changed; refresh Start')
    entries[0]['root'] = str(chosen)
    _write_manifest(registry, original, updated)
    from .storage_paths import refresh_storage_paths
    refresh_storage_paths(chosen)
    return chosen


def remove_series_registration(root):
    """Remove the selected local Start registration; all project files stay put."""
    from .storage_paths import same_storage_path
    registry = series_registry_path()
    original, document = _read(registry)
    updated = deepcopy(document)
    rows = updated.get('series') or []
    selected = [row for row in rows if same_storage_path(_path(registry.parent, row['root']), root)]
    if len(selected) != 1:
        raise ManifestChangedError('The selected registration changed; refresh Start')
    updated['series'] = [row for row in rows if row is not selected[0]]
    _write_manifest(registry, original, updated)
    from .storage_paths import refresh_storage_paths
    refresh_storage_paths(root)
    return series_key(selected[0]['id'])


class SeriesFoldersPanel(QWidget):
    folders_saved = Signal(str)
    setup_selected = Signal(str)
    back_requested = Signal()

    def __init__(self, root, parent=None, *, settings_factory=StoreSettingsDialog):
        super().__init__(parent)
        self.root, self.snapshot, self._thread = Path(root), None, None
        layout = QVBoxLayout(self)
        back = QPushButton('Back to series')
        back.clicked.connect(self.back_requested)
        layout.addWidget(back)
        layout.addWidget(QLabel('Folders for the selected series'))
        self.setup = QLineEdit(str(self.root))
        choose = QPushButton('Choose existing DarSIA folder…')
        choose.clicked.connect(self._choose_setup)
        form = QFormLayout()
        form.addRow('DarSIA setup folder', self.setup)
        form.addRow(choose)
        use = QPushButton('Use existing setup folder')
        use.clicked.connect(self._use_setup)
        form.addRow(use)
        layout.addLayout(form)
        note = QLabel('Folder changes move no files. Image-location changes retain saved image identities; different images still need their normal review.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.images = QTableWidget(0, 2)
        self.images.setHorizontalHeaderLabels(['Image source', 'Image folder'])
        self.images.horizontalHeader().setStretchLastSection(True)
        self.images.setMaximumHeight(170)
        layout.addWidget(self.images)
        row = QHBoxLayout()
        load, browse, save = QPushButton('Reload image locations'), QPushButton('Browse selected image folder…'), QPushButton('Save image locations')
        load.clicked.connect(self.load_images)
        browse.clicked.connect(self._browse_image)
        save.clicked.connect(self._save_images)
        for button in (load, browse, save):row.addWidget(button)
        layout.addLayout(row)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.settings = settings_factory(self.root, self)
        self.settings.setWindowFlags(Qt.WindowType.Widget)
        self.settings.setModal(False)
        for box in self.settings.findChildren(QDialogButtonBox):box.hide()
        self.settings.series_locations_saved.connect(self.folders_saved)
        self.settings.jobs_folder_saved.connect(self.folders_saved)
        self.settings.saved.connect(lambda _path:self.folders_saved.emit(str(self.root)))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.settings)
        layout.addWidget(scroll, 1)
        original_show = self.settings._show_locations
        def show_locations(value):
            original_show(value)
            if not self.settings._loaded and self.settings.path.is_file():
                QTimer.singleShot(0, self.settings.load)
        self.settings._show_locations = show_locations
        self.load_images()
        self.settings.load_locations()

    def _call(self, call, ready):
        if self._thread is not None:return
        task = _MetadataCall(call, self)
        self._thread = task
        def finished():
            self._thread = None
            if task.error:self.status.setText(task.error.strip().splitlines()[-1])
            else:ready(task.value)
            task.deleteLater()
        task.finished.connect(finished)
        task.start()

    def load_images(self):
        self._call(lambda:read_image_locations(self.root), self._show_images)

    def _show_images(self, value):
        self.snapshot = value
        self.images.setRowCount(len(value['images']))
        for index, row in enumerate(value['images']):
            label = QTableWidgetItem(row['alias'])
            label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.images.setItem(index, 0, label)
            self.images.setItem(index, 1, QTableWidgetItem(row['folder']))
        self.status.setText('Saved image locations loaded. No image files were read.')

    def _browse_image(self):
        row = self.images.currentRow()
        if row < 0:return
        chosen = QFileDialog.getExistingDirectory(self, 'Choose image folder', self.images.item(row, 1).text())
        if chosen:self.images.item(row, 1).setText(chosen)

    def _save_images(self):
        if self.snapshot is None:return
        choices = {self.images.item(i, 0).text():self.images.item(i, 1).text() for i in range(self.images.rowCount())}
        def done(value):
            self._show_images(value)
            self.status.setText('Image locations saved. No image files moved or approved.')
            self.folders_saved.emit(str(self.root))
        self._call(lambda:save_image_locations(self.snapshot, choices), done)

    def _choose_setup(self):
        chosen = QFileDialog.getExistingDirectory(self, 'Choose existing DarSIA setup', self.setup.text())
        if chosen:self.setup.setText(chosen)

    def _use_setup(self):
        chosen = self.setup.text()
        self._call(lambda:select_existing_setup(self.root, chosen), lambda root:self.setup_selected.emit(str(root)))

    def can_close(self):
        return self._thread is None and self.settings._thread is None
