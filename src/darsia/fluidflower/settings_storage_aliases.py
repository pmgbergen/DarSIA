"""Show configured storage moves in native settings without rewriting their sources."""
from copy import deepcopy
from pathlib import Path
import tomllib

from .series_method import _same_values


def _authored_value(original, shown, current):
    """Keep unchanged authored locators; carry actual edits through the normal sync."""
    if _same_values(shown, current):
        return deepcopy(original)
    if isinstance(original, dict) and isinstance(shown, dict) and isinstance(current, dict):
        keys = dict(zip(shown, original))
        result = {}
        for key, value in current.items():
            authored = keys.get(key, key)
            result[authored] = (_authored_value(original[authored], shown[key], value)
                                if key in keys else deepcopy(value))
        return result
    if isinstance(original, list) and isinstance(shown, list) and isinstance(current, list):
        result = []
        for index, value in enumerate(current):
            result.append(_authored_value(original[index], shown[index], value)
                          if index < len(shown) else deepcopy(value))
        return result
    return deepcopy(current)


def _store_shared_paths(config):
    """Give paths below this computer's shared folders their shared spelling.

    Applies only on a computer with a machine path map
    (``storage_paths.machine_stored_text``), so a value browsed there, such as
    a folder under /Volumes/<share>, is saved as the path every computer reads.
    Every other value is kept as the form gave it.
    """
    from darsia.presets.workflows.config.utils import _map_strings
    from .storage_paths import machine_path_map, machine_stored_text

    if not isinstance(config, dict) or not machine_path_map():
        return
    for section, value in list(config.items()):
        config[section] = _map_strings(value, machine_stored_text)


def display_storage_aliases(window):
    """Adapt the existing native settings reader, sync and save only."""
    factory = window.settings_factory
    if getattr(factory, '_storage_alias_display', None) is not None:
        return
    get_value, sync, save = factory.get_value, factory._sync_settings_inputs_to_config_dict, factory.save_settings
    state = dict(fields={}, syncing=False, dictionary=window.config_dict)
    factory._storage_alias_display = state

    def value(dictionary, key):
        if state['dictionary'] is not window.config_dict:
            state['fields'].clear()
            state['dictionary'] = window.config_dict
        current = get_value(dictionary, key)
        if state['syncing'] or dictionary is not window.config_dict:
            return current
        from darsia.presets.workflows.config.utils import _with_storage_aliases
        shown = _with_storage_aliases(current)
        if key in {'protocols.imaging', 'protocols.injection',
                   'protocols.pressure_temperature', 'protocols.blacklist',
                   'protocols.overflow'} and shown and window.config_file:
            from darsia.presets.workflows.config.protocols import _parse_protocol_value

            def display_reference(reference):
                parsed = _parse_protocol_value(reference, Path(window.config_file))
                return ([str(parsed[0]), parsed[1]] if isinstance(parsed, tuple)
                        else str(parsed))

            shown = ({folder: display_reference(reference)
                      for folder, reference in shown.items()}
                     if isinstance(shown, dict) else display_reference(shown))
        if not _same_values(current, shown):
            state['fields'][key] = (deepcopy(current), deepcopy(shown))
        return shown

    def synchronize(*args, **kwargs):
        if state['dictionary'] is not window.config_dict:
            state['fields'].clear()
            state['dictionary'] = window.config_dict
        records = dict(state['fields'])
        state['syncing'] = True
        try:
            result = sync(*args, **kwargs)
            for key, (original, shown) in records.items():
                current = get_value(window.config_dict, key)
                if current is not None:
                    factory.set_value(window.config_dict, key, _authored_value(original, shown, current))
            _store_shared_paths(window.config_dict)
            return result
        finally:
            state['syncing'] = False

    def save_settings(*args, **kwargs):
        if state['fields']:
            factory._sync_settings_inputs_to_config_dict()
            try:
                before = tomllib.loads(Path(window.config_file).read_text(encoding='utf-8-sig'))
            except (OSError, ValueError, TypeError):
                before = None
            if before is not None and _same_values(before, window.config_dict):
                window.print_log('No settings changed; the configuration file is unchanged.')
                return None
        return save(*args, **kwargs)

    value.__wrapped__ = get_value
    synchronize.__wrapped__ = sync
    save_settings.__wrapped__ = save
    factory.get_value = value
    factory._sync_settings_inputs_to_config_dict = synchronize
    factory.save_settings = save_settings
    controller = getattr(window, 'config_controller', None)
    load = getattr(controller, 'load_config', None)
    if callable(load):
        def load_config(*args, **kwargs):
            state['fields'].clear()
            return load(*args, **kwargs)
        load_config.__wrapped__ = load
        controller.load_config = load_config
