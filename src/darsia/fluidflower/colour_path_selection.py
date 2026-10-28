"""Explicit colour-path sampling choice in the existing series manifest."""
from copy import deepcopy
from pathlib import Path
import tomllib

from .series_method import ManifestChangedError, _manifest_bytes, _write_manifest


def save_colour_path_selection(root, count, *, expected_bytes, context_current=None):
    """Use the native lossless compare-and-swap writer; retain every image choice."""
    from .pipeline.current_optical_preparation import colour_path_image_count
    from .series_calibration_approval import require_unlocked

    path = Path(root) / "series.toml"
    original = path.read_bytes()
    if original != expected_bytes:
        raise ManifestChangedError("The series changed; reload before saving colour-path selection.")
    document = tomllib.loads(original.decode("utf-8-sig"))
    require_unlocked(document)
    updated = deepcopy(document)
    table = updated.setdefault("image_selection", {})
    if not isinstance(table, dict):
        raise ValueError("The image-selection section must be a table.")
    choice = table.setdefault("color_paths", {})
    if not isinstance(choice, dict):
        raise ValueError("The colour-path selection must be a table.")
    choice["max_images_per_run"] = count
    colour_path_image_count(updated)
    raw, _ = _manifest_bytes(original, updated)
    _write_manifest(path, original, updated, context_current=context_current)
    return raw
