"""Introspect FluidFlowerConfig section dataclasses to derive GUI widget schema."""

import types
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from darsia.presets.workflows.config.analysis import AnalysisConfig
from darsia.presets.workflows.config.calibration import CalibrationConfig
from darsia.presets.workflows.config.corrections import CorrectionsConfig
from darsia.presets.workflows.config.data import DataConfig
from darsia.presets.workflows.config.depth import DepthConfig
from darsia.presets.workflows.config.facies import FaciesConfig
from darsia.presets.workflows.config.helper import HelperConfig
from darsia.presets.workflows.config.image_porosity import ImagePorosityConfig
from darsia.presets.workflows.config.labeling import LabelingConfig
from darsia.presets.workflows.config.protocol import ProtocolConfig
from darsia.presets.workflows.config.restoration import RestorationConfig
from darsia.presets.workflows.config.rig import RigConfig

# Mapping from section name to its config dataclass
SECTION_TO_DATACLASS = {
    "analysis": AnalysisConfig,
    "calibration": CalibrationConfig,
    "corrections": CorrectionsConfig,
    "data": DataConfig,
    "depth": DepthConfig,
    "facies": FaciesConfig,
    "helper": HelperConfig,
    "image_porosity": ImagePorosityConfig,
    "labeling": LabelingConfig,
    "protocol": ProtocolConfig,
    "restoration": RestorationConfig,
    "rig": RigConfig,
}


def _unwrap_optional(field_type: Any) -> Any:
    """Unwrap Optional[X] (Union[X, None]) to X, leaving list/tuple alone.

    Args:
        field_type: The type annotation to unwrap.

    Returns:
        The unwrapped type, or the original if not Optional.
    """
    origin = get_origin(field_type)
    # Only unwrap actual Union types (Union[X, None] or X | None), not list/tuple
    if origin is Union or origin is types.UnionType:
        args = get_args(field_type)
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return field_type


def _infer_widget_type(field_type: Any, metadata: dict) -> str:
    """Infer widget type from field type annotation and metadata.

    Args:
        field_type: The dataclass field's type annotation.
        metadata: The field's metadata dict.

    Returns:
        Widget type string: "bool", "file", "folder", "string", "int", "float", "list", etc.
    """
    # Check for explicit widget override in metadata
    if "widget" in metadata:
        return metadata["widget"]

    # Unwrap Optional[X] to X, but leave list/tuple alone
    field_type = _unwrap_optional(field_type)

    # Check for list/tuple types
    origin = get_origin(field_type)
    if origin in (list, tuple):
        return "list"

    # Map Python scalar types to widget types
    if field_type is bool:
        return "bool"
    elif field_type is Path or (
        isinstance(field_type, type) and issubclass(field_type, Path)
    ):
        return "file"  # Default for Path; can be overridden via metadata["widget"]
    elif field_type is int:
        return "int"
    elif field_type is float:
        return "float"
    elif field_type is str:
        return "string"
    else:
        # For other complex types (dict, etc.), use generic string input
        return "string"


def _field_default(field: Any) -> Any:
    """Resolve a dataclass field's default value, or None if not usable as a pre-fill.

    Args:
        field: A dataclass field object.

    Returns:
        The field's default value if usable for pre-filling a widget, or None otherwise.
        Skips empty collections ([], {}, Path()) and nested dataclass instances.
    """
    if field.default is not MISSING:
        default = field.default
    elif field.default_factory is not MISSING:
        try:
            default = field.default_factory()
        except Exception:
            return None
    else:
        return None

    # Skip empty Path() (equivalent to ".") for file/folder choosers.
    if isinstance(default, Path) and default == Path():
        return None

    # Skip empty collections — not meaningful pre-fills for list/dict widgets.
    if isinstance(default, (list, dict)) and len(default) == 0:
        return None

    # Skip nested dataclass instances (e.g., CorrectionsConfig.type:
    # TypeCorrectionConfig) — they don't round-trip cleanly through scalar widgets.
    if is_dataclass(default) and not isinstance(default, type):
        return None

    return default


def _infer_list_type(field_type: Any) -> str:
    """Infer the element type of a list/tuple field for the UI label.

    Args:
        field_type: The list/tuple type annotation (e.g., list[str], tuple[int, int]).

    Returns:
        Widget type string for the element type (e.g., "string", "int").
    """
    args = get_args(field_type)
    if not args:
        return "string"  # Fallback for bare list/tuple
    inner = _unwrap_optional(args[0])
    return _infer_widget_type(inner, {})


def _build_fields(dataclass_type: type, key_prefix: str) -> list[dict[str, Any]]:
    """Build field schema for a dataclass type, recursing into Optional[dataclass] fields.

    Args:
        dataclass_type: The dataclass type to introspect.
        key_prefix: The TOML key prefix for this level (e.g., "corrections" or
            "corrections.resize").

    Returns:
        List of setting dicts with key, type, help, link, options, fields (for groups),
        list_type (for lists), default, etc.
    """
    settings = []
    for field in fields(dataclass_type):
        # Skip fields marked as hidden (outputs, derived fields)
        if field.metadata.get("hidden", False):
            continue

        key = f"{key_prefix}.{field.name}"
        inner_type = _unwrap_optional(field.type)

        # Check if this is an Optional[dataclass] field — render as a group
        if is_dataclass(inner_type):
            setting_dict = {
                "key": key,
                "type": "group",
                "help": field.metadata.get("help", None),
                "link": field.metadata.get("link", None),
                "active_list_key": field.metadata.get("active_list_key", None),
                "fields": _build_fields(inner_type, key),
            }
        else:
            widget_type = _infer_widget_type(field.type, field.metadata)
            setting_dict = {
                "key": key,
                "type": widget_type,
                "help": field.metadata.get("help", None),
                "link": field.metadata.get("link", None),
                "options": field.metadata.get("options", None),
                "default": _field_default(field),
            }

            # For list/tuple types, add the element type label
            if widget_type == "list":
                setting_dict["list_type"] = _infer_list_type(field.type)

        # Remove None values to keep the dict clean
        setting_dict = {k: v for k, v in setting_dict.items() if v is not None}
        settings.append(setting_dict)

    return settings


def get_section_fields(section: str) -> list[dict[str, Any]] | None:
    """Get GUI widget schema for all fields in a config section.

    Args:
        section: Section name (e.g., "rig", "depth", "calibration")

    Returns:
        List of setting dicts with key, type, help, link, options, fields (for groups),
        list_type (for lists), default, etc.
        Returns None if section is not recognized.
    """
    if section not in SECTION_TO_DATACLASS:
        return None

    dataclass_type = SECTION_TO_DATACLASS[section]
    if not is_dataclass(dataclass_type):
        return None

    toml_section = TOML_SECTION_ALIASES.get(section, section)
    return _build_fields(dataclass_type, toml_section)
