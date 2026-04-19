"""Registry for analysis export format presets loaded from top-level ``[format.*.*]``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _convert_none, _get_section_from_toml

SUPPORTED_EXPORT_FORMATS = {"jpg", "png", "npz", "npy", "csv"}
REMOVED_EXPORT_NAMES = {"name_stem", "name_time_hh:mm:ss"}
NAME_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(stem|dd|hh|mm|ss)(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)


@dataclass
class ImageExportFormat:
    """Export format specification resolved from the format registry."""

    type: str
    identifier: str
    name: str = "stem"
    resolution: tuple[int, int] | None = None
    dpi: int | None = None
    cmap: str | None = None
    keep_ratio: bool = False
    dtype: str | None = None
    quality: int | None = None
    compression: int | None = None
    delimiter: str = ","
    header: str | None = None
    float_format: str = "{:.2e}"

    @property
    def folder_name(self) -> str:
        return self.identifier


def _validate_name_mask(name: str, context: str) -> None:
    lower_name = name.lower()
    if lower_name in REMOVED_EXPORT_NAMES:
        raise ValueError(
            f"Unsupported name option '{name}' for {context}. "
            f"Removed options: {sorted(REMOVED_EXPORT_NAMES)}."
        )
    if NAME_IDENTIFIER_PATTERN.search(name) is None:
        raise ValueError(
            f"Unsupported name option '{name}' for {context}. "
            "Name must contain at least one identifier token: stem, hh, mm, ss, or dd."
        )


def _parse_resolution(value) -> tuple[int, int] | None:
    value = _convert_none(value)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("resolution must be a list [rows, cols] or 'None'.")
    rows = int(value[0])
    cols = int(value[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("resolution entries must be positive integers.")
    return rows, cols


@dataclass
class FormatRegistry:
    """Registry for named export format presets."""

    _registry: dict[str, ImageExportFormat] = field(default_factory=dict)

    def load(self, path: Path | list[Path]) -> "FormatRegistry":
        sec = _get_section_from_toml(path, "format")
        self._registry = {}
        for format_type, entries in sec.items():
            _type = str(format_type).strip().lower()
            if _type not in SUPPORTED_EXPORT_FORMATS:
                raise ValueError(
                    f"Unsupported format type '{format_type}'. "
                    f"Supported: {sorted(SUPPORTED_EXPORT_FORMATS)}"
                )
            if not isinstance(entries, dict):
                raise ValueError(f"[format.{_type}] must be a table of named entries.")
            for identifier, entry in entries.items():
                if identifier in self._registry:
                    raise ValueError(
                        f"Format identifier '{identifier}' is duplicated across "
                        "format sections. Identifiers must be globally unique."
                    )
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"[format.{_type}.{identifier}] must be a table/dict."
                    )

                spec = ImageExportFormat(type=_type, identifier=str(identifier))
                spec.name = str(entry.get("name", "stem")).strip()
                _validate_name_mask(spec.name, f"[format.{_type}.{identifier}]")
                spec.resolution = _parse_resolution(entry.get("resolution"))
                spec.keep_ratio = bool(entry.get("keep_ratio", False))

                if _type in {"jpg", "png"}:
                    dpi = _convert_none(entry.get("dpi"))
                    spec.dpi = None if dpi is None else int(dpi)
                    cmap = _convert_none(entry.get("cmap"))
                    spec.cmap = None if cmap is None else str(cmap)
                    quality = _convert_none(entry.get("quality"))
                    spec.quality = None if quality is None else int(quality)
                    if spec.quality is not None and not (0 <= spec.quality <= 100):
                        raise ValueError(
                            f"quality in [format.{_type}.{identifier}] must be in [0, 100]."
                        )
                    compression = _convert_none(entry.get("compression"))
                    spec.compression = None if compression is None else int(compression)
                    if spec.compression is not None and not (
                        0 <= spec.compression <= 9
                    ):
                        raise ValueError(
                            f"compression in [format.{_type}.{identifier}] must be in [0, 9]."
                        )

                if _type in {"npz", "npy", "csv"}:
                    dtype = _convert_none(entry.get("dtype"))
                    spec.dtype = None if dtype is None else str(dtype)

                if _type == "csv":
                    spec.delimiter = str(entry.get("delimiter", ","))
                    header = _convert_none(entry.get("header", "none"))
                    spec.header = None if header is None else str(header)
                    spec.float_format = str(entry.get("float_format", "{:.2e}"))

                self._registry[spec.identifier] = spec
        return self

    def keys(self) -> list[str]:
        return sorted(self._registry.keys())

    def resolve(self, keys: str | list[str]) -> list[ImageExportFormat]:
        if isinstance(keys, str):
            keys = [keys]
        specs: list[ImageExportFormat] = []
        for key in keys:
            if key not in self._registry:
                raise KeyError(
                    f"Format key '{key}' not found in format registry. "
                    f"Available keys: {sorted(self._registry.keys())}"
                )
            specs.append(self._registry[key])
        return specs
