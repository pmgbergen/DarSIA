"""Registry for analysis export format presets loaded from top-level ``[format.*.*]``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _convert_none

SUPPORTED_EXPORT_FORMATS = {"jpg", "png", "npz", "npy", "csv"}
NAME_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(stem|dd|hh|mm|ss)(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)


@dataclass
class ImageExportFormat:
    """Export format specification resolved from the format registry."""

    type: str
    name: str
    filename_pattern: str = "stem"
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
        return self.name


def _validate_name_mask(name: str, context: str) -> None:
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

    formats: dict[str, ImageExportFormat] = field(
        default_factory=dict,
        metadata={
            "name": "Format Entries",
            "help": "Named export format specifications (jpg, csv, npz, etc.).",
            "widget": "format_map",
            "toml_array_key": "format",
        },
    )

    def load(self, path: Path | list[Path]) -> "FormatRegistry":
        # Get the raw TOML to handle array-of-tables format: [[format]]
        try:
            import tomllib

            use_tomllib = True
        except ImportError:
            use_tomllib = False

        if isinstance(path, Path):
            paths = [path]
        else:
            paths = path

        self.formats = {}

        for p in paths:
            if not p.exists():
                continue

            with open(p, "rb" if use_tomllib else "r") as f:
                if use_tomllib:
                    data = tomllib.load(f)
                else:
                    import toml

                    data = toml.load(f)

            if "format" not in data:
                continue

            format_data = data["format"]

            # Handle array-of-tables format: [[format]]
            if not isinstance(format_data, list):
                raise ValueError(
                    "[format] section must be an array-of-tables "
                    "(use [[format]]), not nested tables."
                )

            format_list = format_data

            for idx, entry in enumerate(format_list):
                if not isinstance(entry, dict):
                    raise ValueError(f"[[format]] entry {idx} must be a table/dict.")

                # Extract required fields
                format_type = entry.get("type")
                if format_type is None:
                    raise ValueError(
                        f"[[format]] entry {idx} is missing required 'type' field."
                    )

                name = entry.get("name")
                if name is None:
                    raise ValueError(
                        f"[[format]] entry {idx} is missing required 'name'"
                        " field (the registry key)."
                    )

                _type = str(format_type).strip().lower()
                if _type not in SUPPORTED_EXPORT_FORMATS:
                    raise ValueError(
                        f"Unsupported format type '{format_type}' in [[format]] entry {idx}. "
                        f"Supported: {sorted(SUPPORTED_EXPORT_FORMATS)}"
                    )

                name = str(name).strip()
                if name in self.formats:
                    raise ValueError(
                        f"Format name '{name}' is duplicated. Names must be globally unique."
                    )

                spec = ImageExportFormat(type=_type, name=name)
                filename_pattern = str(entry.get("filename_pattern", "stem")).strip()
                _validate_name_mask(filename_pattern, f"[[format]] entry '{name}'")
                spec.filename_pattern = filename_pattern
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
                            f"quality in [[format]] entry '{name}' must be in [0, 100]."
                        )
                    compression = _convert_none(entry.get("compression"))
                    spec.compression = None if compression is None else int(compression)
                    if spec.compression is not None and not (
                        0 <= spec.compression <= 9
                    ):
                        raise ValueError(
                            f"compression in [[format]] entry '{name}' must be in [0, 9]."
                        )

                if _type in {"npz", "npy", "csv"}:
                    dtype = _convert_none(entry.get("dtype"))
                    spec.dtype = None if dtype is None else str(dtype)

                if _type == "csv":
                    spec.delimiter = str(entry.get("delimiter", ","))
                    header = _convert_none(entry.get("header", "none"))
                    spec.header = None if header is None else str(header)
                    spec.float_format = str(entry.get("float_format", "{:.2e}"))

                self.formats[spec.name] = spec

        return self

    def keys(self) -> list[str]:
        return sorted(self.formats.keys())

    def resolve(self, keys: str | list[str]) -> list[ImageExportFormat]:
        if isinstance(keys, str):
            keys = [keys]
        specs: list[ImageExportFormat] = []
        for key in keys:
            if key not in self.formats:
                raise KeyError(
                    f"Format key '{key}' not found in format registry. "
                    f"Available keys: {sorted(self.formats.keys())}"
                )
            specs.append(self.formats[key])
        return specs

    def to_toml_dict(self) -> dict:
        """Serialize registry to TOML-compatible dict for save round-trips."""
        format_list = []
        for spec in sorted(self.formats.values(), key=lambda s: s.name):
            entry = {
                "type": spec.type,
                "name": spec.name,
                "filename_pattern": spec.filename_pattern,
            }
            if spec.resolution is not None:
                entry["resolution"] = list(spec.resolution)
            if spec.keep_ratio:
                entry["keep_ratio"] = True

            if spec.type in {"jpg", "png"}:
                if spec.dpi is not None:
                    entry["dpi"] = spec.dpi
                if spec.cmap is not None:
                    entry["cmap"] = spec.cmap
                if spec.quality is not None:
                    entry["quality"] = spec.quality
                if spec.compression is not None:
                    entry["compression"] = spec.compression

            if spec.type in {"npz", "npy", "csv"}:
                if spec.dtype is not None:
                    entry["dtype"] = spec.dtype

            if spec.type == "csv":
                if spec.delimiter != ",":
                    entry["delimiter"] = spec.delimiter
                if spec.header is not None:
                    entry["header"] = spec.header
                if spec.float_format != "{:.2e}":
                    entry["float_format"] = spec.float_format

            format_list.append(entry)

        return {"format": format_list}
