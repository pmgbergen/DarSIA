"""Shared image export helper for workflow analyses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

import darsia
from darsia.presets.workflows.config.format_registry import (
    ImageExportFormat,
    SUPPORTED_EXPORT_FORMATS,
)

logger = logging.getLogger(__name__)


def _parse_numpy_dtype(dtype_value: str | None):
    if dtype_value is None:
        return None
    value = dtype_value.strip()
    if not value or value.lower() == "none":
        return None
    if value.startswith("np."):
        value = value[3:]
    return np.dtype(value)


def _format_to_numpy_fmt(float_format: str) -> str:
    fmt = float_format.strip()
    if fmt.startswith("{:") and fmt.endswith("}"):
        return "%" + fmt[2:-1]
    return fmt


class ImageExportFormats:
    """Resolve analysis export format identifiers and write images to disk."""

    def __init__(self, config: Any, format_keys: list[str] | None):
        self.config = config
        self._registry = getattr(config, "format_registry", None)
        self.formats = self._resolve_formats(format_keys or [])

    @classmethod
    def from_analysis_config(
        cls,
        config: Any,
        *,
        fallback_formats: list[str] | None = None,
    ) -> "ImageExportFormats":
        format_keys = None
        if getattr(config, "analysis", None) is not None:
            format_keys = getattr(config.analysis, "formats", None)
        if format_keys is None:
            format_keys = fallback_formats
        return cls(config, format_keys)

    def _resolve_formats(self, format_keys: list[str]) -> list[ImageExportFormat]:
        resolved: list[ImageExportFormat] = []
        seen: set[str] = set()
        available_registry = set(self._registry.keys()) if self._registry else set()
        for raw_key in format_keys:
            key = raw_key.strip()
            if not key:
                continue
            if key in available_registry:
                specs = self._registry.resolve(key)
            else:
                format_type = key.lower()
                if format_type not in SUPPORTED_EXPORT_FORMATS:
                    raise ValueError(
                        f"Unsupported export format key '{key}'. "
                        f"Known format types: {sorted(SUPPORTED_EXPORT_FORMATS)}."
                    )
                specs = [ImageExportFormat(type=format_type, identifier=format_type)]
            for spec in specs:
                if spec.folder_name in seen:
                    continue
                seen.add(spec.folder_name)
                resolved.append(spec)
        return resolved

    def _load_color_path_cmap(self, cmap_name: str):
        # Supported pattern: color_path.from_facies.5 or color_path.from_labels.5
        parts = cmap_name.split(".")
        if len(parts) != 3:
            return None
        basis = parts[1]
        try:
            label = int(parts[2])
        except ValueError:
            return None
        if basis not in {"from_facies", "from_labels"}:
            return None
        if getattr(self.config, "data", None) is None:
            return None
        folder = self.config.data.results / "calibration" / "color_paths" / basis
        path = folder / f"color_path_{label}.json"
        if not path.exists():
            return None
        return darsia.ColorPath.load(path).get_color_map()

    def _resolve_cmap(self, cmap_name: str | None):
        if cmap_name is None:
            return None
        if cmap_name.startswith("matplotlib."):
            import matplotlib.cm as cm

            return cm.get_cmap(cmap_name.split(".", 1)[1])
        if cmap_name.startswith("color_path."):
            return self._load_color_path_cmap(cmap_name)
        return None

    def _prepare_image(self, image: darsia.Image, spec: ImageExportFormat) -> darsia.Image:
        prepared = image.copy()

        if spec.resolution is not None:
            rows, cols = spec.resolution
            if spec.keep_ratio:
                base_rows, base_cols = prepared.img.shape[:2]
                cols = max(1, int(round(base_cols * rows / max(base_rows, 1))))
            prepared = darsia.resize(prepared, shape=(rows, cols))

        dtype = _parse_numpy_dtype(spec.dtype)
        if dtype is not None:
            prepared = prepared.astype(dtype.type)

        return prepared

    def export_image(
        self,
        image: darsia.Image,
        folder: Path,
        stem: str,
        *,
        supported_types: set[str] | None = None,
        subfolder: str | Path | None = None,
        jpg_quality: int = 50,
        png_compression: int = 6,
    ) -> list[Path]:
        written: list[Path] = []
        for spec in self.formats:
            if supported_types is not None and spec.type not in supported_types:
                continue

            prepared = self._prepare_image(image, spec)
            out_dir = folder / spec.folder_name
            if subfolder is not None:
                out_dir = out_dir / Path(subfolder)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{stem}.{spec.type}"

            if spec.type == "npz":
                prepared.save(path)
            elif spec.type == "npy":
                np.save(path, prepared.img)
            elif spec.type == "csv":
                if not prepared.scalar:
                    logger.info(
                        "Skipping csv export for non-scalar image '%s' in format '%s'.",
                        stem,
                        spec.identifier,
                    )
                    continue
                if hasattr(prepared, "to_csv"):
                    prepared.to_csv(
                        path,
                        delimiter=spec.delimiter,
                        header=spec.header,
                        float_format=spec.float_format,
                    )
                else:
                    header = "" if spec.header in (None, "none") else str(spec.header)
                    np.savetxt(
                        path,
                        np.asarray(prepared.img),
                        delimiter=spec.delimiter,
                        header=header,
                        comments="",
                        fmt=_format_to_numpy_fmt(spec.float_format),
                    )
            elif spec.type in {"jpg", "png"}:
                kwargs: dict[str, Any] = {}
                if spec.type == "jpg":
                    kwargs["quality"] = jpg_quality
                if spec.type == "png":
                    kwargs["compression"] = png_compression
                if prepared.scalar:
                    cmap = self._resolve_cmap(spec.cmap)
                    if cmap is not None:
                        kwargs["cmap"] = cmap
                prepared.write(path, **kwargs)
            else:
                raise ValueError(f"Unsupported export format type '{spec.type}'.")

            written.append(path)
        return written

