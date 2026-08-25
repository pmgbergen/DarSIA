"""Centralized registry for color embeddings."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import darsia
from darsia.signals.color import (
    ColorChannelEmbedding,
    ColorEmbedding,
    ColorEmbeddingBasis,
    ColorPathEmbedding,
    ColorRangeEmbedding,
    parse_color_embedding_basis,
)

from .restoration import RestorationConfig
from .roi_registry import RoiRegistry, _load_roi_key_list
from .utils import _convert_none, _validate_choice

if TYPE_CHECKING:
    from .data_registry import DataRegistry


def _parse_mode(value: str, *, context: str) -> darsia.ColorMode:
    try:
        return darsia.ColorMode(value.lower().strip())
    except Exception as exc:
        raise ValueError(
            f"Invalid {context}.mode '{value}'. Supported values are 'relative' and "
            "'absolute'."
        ) from exc


def parse_color_path_embedding(
    cfg: dict,
    embedding_id: str,
    color_root: Path | None,
    data: Path | None,
    data_registry: DataRegistry | None,
    roi_registry: RoiRegistry | None,
) -> ColorPathEmbedding:
    mode = _parse_mode(
        cfg.get("mode", "relative"), context=f"color.path.{embedding_id}"
    )
    basis = parse_color_embedding_basis(cfg.get("basis", "labels"))
    calibration_root = (
        Path(cfg["calibration_folder"])
        if "calibration_folder" in cfg
        else (color_root / embedding_id if color_root is not None else Path())
    )
    ignore_baseline_spectrum = _validate_choice(
        str(cfg.get("ignore_baseline_spectrum", "expanded")).strip(),
        allowed={"none", "baseline", "expanded"},
        context=f"color.path.{embedding_id}",
        key="ignore_baseline_spectrum",
    )
    histogram_weighting = _validate_choice(
        str(cfg.get("histogram_weighting", "threshold")).strip(),
        allowed={"threshold", "wls", "wls_sqrt", "wls_log"},
        context=f"color.path.{embedding_id}",
        key="histogram_weighting",
    )
    raw_calibration_mode = cfg.get("mode_calibration")
    if raw_calibration_mode is None:
        raw_calibration_mode = cfg.get("calibration_mode", "auto")
    calibration_mode = _validate_choice(
        str(raw_calibration_mode).strip(),
        allowed={"auto", "manual"},
        context=f"color.path.{embedding_id}",
        key="calibration_mode",
    )

    rois = _load_roi_key_list(
        cfg,
        "rois",
        context=f"color.path.{embedding_id}.rois",
        roi_registry=roi_registry,
        restricted=False,
    )

    embedding = ColorPathEmbedding(
        embedding_id=embedding_id,
        mode=mode,
        basis=basis,
        calibration_root=calibration_root,
        num_segments=int(cfg.get("num_segments", 1)),
        ignore_labels=list(cfg.get("ignore_labels", [])),
        resolution=int(cfg.get("resolution", 51)),
        threshold_baseline=float(cfg.get("threshold_baseline", 0.0)),
        threshold_calibration=float(cfg.get("threshold_calibration", 0.0)),
        reference_label=int(cfg.get("reference_label", 0)),
        rois=rois,
        ignore_baseline_spectrum=ignore_baseline_spectrum,
        histogram_weighting=histogram_weighting,
        calibration_mode=calibration_mode,
    )
    embedding.baseline_data = (
        data_registry.resolve(cfg["baseline"])
        if data_registry and "baseline" in cfg
        else None
    )
    embedding.data = (
        data_registry.resolve(cfg["data"]) if data_registry and "data" in cfg else None
    )
    return embedding


def parse_color_range_embedding(
    cfg: dict,
    embedding_id: str,
    color_root: Path | None,
    data: Path | None,
    data_registry: DataRegistry | None,
    roi_registry: RoiRegistry | None,
) -> ColorRangeEmbedding:
    mode = _parse_mode(
        cfg.get("mode", "absolute"), context=f"color.range.{embedding_id}"
    )
    basis = parse_color_embedding_basis(cfg.get("basis", "global"))
    raw_range = cfg.get("range")
    if not isinstance(raw_range, list) or len(raw_range) != 3:
        raise ValueError(
            f"color.range.{embedding_id}.range must be a list of 3 [min,max] bounds."
        )
    ranges: list[tuple[float | None, float | None]] = []
    for i, bound in enumerate(raw_range):
        if not isinstance(bound, list) or len(bound) != 2:
            raise ValueError(
                f"color.range.{embedding_id}.range[{i}] must have two entries."
            )
        low = _convert_none(bound[0])
        high = _convert_none(bound[1])
        ranges.append(
            (
                None if low is None else float(low),
                None if high is None else float(high),
            )
        )
    calibration_root = (
        Path(cfg["calibration_folder"])
        if "calibration_folder" in cfg
        else (color_root / embedding_id if color_root is not None else Path())
    )
    if "color_space" not in cfg:
        raise ValueError(f"color.range.{embedding_id}.color_space is required.")
    if "restoration" in cfg:
        if not isinstance(cfg["restoration"], dict):
            raise ValueError(
                f"color.channel.{embedding_id}.restoration must be a table."
            )
        from .restoration import RestorationConfig

        restoration_config = RestorationConfig().load(cfg["restoration"])
    embedding = ColorRangeEmbedding(
        embedding_id=embedding_id,
        mode=mode,
        basis=basis,
        calibration_root=calibration_root,
        color_space=str(cfg["color_space"]).upper().strip(),
        ranges=ranges,
        restoration_config=restoration_config if "restoration" in cfg else None,
    )
    return embedding


def parse_color_channel_embedding(
    cfg: dict,
    embedding_id: str,
    color_root: Path | None,
    data: Path | None,
    data_registry: DataRegistry | None,
    roi_registry: RoiRegistry | None,
) -> ColorChannelEmbedding:
    mode = _parse_mode(
        cfg.get("mode", "absolute"), context=f"color.channel.{embedding_id}"
    )
    basis = parse_color_embedding_basis(cfg.get("basis", "global"))
    if basis != ColorEmbeddingBasis.GLOBAL:
        raise NotImplementedError(
            "color.channel.<id> currently only supports basis='global'."
        )
    calibration_root = (
        Path(cfg["calibration_folder"])
        if "calibration_folder" in cfg
        else (color_root / embedding_id if color_root is not None else Path())
    )
    for key in ["color_space", "channel"]:
        if key not in cfg:
            raise ValueError(f"color.channel.{embedding_id}.{key} is required.")
    mask_embedding: ColorRangeEmbedding | None = None
    if "mask" in cfg:
        if not isinstance(cfg["mask"], dict):
            raise ValueError(f"color.channel.{embedding_id}.mask must be a table.")
        mask_embedding = parse_color_range_embedding(
            cfg=cfg["mask"],
            embedding_id=f"{embedding_id}_mask",
            color_root=calibration_root,
            data=data,
            data_registry=data_registry,
            roi_registry=roi_registry,
        )
    if "restoration" in cfg:
        if not isinstance(cfg["restoration"], dict):
            raise ValueError(
                f"color.channel.{embedding_id}.restoration must be a table."
            )
        from .restoration import RestorationConfig

        restoration_config = RestorationConfig().load(cfg["restoration"])
    embedding = ColorChannelEmbedding(
        embedding_id=embedding_id,
        mode=mode,
        basis=basis,
        calibration_root=calibration_root,
        color_space=str(cfg["color_space"]).upper().strip(),
        channel=str(cfg["channel"]).lower().strip(),
        mask_embedding=mask_embedding,
        restoration_config=restoration_config if "restoration" in cfg else None,
    )
    return embedding


SUPPORTED_COLOR_TYPES = {"path", "range", "channel"}


@dataclass
class ColorEmbeddingRegistry:
    """Registry of configured color embeddings loaded from [[color]] array-of-tables."""

    _embeddings: dict[str, ColorEmbedding] = field(default_factory=dict, repr=False)

    def load(
        self,
        path: Path | list[Path],
        *,
        data: Path | None,
        results: Path | None,
        data_registry: DataRegistry | None = None,
        roi_registry: "RoiRegistry | None" = None,
    ) -> "ColorEmbeddingRegistry":
        """Load color embeddings from [[color_path]], [[color_range]], [[color_channel]]
        arrays in TOML.

        Hand-parses TOML (like FormatRegistry and RoiRegistry) since array-of-tables
        is not supported by the generic _get_section_from_toml helper.

        Args:
            path: Path or list of Paths to TOML config file(s).
            data: Data folder path.
            results: Results folder path.
            data_registry: Optional DataRegistry for resolving data references.
            roi_registry: Optional RoiRegistry for validating ROI references.

        Returns:
            self

        Raises:
            ValueError: If any array section is not an array-of-tables, if any
                entry is missing required field (name), or if names are duplicated.
        """
        paths = [path] if isinstance(path, Path) else path
        self._embeddings = {}
        color_root = results / "calibration" / "color" if results is not None else None

        for p in paths:
            if not p.exists():
                continue

            with open(p, "rb") as f:
                toml_data = tomllib.load(f)

            # Load from three separate arrays: color_path, color_range, color_channel
            for array_name, parser in (
                ("color_path", parse_color_path_embedding),
                ("color_range", parse_color_range_embedding),
                ("color_channel", parse_color_channel_embedding),
            ):
                if array_name not in toml_data:
                    continue

                entries = toml_data[array_name]

                # Strict format: [[color_path/range/channel]] is a list, not nested dicts
                if not isinstance(entries, list):
                    raise ValueError(
                        f"The [{array_name}] section must be an array-of-tables format "
                        f"(use [[{array_name}]]), not nested tables."
                    )

                for idx, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        raise ValueError(
                            f"[[{array_name}]] entry {idx} must be a table/dict."
                        )

                    # Extract required field: name
                    entry_name = entry.get("name")
                    if entry_name is None:
                        raise ValueError(
                            f"[[{array_name}]] entry {idx} is missing required 'name' "
                            "field (the registry key)."
                        )

                    entry_name = str(entry_name).strip()
                    if entry_name in self._embeddings:
                        raise ValueError(
                            f"Color embedding name '{entry_name}' is duplicated. "
                            "Names must be globally unique."
                        )

                    # Extract config (exclude name)
                    cfg = {k: v for k, v in entry.items() if k != "name"}

                    # Call the appropriate parser
                    self._embeddings[entry_name] = parser(
                        cfg=cfg,
                        embedding_id=entry_name,
                        color_root=color_root,
                        data=data,
                        data_registry=data_registry,
                        roi_registry=roi_registry,
                    )

        return self

    def keys(self) -> list[str]:
        """Return all registered embedding names."""
        return list(self._embeddings.keys())

    def __contains__(self, name: str) -> bool:
        """Check if an embedding name is registered (supports 'name in registry')."""
        return name in self._embeddings

    def resolve(self, embedding: str | ColorEmbedding) -> ColorEmbedding:
        """Resolve embedding identifier or object to embedding object.

        Args:
            embedding: Either a string identifier of a registered embedding, or a
                ColorEmbedding object. If an object is provided, it is verified to be
                registered in self._embeddings.

        Returns:
            The corresponding ColorEmbedding object.

        """
        if isinstance(embedding, str):  # embedding_id
            if embedding not in self._embeddings:
                available = sorted(self._embeddings.keys())
                raise KeyError(
                    "ColorEmbeddingRegistry: key "
                    f"'{embedding}' not found. Available keys: {available}"
                )
            return self._embeddings[embedding]
        else:  # embedding object
            # Make sure embedding is registered in self._embeddings.
            if embedding.embedding_id not in self._embeddings:
                raise KeyError(
                    f"ColorEmbeddingRegistry: embedding with id "
                    f"'{embedding.embedding_id}' not found in registry."
                )
        return embedding

    def resolve_all(self) -> dict[str, ColorEmbedding]:
        """Return a dict of all registered embeddings (already resolved at load time).

        Returns:
            Dict mapping embedding names to their ColorEmbedding objects.
        """
        return dict(self._embeddings)


@dataclass
class ColorPathEmbeddingConfig:
    """GUI-editable view of a single [[color_path]] entry."""

    name: str = field(
        default="",
        metadata={
            "name": "Name",
            "help": "Unique registry key for this color embedding.",
        },
    )
    mode: str = field(
        default="relative",
        metadata={
            "name": "Mode",
            "help": "Color mode: relative (baseline-subtracted) or absolute.",
            "options": ["relative", "absolute"],
        },
    )
    basis: str = field(
        default="labels",
        metadata={
            "name": "Basis",
            "help": "Calibration basis: labels, facies, or global.",
            "options": ["labels", "facies", "global"],
        },
    )
    calibration_mode: str = field(
        default="auto",
        metadata={
            "name": "Calibration Mode",
            "help": "Calibration mode: auto or manual.",
            "options": ["auto", "manual"],
        },
    )
    baseline: str = field(
        default="",
        metadata={
            "name": "Baseline",
            "help": (
                "DataRegistry key for the baseline image/time series. "
                "See the [registry] section for available keys."
            ),
        },
    )
    data: str = field(
        default="",
        metadata={
            "name": "Data",
            "help": (
                "DataRegistry key for the calibration data. "
                "See the [registry] section for available keys."
            ),
        },
    )
    rois: list[str] = field(
        default_factory=list,
        metadata={"name": "ROIs", "help": "ROI registry keys (comma-separated in UI)."},
    )
    num_segments: int = field(
        default=1,
        metadata={"name": "Num Segments", "help": "Number of color path segments."},
    )
    resolution: int = field(
        default=51, metadata={"name": "Resolution", "help": "Color path resolution."}
    )
    threshold_baseline: float = field(
        default=0.0,
        metadata={"name": "Threshold Baseline", "help": "Baseline color threshold."},
    )
    threshold_calibration: float = field(
        default=0.0,
        metadata={
            "name": "Threshold Calibration",
            "help": "Calibration color threshold.",
        },
    )
    reference_label: int = field(
        default=0,
        metadata={"name": "Reference Label", "help": "Reference label index."},
    )
    ignore_labels: list[int] = field(
        default_factory=list,
        metadata={
            "name": "Ignore Labels",
            "help": "Label ids to ignore (comma-separated in UI).",
        },
    )
    ignore_baseline_spectrum: str = field(
        default="expanded",
        metadata={
            "name": "Ignore Baseline Spectrum",
            "help": "How to treat the baseline color spectrum.",
            "options": ["none", "baseline", "expanded"],
        },
    )
    histogram_weighting: str = field(
        default="threshold",
        metadata={
            "name": "Histogram Weighting",
            "help": "Histogram weighting scheme.",
            "options": ["threshold", "wls", "wls_sqrt", "wls_log"],
        },
    )
    calibration_folder: str | None = field(
        default=None,
        metadata={
            "name": "Calibration Folder",
            "help": "Optional override for the calibration output folder.",
        },
    )


@dataclass
class ColorRangeEmbeddingConfig:
    """GUI-editable view of a single [[color_range]] entry (also reused for mask
    sub-tables)."""

    name: str = field(
        default="",
        metadata={
            "name": "Name",
            "help": "Unique registry key for this color embedding.",
        },
    )
    mode: str = field(
        default="absolute",
        metadata={
            "name": "Mode",
            "help": "Color mode: relative or absolute.",
            "options": ["relative", "absolute"],
        },
    )
    basis: str = field(
        default="global",
        metadata={
            "name": "Basis",
            "help": "Calibration basis: labels, facies, or global.",
            "options": ["labels", "facies", "global"],
        },
    )
    color_space: str = field(
        default="",
        metadata={"name": "Color Space", "help": "Color space name (e.g. RGB, HSV)."},
    )
    range: str = field(
        default="",
        metadata={
            "name": "Range",
            "help": (
                "Three [min,max] bounds as 6 comma-separated values "
                "(use 'none' for open bounds), e.g. none,1.0,0.2,0.8,none,none."
            ),
            "widget": "string",
            "range_encoding": "flat6",
        },
    )
    restoration: RestorationConfig | None = field(
        default=None,
        metadata={
            "name": "Restoration",
            "help": "Optional restoration applied to this range/mask.",
        },
    )
    calibration_folder: str | None = field(
        default=None,
        metadata={
            "name": "Calibration Folder",
            "help": "Optional override for the calibration output folder.",
        },
    )


@dataclass
class ColorChannelEmbeddingConfig:
    """GUI-editable view of a single [[color_channel]] entry."""

    name: str = field(
        default="",
        metadata={
            "name": "Name",
            "help": "Unique registry key for this color embedding.",
        },
    )
    mode: str = field(
        default="absolute",
        metadata={
            "name": "Mode",
            "help": "Color mode: relative or absolute.",
            "options": ["relative", "absolute"],
        },
    )
    color_space: str = field(
        default="",
        metadata={"name": "Color Space", "help": "Color space name (e.g. RGB, HSV)."},
    )
    channel: str = field(
        default="",
        metadata={
            "name": "Channel",
            "help": "Channel name within the color space (e.g. r, g, b).",
        },
    )
    mask: ColorRangeEmbeddingConfig | None = field(
        default=None,
        metadata={
            "name": "Mask",
            "help": "Optional range-based mask restricting this channel.",
        },
    )
    restoration: RestorationConfig | None = field(
        default=None,
        metadata={
            "name": "Restoration",
            "help": "Optional restoration applied to this channel.",
        },
    )
    calibration_folder: str | None = field(
        default=None,
        metadata={
            "name": "Calibration Folder",
            "help": "Optional override for the calibration output folder.",
        },
    )


@dataclass
class ColorEmbeddingRegistryConfig:
    """GUI-editable view of the [[color_path]], [[color_range]], [[color_channel]] TOML arrays.

    This dataclass exists purely for GUI schema introspection and does not replace
    ColorEmbeddingRegistry, which remains the canonical runtime registry.
    ColorEmbeddingRegistryConfig fields are read/written via the generic dataclass_group_map
    widget mechanism, which reads from and writes to config_dict["color_path"],
    config_dict["color_range"], and config_dict["color_channel"] respectively.

    Each field's metadata declares its widget type and array_key so the schema introspection
    system knows to create the corresponding multi-row editor and which TOML array to map to.
    """

    color_path_embeddings: dict[str, ColorPathEmbeddingConfig] = field(
        default_factory=dict,
        metadata={
            "name": "Color Paths",
            "help": "Color path embeddings for calibration and analysis.",
            "widget": "dataclass_group_map",
            "array_key": "color_path",
        },
    )
    """Dict of color path embeddings, keyed by name."""

    color_range_embeddings: dict[str, ColorRangeEmbeddingConfig] = field(
        default_factory=dict,
        metadata={
            "name": "Color Ranges",
            "help": "Color range embeddings (HSV/RGB bounds).",
            "widget": "dataclass_group_map",
            "array_key": "color_range",
        },
    )
    """Dict of color range embeddings, keyed by name."""

    color_channel_embeddings: dict[str, ColorChannelEmbeddingConfig] = field(
        default_factory=dict,
        metadata={
            "name": "Color Channels",
            "help": "Color channel embeddings with optional masks.",
            "widget": "dataclass_group_map",
            "array_key": "color_channel",
        },
    )
    """Dict of color channel embeddings, keyed by name."""
