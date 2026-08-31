"""Centralized registry for color embeddings."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

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
from .utils import _convert_none


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
) -> ColorPathEmbedding:
    mode = _parse_mode(
        cfg.get("mode", "relative"), context=f"color.path.{embedding_id}"
    )
    basis = parse_color_embedding_basis(cfg.get("basis", "labels"))
    root = (
        Path(cfg["root"])
        if "root" in cfg
        else (
            color_root / "color_path" / embedding_id
            if color_root is not None
            else Path()
        )
    )

    embedding = ColorPathEmbedding(
        embedding_id=embedding_id,
        mode=mode,
        basis=basis,
        root=root,
        reference_label=int(cfg.get("reference_label", 0)),
    )
    return embedding


def parse_color_range_embedding(
    cfg: dict,
    embedding_id: str,
    color_root: Path | None,
    data: Path | None,
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
    root = (
        Path(cfg["root"])
        if "root" in cfg
        else (
            color_root / "color_range" / embedding_id
            if color_root is not None
            else Path()
        )
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
        root=root,
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
) -> ColorChannelEmbedding:
    mode = _parse_mode(
        cfg.get("mode", "absolute"), context=f"color.channel.{embedding_id}"
    )
    basis = parse_color_embedding_basis(cfg.get("basis", "global"))
    if basis != ColorEmbeddingBasis.GLOBAL:
        raise NotImplementedError(
            "color.channel.<id> currently only supports basis='global'."
        )
    root = (
        Path(cfg["root"])
        if "root" in cfg
        else (
            color_root / "color_channel" / embedding_id
            if color_root is not None
            else Path()
        )
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
            color_root=root,
            data=data,
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
        root=root,
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
    ) -> "ColorEmbeddingRegistry":
        """Load color embeddings from [[color_path]], [[color_range]], [[color_channel]]
        arrays in TOML.

        Hand-parses TOML (like FormatRegistry and RoiRegistry) since array-of-tables
        is not supported by the generic _get_section_from_toml helper.

        Args:
            path: Path or list of Paths to TOML config file(s).
            data: Data folder path.
            results: Results folder path.

        Returns:
            self

        Raises:
            ValueError: If any array section is not an array-of-tables, if any
                entry is missing required field (name), or if names are duplicated.
        """
        paths = [path] if isinstance(path, Path) else path
        self._embeddings = {}
        color_root = results / "color" if results is not None else None

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
            "group": "Properties",
        },
    )
    basis: str = field(
        default="labels",
        metadata={
            "name": "Basis",
            "help": "Calibration basis: labels, facies, or global.",
            "options": ["labels", "facies", "global"],
            "group": "Properties",
        },
    )
    reference_label: int = field(
        default=0,
        metadata={
            "name": "Reference Label",
            "help": "Reference label index used to define color mapping, e.g., for plotting.",
            "group": "Properties",
        },
    )
    root: Path | None = field(
        default=None,
        metadata={
            "name": "Calibration Root",
            "help": "Optional override for the calibration root folder (baseline spectrum, "
            "color range, etc.).",
            "hidden": True,
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
    root: Path | None = field(
        default=None,
        metadata={
            "name": "Calibration Root",
            "help": "Optional override for the calibration root folder.",
            "hidden": True,
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
    root: Path | None = field(
        default=None,
        metadata={
            "name": "Calibration Root",
            "help": "Optional override for the calibration root folder.",
            "hidden": True,
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
