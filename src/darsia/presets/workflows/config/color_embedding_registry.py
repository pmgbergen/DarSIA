"""Centralized registry for color embeddings."""

# TODO: refactor, and extract the three single embedding setups.

from __future__ import annotations

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

from .data_registry import DataRegistry
from .time_data import TimeData
from .utils import _convert_none, _get_section_from_toml

if TYPE_CHECKING:
    from .roi_registry import RoiRegistry


def _parse_mode(value: str, *, context: str) -> darsia.ColorMode:
    try:
        return darsia.ColorMode(value.lower().strip())
    except Exception as exc:
        raise ValueError(
            f"Invalid {context}.mode '{value}'. Supported values are 'relative' and "
            "'absolute'."
        ) from exc


# TODO Make this part of DataRegistry?
def _resolve_selector(
    cfg: dict,
    key: str,
    *,
    section: str,
    data_registry: DataRegistry | None,
    required: bool = True,
) -> TimeData | None:
    if key not in cfg:
        if required:
            raise KeyError(f"{section}.{key}")
        return None
    selector = cfg[key]
    if isinstance(selector, list) and not all(
        isinstance(token, str) for token in selector
    ):
        raise ValueError(f"{section}.{key} selector lists must contain only strings.")
    if not isinstance(selector, (str, list)):
        raise ValueError(
            f"{section}.{key} must reference [data.*] selector key(s) as string/list."
        )
    if data_registry is None:
        raise ValueError(
            f"{section}.{key} references [data.*] selector key(s), but no "
            "DataRegistry is available."
        )
    return data_registry.resolve(selector)


@dataclass
class ColorEmbeddingRegistry:
    """Registry of configured color embeddings."""

    embeddings: dict[str, ColorEmbedding] = field(default_factory=dict)

    def load(
        self,
        path: Path | list[Path],
        *,
        data: Path | None,
        results: Path | None,
        data_registry: DataRegistry | None = None,
        roi_registry: "RoiRegistry | None" = None,
    ) -> "ColorEmbeddingRegistry":
        sec = _get_section_from_toml(path, "color")
        if not isinstance(sec, dict):
            raise ValueError("[color] must be a table.")

        self.embeddings = {}
        seen: set[str] = set()
        color_root = results / "calibration" / "color" if results is not None else None

        # [color.path.<id>]
        path_sec = sec.get("path", {})
        if isinstance(path_sec, dict):
            for embedding_id, cfg in path_sec.items():
                if embedding_id in seen:
                    raise ValueError(
                        f"Duplicate color embedding identifier '{embedding_id}'."
                    )
                seen.add(embedding_id)
                mode = _parse_mode(
                    cfg.get("mode", "relative"), context=f"color.path.{embedding_id}"
                )
                basis = parse_color_embedding_basis(cfg.get("basis", "labels"))
                calibration_root = (
                    Path(cfg["calibration_folder"])
                    if "calibration_folder" in cfg
                    else (
                        color_root / embedding_id if color_root is not None else Path()
                    )
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
                    rois=list(cfg.get("rois", [])),
                    ignore_baseline_spectrum=str(
                        cfg.get("ignore_baseline_spectrum", "expanded")
                    ),
                    histogram_weighting=str(
                        cfg.get("histogram_weighting", "threshold")
                    ),
                    calibration_mode=str(
                        cfg.get("mode_calibration", cfg.get("calibration_mode", "auto"))
                    ),
                )
                embedding.baseline_data = _resolve_selector(
                    cfg,
                    "baseline",
                    section=f"color.path.{embedding_id}",
                    data_registry=data_registry,
                )
                embedding.data = _resolve_selector(
                    cfg,
                    "data",
                    section=f"color.path.{embedding_id}",
                    data_registry=data_registry,
                )
                if (
                    "roi" in cfg
                    and isinstance(cfg["roi"], dict)
                    and roi_registry is not None
                ):
                    from .roi import RoiAndLabelConfig, RoiConfig

                    for key, entry in cfg["roi"].items():
                        roi_obj: RoiConfig | RoiAndLabelConfig
                        if "label" in entry:
                            roi_obj = RoiAndLabelConfig().load(entry)
                        else:
                            roi_obj = RoiConfig().load(entry)
                        roi_registry.register(key, roi_obj)
                        if key not in embedding.rois:
                            embedding.rois.append(key)
                self.embeddings[embedding_id] = embedding

        # [color.range.<id>]
        range_sec = sec.get("range", {})
        if isinstance(range_sec, dict):
            for embedding_id, cfg in range_sec.items():
                if embedding_id in seen:
                    raise ValueError(
                        f"Duplicate color embedding identifier '{embedding_id}'."
                    )
                seen.add(embedding_id)
                mode = _parse_mode(
                    cfg.get("mode", "absolute"), context=f"color.range.{embedding_id}"
                )
                basis = parse_color_embedding_basis(cfg.get("basis", "global"))
                raw_range = cfg.get("range")
                if not isinstance(raw_range, list) or len(raw_range) != 3:
                    raise ValueError(
                        f"color.range.{embedding_id}.range must be a list of 3 "
                        "[min,max] bounds."
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
                    else (
                        color_root / embedding_id if color_root is not None else Path()
                    )
                )
                if "color_space" not in cfg:
                    raise ValueError(
                        f"color.range.{embedding_id}.color_space is required."
                    )
                self.embeddings[embedding_id] = ColorRangeEmbedding(
                    embedding_id=embedding_id,
                    mode=mode,
                    basis=basis,
                    calibration_root=calibration_root,
                    color_space=str(cfg["color_space"]).upper().strip(),
                    ranges=ranges,
                )

        # [color.channel.<id>]
        channel_sec = sec.get("channel", {})
        if isinstance(channel_sec, dict):
            for embedding_id, cfg in channel_sec.items():
                if embedding_id in seen:
                    raise ValueError(
                        f"Duplicate color embedding identifier '{embedding_id}'."
                    )
                seen.add(embedding_id)
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
                    else (
                        color_root / embedding_id if color_root is not None else Path()
                    )
                )
                for key in ["color_space", "channel"]:
                    if key not in cfg:
                        raise ValueError(
                            f"color.channel.{embedding_id}.{key} is required."
                        )
                self.embeddings[embedding_id] = ColorChannelEmbedding(
                    embedding_id=embedding_id,
                    mode=mode,
                    basis=basis,
                    calibration_root=calibration_root,
                    color_space=str(cfg["color_space"]).upper().strip(),
                    channel=str(cfg["channel"]).lower().strip(),
                )
        return self

    def resolve(self, embedding: str | ColorEmbedding) -> ColorEmbedding:
        if isinstance(embedding, str):
            if embedding not in self.embeddings:
                available = sorted(self.embeddings.keys())
                raise KeyError(
                    "ColorEmbeddingRegistry: key "
                    f"'{embedding}' not found. Available keys: {available}"
                )
            return self.embeddings[embedding]
        else:
            # Make sure embedding is registered in self.embeddings.
            if embedding.embedding_id not in self.embeddings:
                raise KeyError(
                    f"ColorEmbeddingRegistry: embedding with id "
                    f"'{embedding.embedding_id}' not found in registry."
                )
        return embedding

    def keys(self) -> list[str]:
        return sorted(self.embeddings.keys())
