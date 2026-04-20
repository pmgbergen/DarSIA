"""Shared resolution of analysis modes to scalar images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import darsia
from darsia.presets.workflows.color_embedding import ColorEmbeddingRuntime
from darsia.presets.workflows.config.color_embedding_registry import (
    ColorEmbeddingRegistry,
)

LEGACY_COLOR_TO_MASS_MODES = {
    "concentration_aq",
    "saturation_g",
    "mass",
    "mass_total",
    "mass_g",
    "mass_aq",
}

SCALAR_PRODUCT_MODES = {
    "rescaled_mass",
    "rescaled_saturation_g",
    "rescaled_concentration_aq",
}


@dataclass(frozen=True)
class ColorEmbeddingMode:
    name: str


def parse_color_mode(mode: str) -> ColorEmbeddingMode | None:
    parts = mode.split(".")
    if len(parts) != 2 or parts[0].lower() != "color":
        return None
    return ColorEmbeddingMode(name=parts[1].strip())


def validate_mode_syntax(
    mode: str,
    color_embedding_registry: ColorEmbeddingRegistry | None = None,
) -> bool:
    mode = mode.strip()
    if mode in LEGACY_COLOR_TO_MASS_MODES or mode in SCALAR_PRODUCT_MODES:
        return True
    color_mode = parse_color_mode(mode)
    if color_mode is None:
        return False
    if color_embedding_registry is None:
        return True
    return color_mode.name in color_embedding_registry.embeddings


def mode_requires_color_to_mass(mode: str) -> bool:
    mode = mode.strip()
    if mode in LEGACY_COLOR_TO_MASS_MODES or mode in SCALAR_PRODUCT_MODES:
        return True
    if parse_color_mode(mode) is not None:
        return False
    # Keep conservative default for unknown modes.
    return True


def _resolve_legacy_mode(mode: str, mass_analysis_result: Any) -> darsia.Image:
    if mass_analysis_result is None:
        raise ValueError(f"Mode '{mode}' requires color-to-mass analysis results.")
    if mode == "mass_total" or mode == "mass":
        return mass_analysis_result.mass
    if mode == "mass_g":
        return mass_analysis_result.mass_g
    if mode == "mass_aq":
        return mass_analysis_result.mass_aq
    if mode == "concentration_aq":
        return mass_analysis_result.concentration_aq
    if mode == "saturation_g":
        return mass_analysis_result.saturation_g
    raise ValueError(f"Unsupported legacy mode '{mode}'.")


def _resolve_color_mode(
    mode: str,
    image: darsia.Image,
    color_embedding_registry: ColorEmbeddingRegistry | None,
    color_embedding_runtime: ColorEmbeddingRuntime | None,
) -> darsia.ScalarImage:
    if color_embedding_registry is None:
        raise ValueError(
            "Color mode resolution requires color embedding registry. "
            "Define [color.path.*], [color.range.*], or [color.channel.*]."
        )
    if color_embedding_runtime is None:
        raise ValueError(
            "Color mode resolution requires runtime context with rig/baseline."
        )
    embedding = color_embedding_registry.resolve_mode(mode)
    return embedding.to_scalar_image(image, color_embedding_runtime)


def resolve_mode_image(
    mode: str,
    image: darsia.Image,
    mass_analysis_result: Any = None,
    color_embedding_registry: ColorEmbeddingRegistry | None = None,
    color_embedding_runtime: ColorEmbeddingRuntime | None = None,
    scalar_products: dict[str, darsia.Image | None] | None = None,
) -> darsia.Image:
    mode = mode.strip()
    if scalar_products is not None and mode in scalar_products:
        value = scalar_products[mode]
        if value is not None:
            return value
    if mode in LEGACY_COLOR_TO_MASS_MODES:
        return _resolve_legacy_mode(mode, mass_analysis_result)
    if parse_color_mode(mode) is not None:
        return _resolve_color_mode(
            mode,
            image,
            color_embedding_registry,
            color_embedding_runtime,
        )
    raise ValueError(f"Unsupported analysis mode '{mode}'.")
