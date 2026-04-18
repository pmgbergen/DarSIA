"""Shared resolution of analysis modes to scalar images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import darsia
from darsia.presets.workflows.config.colorchannel_registry import ColorChannelRegistry
from darsia.presets.workflows.config.colorrange import ColorRangeConfig

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
class ColorChannelMode:
    name: str


@dataclass(frozen=True)
class ColorRangeMode:
    name: str


def parse_colorchannel_mode(mode: str) -> ColorChannelMode | None:
    parts = mode.split(".")
    if len(parts) != 2 or parts[0].lower() != "colorchannel":
        return None
    if len(parts[1].strip()) == 0:
        return None
    return ColorChannelMode(name=parts[1].strip())


def parse_colorrange_mode(mode: str) -> ColorRangeMode | None:
    parts = mode.split(".")
    if len(parts) != 2 or parts[0].lower() != "colorrange":
        return None
    return ColorRangeMode(name=parts[1])


def validate_mode_syntax(
    mode: str, colorchannel_registry: ColorChannelRegistry | None = None
) -> bool:
    mode = mode.strip()
    if mode in LEGACY_COLOR_TO_MASS_MODES or mode in SCALAR_PRODUCT_MODES:
        return True
    colorchannel = parse_colorchannel_mode(mode)
    if colorchannel is not None:
        if colorchannel_registry is None:
            return True
        return colorchannel.name in colorchannel_registry.channels

    # Explicitly reject removed fast syntax: colorchannel.<space>.<channel>
    parts = mode.split(".")
    if len(parts) == 3 and parts[0].lower() == "colorchannel":
        return False
    return parse_colorrange_mode(mode) is not None


def mode_requires_color_to_mass(mode: str) -> bool:
    mode = mode.strip()
    if mode in LEGACY_COLOR_TO_MASS_MODES or mode in SCALAR_PRODUCT_MODES:
        return True
    if parse_colorchannel_mode(mode) is not None:
        return False
    if parse_colorrange_mode(mode) is not None:
        return False
    # Keep conservative default for unknown modes.
    return True


def _normalized_trichromatic(
    image: darsia.Image, color_space: str
) -> tuple[np.ndarray, str]:
    if not isinstance(image, darsia.OpticalImage):
        raise ValueError("Color-based modes require an optical image input.")
    converted = image.to_trichromatic(color_space, return_image=True)
    arr = converted.img.astype(np.float32, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Color-based mode resolution requires finite image values.")
    cs = color_space.upper()

    if cs in {"RGB", "BGR"}:
        if np.max(arr) > 1.0:
            arr = arr / 255.0
    elif cs in {"HSV", "HLS"}:
        # OpenCV conventions differ by dtype; normalize all channels to [0, 1].
        hue = arr[..., 0]
        hue_scale = 360.0 if np.max(hue) > 180.0 else 180.0
        arr[..., 0] = hue / hue_scale
        if np.max(arr[..., 1:]) > 1.0:
            arr[..., 1:] = arr[..., 1:] / 255.0
    elif cs == "LAB":
        # Normalize LAB to [0, 1] per component.
        if np.nanmin(arr[..., 1:]) < 0.0:
            # float LAB: L in [0, 100], a/b in ~[-127,127]
            arr[..., 0] = arr[..., 0] / 100.0
            arr[..., 1] = (arr[..., 1] + 127.0) / 254.0
            arr[..., 2] = (arr[..., 2] + 127.0) / 254.0
        else:
            # uint8-style LAB in [0, 255]
            arr = arr / 255.0
    else:
        raise ValueError(f"Unsupported color space '{color_space}'.")

    return np.clip(arr, 0.0, 1.0), cs


def _to_scalar_image(template: darsia.Image, values: np.ndarray) -> darsia.ScalarImage:
    metadata = template.metadata()
    metadata.pop("color_space", None)
    return darsia.ScalarImage(img=values, **metadata)


def _channel_index(color_space: str, channel: str) -> int:
    channel_map = {
        "RGB": {"r": 0, "g": 1, "b": 2},
        "BGR": {"b": 0, "g": 1, "r": 2},
        "HSV": {"h": 0, "s": 1, "v": 2},
        "HLS": {"h": 0, "l": 1, "s": 2},
        "LAB": {"l": 0, "a": 1, "b": 2},
    }
    mapping = channel_map.get(color_space.upper(), {})
    if channel not in mapping:
        raise ValueError(
            f"Unsupported channel '{channel}' for color space '{color_space}'."
        )
    return mapping[channel]


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


def _resolve_colorchannel_mode(
    mode: str,
    image: darsia.Image,
    colorchannel_registry: ColorChannelRegistry | None = None,
) -> darsia.ScalarImage:
    parsed = parse_colorchannel_mode(mode)
    if parsed is None:
        raise ValueError(f"Invalid color channel mode '{mode}'.")
    if colorchannel_registry is None:
        raise ValueError(
            "Color channel mode resolution requires a colorchannel registry. "
            "Define [colorchannel.<name>] and use mode 'colorchannel.<name>'."
        )
    try:
        named_colorchannel = colorchannel_registry.resolve(parsed.name)[parsed.name]
    except KeyError as exc:
        raise ValueError(
            f"Undefined colorchannel '{parsed.name}'. "
            f"Configure [colorchannel.{parsed.name}]."
        ) from exc

    trichromatic, color_space = _normalized_trichromatic(
        image, named_colorchannel.color_space
    )
    index = _channel_index(color_space, named_colorchannel.channel)
    return _to_scalar_image(image, trichromatic[..., index].astype(np.float32))


def _resolve_colorrange_mode(
    mode: str,
    image: darsia.Image,
    colorrange_config: ColorRangeConfig | None = None,
) -> darsia.ScalarImage:
    parsed = parse_colorrange_mode(mode)
    if parsed is None:
        raise ValueError(f"Invalid color range mode '{mode}'.")
    if colorrange_config is None or parsed.name not in colorrange_config.ranges:
        raise ValueError(
            f"Undefined colorrange '{parsed.name}'. Configure [colorrange.{parsed.name}]."
        )

    named_range = colorrange_config.ranges[parsed.name]
    trichromatic, color_space = _normalized_trichromatic(image, named_range.color_space)
    mask = np.ones(trichromatic.shape[:2], dtype=bool)

    for channel, (lower, upper) in enumerate(named_range.ranges):
        values = trichromatic[..., channel]
        if lower is None and upper is None:
            component_mask = np.ones_like(values, dtype=bool)
        elif lower is None:
            component_mask = values <= float(upper)
        elif upper is None:
            component_mask = values >= float(lower)
        elif (
            channel == 0
            and color_space in {"HSV", "HLS"}
            and float(lower) > float(upper)
        ):
            # Hue wrap-around interval.
            component_mask = np.logical_or(
                values >= float(lower), values <= float(upper)
            )
        else:
            component_mask = np.logical_and(
                values >= float(lower), values <= float(upper)
            )
        mask = np.logical_and(mask, component_mask)

    return _to_scalar_image(image, mask.astype(np.float32))


def resolve_mode_image(
    mode: str,
    image: darsia.Image,
    mass_analysis_result: Any = None,
    colorrange_config: ColorRangeConfig | None = None,
    colorchannel_registry: ColorChannelRegistry | None = None,
    scalar_products: dict[str, darsia.Image | None] | None = None,
) -> darsia.Image:
    mode = mode.strip()
    if scalar_products is not None and mode in scalar_products:
        value = scalar_products[mode]
        if value is not None:
            return value
    if mode in LEGACY_COLOR_TO_MASS_MODES:
        return _resolve_legacy_mode(mode, mass_analysis_result)
    if parse_colorchannel_mode(mode) is not None:
        return _resolve_colorchannel_mode(mode, image, colorchannel_registry)
    parts = mode.split(".")
    if len(parts) == 3 and parts[0].lower() == "colorchannel":
        raise ValueError(
            "Inline colorchannel modes like 'colorchannel.<space>.<channel>' are no "
            "longer supported. Define [colorchannel.<name>] and use "
            "'colorchannel.<name>'."
        )
    if parse_colorrange_mode(mode) is not None:
        return _resolve_colorrange_mode(mode, image, colorrange_config)
    raise ValueError(f"Unsupported analysis mode '{mode}'.")
