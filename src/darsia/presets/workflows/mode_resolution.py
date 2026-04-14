"""Shared resolution of analysis modes to scalar images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import darsia
from darsia.presets.workflows.config.colorrange import ColorRangeConfig


LEGACY_COLOR_TO_MASS_MODES = {
    "concentration_aq",
    "saturation_g",
    "mass",
    "mass_total",
    "mass_g",
    "mass_aq",
}


@dataclass(frozen=True)
class ColorChannelMode:
    color_space: str
    channel: str


@dataclass(frozen=True)
class ColorRangeMode:
    name: str


def parse_colorchannel_mode(mode: str) -> ColorChannelMode | None:
    parts = mode.split(".")
    if len(parts) != 3 or parts[0].lower() != "colorchannel":
        return None
    return ColorChannelMode(color_space=parts[1].upper(), channel=parts[2].lower())


def parse_colorrange_mode(mode: str) -> ColorRangeMode | None:
    parts = mode.split(".")
    if len(parts) != 2 or parts[0].lower() != "colorrange":
        return None
    return ColorRangeMode(name=parts[1])


def validate_mode_syntax(mode: str) -> bool:
    mode = mode.strip()
    if mode in LEGACY_COLOR_TO_MASS_MODES:
        return True
    colorchannel = parse_colorchannel_mode(mode)
    if colorchannel is not None:
        return colorchannel.color_space in {"RGB", "BGR", "HSV", "HLS", "LAB"} and (
            colorchannel.channel in {"r", "g", "b", "h", "s", "v", "l", "a"}
        )
    return parse_colorrange_mode(mode) is not None


def mode_requires_color_to_mass(mode: str) -> bool:
    mode = mode.strip()
    if mode in LEGACY_COLOR_TO_MASS_MODES:
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
    cs = color_space.upper()

    if cs in {"RGB", "BGR"}:
        if np.nanmax(arr) > 1.0:
            arr = arr / 255.0
    elif cs in {"HSV", "HLS"}:
        # OpenCV conventions differ by dtype; normalize all channels to [0, 1].
        hue = arr[..., 0]
        hue_scale = 360.0 if np.nanmax(hue) > 180.0 else 180.0
        arr[..., 0] = hue / max(hue_scale, 1.0)
        if np.nanmax(arr[..., 1:]) > 1.0:
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


def _resolve_colorchannel_mode(mode: str, image: darsia.Image) -> darsia.ScalarImage:
    parsed = parse_colorchannel_mode(mode)
    if parsed is None:
        raise ValueError(f"Invalid color channel mode '{mode}'.")
    trichromatic, color_space = _normalized_trichromatic(image, parsed.color_space)
    index = _channel_index(color_space, parsed.channel)
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
            component_mask = np.logical_or(values >= float(lower), values <= float(upper))
        else:
            component_mask = np.logical_and(values >= float(lower), values <= float(upper))
        mask = np.logical_and(mask, component_mask)

    return _to_scalar_image(image, mask.astype(np.float32))


def resolve_mode_image(
    mode: str,
    image: darsia.Image,
    mass_analysis_result: Any = None,
    colorrange_config: ColorRangeConfig | None = None,
) -> darsia.Image:
    mode = mode.strip()
    if mode in LEGACY_COLOR_TO_MASS_MODES:
        return _resolve_legacy_mode(mode, mass_analysis_result)
    if parse_colorchannel_mode(mode) is not None:
        return _resolve_colorchannel_mode(mode, image)
    if parse_colorrange_mode(mode) is not None:
        return _resolve_colorrange_mode(mode, image, colorrange_config)
    raise ValueError(f"Unsupported analysis mode '{mode}'.")
