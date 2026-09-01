"""Configuration for analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import darsia
from darsia.presets.workflows.mode_resolution import validate_mode_syntax

from .contour_smoother import ContourSmootherSelection
from .fingers import FingersConfig
from .roi_registry import _load_roi_key_list
from .segmentation import SegmentationConfig
from .utils import _get_key, _get_section, _get_section_from_toml

if TYPE_CHECKING:
    from darsia.signals.color import ColorEmbedding

    from .color_embedding_registry import ColorEmbeddingRegistry
    from .format_registry import FormatRegistry
    from .roi_registry import RoiRegistry

logger = logging.getLogger(__name__)

SUPPORTED_ANALYSIS_MASS_EXPORT_MODES = {
    "mass",
    "rescaled_mass",
    "extensive_mass",
    "extensive_rescaled_mass",
    "saturation_g",
    "rescaled_saturation_g",
    "concentration_aq",
    "rescaled_concentration_aq",
}


def _to_rgb(color: list[int] | tuple[int, int, int], name: str) -> tuple[int, int, int]:
    if len(color) != 3:
        raise ValueError(f"{name} must have exactly 3 entries [R, G, B].")
    vals = tuple(int(v) for v in color)
    if any(v < 0 or v > 255 for v in vals):
        raise ValueError(f"{name} entries must be in [0, 255].")
    return vals


def _validate_format_keys(
    formats: list[str],
    format_registry: "FormatRegistry | None",
    allowed_types: set[str],
    context: str,
) -> None:
    """Validate that each formats entry resolves (via registry, by type) or is a raw
    type string, in both cases restricted to allowed_types."""
    invalid = []
    for key in formats:
        if format_registry is not None and key in format_registry.keys():
            spec_type = format_registry.resolve(key)[0].type
            if spec_type not in allowed_types:
                invalid.append(key)
        elif key.lower() in allowed_types:
            continue
        else:
            invalid.append(key)
    if invalid:
        raise ValueError(
            f"Unsupported {context} entries: {', '.join(sorted(invalid))}. "
            f"Supported types: {', '.join(sorted(allowed_types))}, or a "
            f"[[format]] registry key resolving to one of these types."
        )


@dataclass
class AnalysisThresholdingLegendConfig:
    show: bool = field(
        default=True,
        metadata={
            "name": "Show",
            "help": "Display legend overlay on thresholded images.",
        },
    )
    font_scale: float = field(
        default=0.7,
        metadata={"name": "Font scale", "help": "Size scaling factor for legend text."},
    )
    thickness: int = field(
        default=2,
        metadata={"name": "Thickness", "help": "Stroke width for text rendering."},
    )
    line_spacing: int = field(
        default=8,
        metadata={
            "name": "Line spacing",
            "help": "Vertical gap between legend entries.",
        },
    )
    position: tuple[int, int] = field(
        default=(20, 20),
        metadata={
            "name": "Position",
            "help": "Pixel coordinates [x, y] for legend placement.",
        },
    )
    text_color: tuple[int, int, int] = field(
        default=(255, 255, 255), metadata={"name": "Text color"}
    )
    box_enabled: bool = field(
        default=True,
        metadata={"name": "Box", "help": "Draw a background box behind legend text."},
    )
    box_color: tuple[int, int, int] = field(
        default=(0, 0, 0), metadata={"name": "Box color"}
    )
    box_alpha: float = field(
        default=0.4,
        metadata={"name": "Box opacity", "help": "Background transparency [0-1]."},
    )
    box_padding: int = field(
        default=10,
        metadata={"name": "Box padding", "help": "Pixel margin inside background box."},
    )

    def load(self, sec: dict) -> "AnalysisThresholdingLegendConfig":
        self.show = bool(_get_key(sec, "show", required=False, default=self.show))
        self.font_scale = float(
            _get_key(sec, "font_scale", required=False, default=self.font_scale)
        )
        self.thickness = int(
            _get_key(sec, "thickness", required=False, default=self.thickness)
        )
        self.line_spacing = int(
            _get_key(sec, "line_spacing", required=False, default=self.line_spacing)
        )
        position = _get_key(sec, "position", required=False, default=self.position)
        if len(position) != 2:
            raise ValueError("analysis.thresholding.legend.position must be [x, y].")
        self.position = (int(position[0]), int(position[1]))
        self.text_color = _to_rgb(
            _get_key(sec, "text_color", required=False, default=self.text_color),
            "analysis.thresholding.legend.text_color",
        )
        self.box_enabled = bool(
            _get_key(sec, "box_enabled", required=False, default=self.box_enabled)
        )
        self.box_color = _to_rgb(
            _get_key(sec, "box_color", required=False, default=self.box_color),
            "analysis.thresholding.legend.box_color",
        )
        self.box_alpha = float(
            _get_key(sec, "box_alpha", required=False, default=self.box_alpha)
        )
        if not (0 <= self.box_alpha <= 1):
            raise ValueError(
                "analysis.thresholding.legend.box_alpha must be in [0, 1]."
            )
        self.box_padding = int(
            _get_key(sec, "box_padding", required=False, default=self.box_padding)
        )
        return self


@dataclass
class AnalysisThresholdingConfig:
    @dataclass
    class LayerConfig:
        mode: str = field(default="concentration_aq", metadata={"name": "Mode"})
        threshold_min: float | None = field(
            default=None, metadata={"name": "Min threshold"}
        )
        threshold_max: float | None = field(
            default=None, metadata={"name": "Max threshold"}
        )
        label: str = field(
            default="",
            metadata={"name": "Label", "help": "Display name for this layer."},
        )
        fill: tuple[int, int, int] = field(
            default=(255, 255, 255), metadata={"name": "Fill color"}
        )
        stroke: tuple[int, int, int] = field(
            default=(0, 0, 0), metadata={"name": "Stroke color"}
        )
        fill_alpha: float = field(
            default=0.35,
            metadata={"name": "Fill opacity", "help": "Fill transparency [0-1]."},
        )
        stroke_width: int = field(
            default=2,
            metadata={"name": "Stroke width", "help": "Contour line thickness."},
        )

        def load(
            self,
            sec: dict,
            *,
            key: str,
            color_embedding_registry: ColorEmbeddingRegistry | None = None,
        ) -> "AnalysisThresholdingConfig.LayerConfig":
            self.mode = _get_key(sec, "mode", required=True, type_=str).strip()
            validate_mode_syntax(
                self.mode,
                color_embedding_registry,
                "analysis.thresholding.layer.{key}.mode",
            )
            self.threshold_min = _get_key(sec, "threshold_min", required=False)
            self.threshold_max = _get_key(sec, "threshold_max", required=False)
            if self.threshold_min is not None:
                self.threshold_min = float(self.threshold_min)
            if self.threshold_max is not None:
                self.threshold_max = float(self.threshold_max)
            if (
                self.threshold_min is not None
                and self.threshold_max is not None
                and self.threshold_min > self.threshold_max
            ):
                raise ValueError(
                    f"analysis.thresholding.layer.{key} has threshold_min > threshold_max."
                )
            if self.threshold_min is None and self.threshold_max is None:
                raise ValueError(
                    f"analysis.thresholding.layer.{key} must have at least one of "
                    "threshold_min or threshold_max."
                )

            self.label = _get_key(sec, "label", required=False, default=key, type_=str)
            self.fill = _to_rgb(
                _get_key(sec, "fill", required=False, default=self.fill),
                f"analysis.thresholding.layer.{key}.fill",
            )
            self.stroke = _to_rgb(
                _get_key(sec, "stroke", required=False, default=self.stroke),
                f"analysis.thresholding.layer.{key}.stroke",
            )
            self.fill_alpha = float(
                _get_key(sec, "fill_alpha", required=False, default=self.fill_alpha)
            )
            if not (0.0 <= self.fill_alpha <= 1.0):
                raise ValueError(
                    f"analysis.thresholding.layer.{key}.fill_alpha must be in [0, 1]."
                )
            self.stroke_width = int(
                _get_key(sec, "stroke_width", required=False, default=self.stroke_width)
            )
            if self.stroke_width < 0:
                raise ValueError(
                    f"analysis.thresholding.layer.{key}.stroke_width must be >= 0."
                )

            return self

    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is used for thresholding.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for thresholding."""
    formats: list[str] = field(
        default_factory=lambda: ["jpg", "npz"],
        metadata={
            "name": "Export formats",
            "help": (
                "Image/mask formats to save for thresholding. Registry entries of any type "
                "are offered."
            ),
            "widget": "format_key_list",
            "format_types": {"jpg", "png", "npz", "npy", "csv"},
        },
    )
    layers: dict[str, LayerConfig] = field(
        default_factory=dict, metadata={"name": "Layers"}
    )
    legend: AnalysisThresholdingLegendConfig = field(
        default_factory=AnalysisThresholdingLegendConfig, metadata={"name": "Legend"}
    )
    folder: Path | None = field(
        default=None,
        metadata={
            "name": "Output folder",
            "help": "Directory for thresholded results.",
            "hidden": True,
        },
    )
    """Path to the results folder for thresholding analysis."""

    def load(
        self,
        sec: dict,
        results: Path | None,
        color_embedding_registry: ColorEmbeddingRegistry | None = None,
        format_registry: "FormatRegistry | None" = None,
    ) -> "AnalysisThresholdingConfig":
        sub_sec = _get_section(sec, "thresholding")

        self.data_selection = _get_key(
            sub_sec, "data_selection", required=False, default=None
        )

        raw_formats = _get_key(sub_sec, "formats", required=False, default=self.formats)
        if not isinstance(raw_formats, list):
            raise ValueError("analysis.thresholding.formats must be a list.")
        if not all(isinstance(fmt, str) for fmt in raw_formats):
            raise ValueError("analysis.thresholding.formats entries must be strings.")
        self.formats = [fmt.strip().lower() for fmt in raw_formats if fmt.strip()]
        if len(self.formats) == 0:
            raise ValueError("analysis.thresholding.formats must not be empty.")
        _validate_format_keys(
            self.formats,
            format_registry,
            {"jpg", "png", "npz", "npy", "csv"},
            "[analysis.thresholding].formats",
        )

        raw_layers = _get_key(sub_sec, "layer", required=False, default={})
        if not isinstance(raw_layers, dict):
            raise ValueError("analysis.thresholding.layer must be a table/dict.")
        self.layers = {}
        if len(raw_layers) > 0:
            for key in raw_layers.keys():
                layer_sec = _get_section(raw_layers, key)
                self.layers[key] = self.LayerConfig().load(
                    layer_sec,
                    key=key,
                    color_embedding_registry=color_embedding_registry,
                )
        legend = _get_key(sub_sec, "legend", required=False, default={})
        if not isinstance(legend, dict):
            raise ValueError("analysis.thresholding.legend must be a table/dict.")
        self.legend.load(legend)

        folder = _get_key(sub_sec, "folder", required=False, type_=Path)
        if not folder:
            assert results is not None
            self.folder = results / "thresholding"
        else:
            self.folder = folder

        return self

    def error(self):
        raise ValueError(
            "Use [analysis.thresholding] in the config file to load thresholding."
        )


@dataclass
class AnalysisSegmentationConfig:
    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is used for segmentation.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for segmentation."""
    formats: list[str] | None = field(
        default=None,
        metadata={
            "name": "Export formats",
            "help": "Image formats to save for segmentation.",
            "widget": "format_key_list",
            "format_types": {"jpg", "png", "npz", "npy"},
        },
    )
    """Output formats for segmentation images."""
    config: SegmentationConfig | dict[str, SegmentationConfig] = field(
        default_factory=lambda: SegmentationConfig(),
        metadata={"name": "Config", "hidden": True},
    )
    folder: Path | None = field(
        default=None,
        metadata={
            "name": "Output folder",
            "help": "Directory for segmentation results.",
            "hidden": True,
        },
    )
    """Path to the results folder for segmentation."""

    def load(
        self,
        sec: dict,
        results: Path | None,
        color_embedding_registry: ColorEmbeddingRegistry | None = None,
        format_registry: "FormatRegistry | None" = None,
    ) -> "AnalysisSegmentationConfig":
        # Allow for two scenarios: single segmentation or multiple segmentations
        sub_sec = _get_section(sec, "segmentation")

        self.data_selection = _get_key(
            sub_sec, "data_selection", required=False, default=None
        )

        raw_formats = _get_key(sub_sec, "formats", required=False, default=None)
        if raw_formats is None:
            self.formats = None
        else:
            if not isinstance(raw_formats, list):
                raise ValueError("analysis.segmentation.formats must be a list.")
            if not all(isinstance(fmt, str) for fmt in raw_formats):
                raise ValueError(
                    "analysis.segmentation.formats entries must be strings."
                )
            self.formats = [fmt.strip().lower() for fmt in raw_formats]
            _validate_format_keys(
                self.formats,
                format_registry,
                {"jpg", "png", "npz", "npy"},
                "[analysis.segmentation].formats",
            )

        try:
            self.config = SegmentationConfig().load(
                sub_sec, color_embedding_registry=color_embedding_registry
            )
        except KeyError:
            self.config = {}
            for key in sub_sec.keys():
                self.config[key] = SegmentationConfig().load(
                    _get_section(sub_sec, key),
                    color_embedding_registry=color_embedding_registry,
                )
            try:
                self.config = {}
                for key in sub_sec.keys():
                    self.config[key] = SegmentationConfig().load(
                        _get_section(sub_sec, key),
                        color_embedding_registry=color_embedding_registry,
                    )
            except KeyError as e:
                raise KeyError(
                    "Segmentation config must be either a single or multiple segmentations."
                ) from e

        folder = _get_key(sub_sec, "folder", required=False, type_=Path)
        if not folder:
            assert results is not None
            self.folder = results / "segmentation"
        return self

    def error(self):
        raise ValueError(
            "Use [analysis.segmentation] in the config file to load segmentation."
        )


@dataclass
class AnalysisMassConfig:
    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is used for mass analysis.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for mass analysis."""
    formats: list[str] | None = field(
        default=None,
        metadata={
            "name": "Export formats",
            "help": "Image formats to save for mass analysis.",
            "widget": "format_key_list",
            "format_types": {"jpg", "png", "npz", "npy"},
        },
    )
    """Output formats for mass analysis images."""
    color: "ColorEmbedding | None" = field(
        default=None,
        metadata={
            "name": "Color embedding",
            "widget": "color_key_list",
            "max_rows": 1,
        },
    )
    """Color embedding identifier used for mass conversion.

    The value must be a non-empty key defined in the centralized
    ``[color.*.*]`` registry.
    """
    roi: list[str] = field(
        default_factory=list,
        metadata={
            "name": "ROIs",
            "help": "ROI definitions for mass analysis.",
            "widget": "roi_key_list",
        },
    )
    """ROI names for mass analysis."""
    export: list[str] | None = field(
        default=None,
        metadata={"name": "Export fields", "help": "Mass analysis scalars to save."},
    )
    """Optional selection of mass-analysis scalar fields exported to disk."""
    folder: Path | None = field(
        default=None,
        metadata={
            "name": "Output folder",
            "help": "Directory for mass analysis results.",
            "hidden": True,
        },
    )
    """Path to the results folder for mass analysis."""
    contour_smoother_selection: ContourSmootherSelection = field(
        default_factory=ContourSmootherSelection,
        metadata={"name": "Contour smoother", "help": "Contour smoothing algorithm."},
    )
    """Contour smoother selection and options."""

    @property
    def contour_smoother(self) -> darsia.ContourSmoother | None:
        """Backward-compatible property: returns the built contour smoother."""
        return self.contour_smoother_selection.build()

    def load(
        self,
        sec: dict,
        results: Path | None,
        roi_registry: RoiRegistry | None = None,
        color_embedding_registry: ColorEmbeddingRegistry | None = None,
        format_registry: "FormatRegistry | None" = None,
    ) -> "AnalysisMassConfig":
        sub_sec = _get_section(sec, "mass")

        self.data_selection = _get_key(
            sub_sec, "data_selection", required=False, default=None
        )

        raw_formats = _get_key(sub_sec, "formats", required=False, default=None)
        if raw_formats is None:
            self.formats = None
        else:
            if not isinstance(raw_formats, list):
                raise ValueError("analysis.mass.formats must be a list.")
            if not all(isinstance(fmt, str) for fmt in raw_formats):
                raise ValueError("analysis.mass.formats entries must be strings.")
            self.formats = [fmt.strip().lower() for fmt in raw_formats]
            _validate_format_keys(
                self.formats,
                format_registry,
                {"jpg", "png", "npz", "npy"},
                "[analysis.mass].formats",
            )

        color_key = _get_key(sub_sec, "color", required=True, type_=str).strip()
        if color_embedding_registry is None:
            raise ValueError(
                "analysis.mass.color references [color.*.*], but no "
                "ColorEmbeddingRegistry is available."
            )
        try:
            self.color = color_embedding_registry.resolve(color_key)
        except KeyError as exc:
            raise ValueError(
                f"Unknown analysis.mass.color embedding '{color_key}'."
            ) from exc

        # Load ROIs – support registry-key references as list[str].
        self.roi = _load_roi_key_list(
            sub_sec,
            "roi",
            context="analysis.mass.roi",
            roi_registry=roi_registry,
        )

        # TODO: Can this be unified and simplified across analysis workflows?
        raw_export = _get_key(sub_sec, "export", required=False, default=None)
        if raw_export is None:
            self.export = None
        else:
            if not isinstance(raw_export, list):
                raise ValueError("analysis.mass.export must be a list.")
            if not all(isinstance(mode, str) for mode in raw_export):
                raise ValueError("analysis.mass.export entries must be strings.")
            export_modes = []
            for mode in raw_export:
                stripped_mode = mode.strip()
                if stripped_mode:
                    export_modes.append(stripped_mode.lower())
            invalid_modes = sorted(
                set(export_modes) - SUPPORTED_ANALYSIS_MASS_EXPORT_MODES
            )
            if len(invalid_modes) > 0:
                raise ValueError(
                    "Unsupported [analysis.mass].export entries: "
                    f"{', '.join(invalid_modes)}. "
                    "Supported values: "
                    f"{', '.join(sorted(SUPPORTED_ANALYSIS_MASS_EXPORT_MODES))}."
                )
            # Deduplicate while preserving first-seen order.
            self.export = list(dict.fromkeys(export_modes))

        folder = _get_key(sub_sec, "folder", required=False, type_=Path)
        if not folder:
            assert results is not None
            folder = results / "mass"
        self.folder = folder

        # Load contour smoother
        self.contour_smoother_selection = ContourSmootherSelection().load(sub_sec)

        return self


@dataclass
class AnalysisVolumeConfig:
    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is used for volume analysis.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for volume analysis."""
    formats: list[str] | None = field(
        default=None,
        metadata={
            "name": "Export formats",
            "help": "Image formats to save for volume analysis.",
            "widget": "format_key_list",
            "format_types": {"jpg", "png", "npz", "npy"},
        },
    )
    """Output formats for volume analysis images."""
    roi: list[str] = field(
        default_factory=list,
        metadata={
            "name": "ROIs",
            "help": "ROI definitions for volume analysis.",
            "widget": "roi_key_list",
        },
    )
    """ROI names for volume analysis."""
    folder: Path | None = field(
        default=None,
        metadata={
            "name": "Output folder",
            "help": "Directory for volume analysis results.",
            "hidden": True,
        },
    )
    """Path to the results folder for volume analysis."""

    def load(
        self,
        sec: dict,
        results: Path | None,
        roi_registry: RoiRegistry | None = None,
        format_registry: "FormatRegistry | None" = None,
    ) -> "AnalysisVolumeConfig":
        sub_sec = _get_section(sec, "volume")

        self.data_selection = _get_key(
            sub_sec, "data_selection", required=False, default=None
        )

        raw_formats = _get_key(sub_sec, "formats", required=False, default=None)
        if raw_formats is None:
            self.formats = None
        else:
            if not isinstance(raw_formats, list):
                raise ValueError("analysis.volume.formats must be a list.")
            if not all(isinstance(fmt, str) for fmt in raw_formats):
                raise ValueError("analysis.volume.formats entries must be strings.")
            self.formats = [fmt.strip().lower() for fmt in raw_formats]
            _validate_format_keys(
                self.formats,
                format_registry,
                {"jpg", "png", "npz", "npy"},
                "[analysis.volume].formats",
            )

        # Load ROIs – support registry-key references as list[str].
        self.roi = _load_roi_key_list(
            sub_sec,
            "roi",
            context="analysis.volume.roi",
            roi_registry=roi_registry,
        )

        folder = _get_key(sub_sec, "folder", required=False, type_=Path)
        if not folder:
            assert results is not None
            self.folder = results / "volume"
        return self


@dataclass
class AnalysisExpertKnowledgeConfig:
    """Configuration for expert-knowledge ROI constraints on analysis fields."""

    saturation_g: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Saturation ROIs",
            "help": "ROI keys where saturation_g constraints apply.",
            "widget": "roi_key_list",
        },
    )
    """ROI registry keys constraining where saturation_g may be non-zero."""
    concentration_aq: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Concentration ROIs",
            "help": "ROI keys where concentration_aq constraints apply.",
            "widget": "roi_key_list",
        },
    )
    """ROI registry keys constraining where concentration_aq may be non-zero."""

    def load(
        self, sec: dict, roi_registry: RoiRegistry | None = None
    ) -> "AnalysisExpertKnowledgeConfig":
        sub_sec = _get_section(sec, "expert_knowledge")

        self.saturation_g = _get_key(
            sub_sec, "saturation_g", required=False, default=[]
        )
        self.concentration_aq = _get_key(
            sub_sec, "concentration_aq", required=False, default=[]
        )

        if not isinstance(self.saturation_g, list) or not all(
            isinstance(key, str) for key in self.saturation_g
        ):
            raise ValueError(
                "analysis.expert_knowledge.saturation_g must be a list[str]."
            )
        if not isinstance(self.concentration_aq, list) or not all(
            isinstance(key, str) for key in self.concentration_aq
        ):
            raise ValueError(
                "analysis.expert_knowledge.concentration_aq must be a list[str]."
            )

        # Validate registry references eagerly when provided.
        if roi_registry is not None:
            if len(self.saturation_g) > 0:
                roi_registry.resolve_rois(self.saturation_g)
            if len(self.concentration_aq) > 0:
                roi_registry.resolve_rois(self.concentration_aq)
        elif len(self.saturation_g) > 0 or len(self.concentration_aq) > 0:
            raise ValueError(
                "analysis.expert_knowledge requires a loaded ROI registry when "
                "saturation_g or concentration_aq keys are provided."
            )

        return self


@dataclass
class AnalysisFingersConfig:
    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is used for fingers analysis.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for fingers analysis."""
    formats: list[str] | None = field(
        default=None,
        metadata={
            "name": "Export formats",
            "help": "Image formats to save for fingers analysis.",
            "widget": "format_key_list",
            "format_types": {"jpg", "png", "npz", "npy"},
        },
    )
    """Output formats for fingers analysis images."""
    config: FingersConfig | dict[str, FingersConfig] = field(
        default_factory=lambda: FingersConfig(),
        metadata={"name": "Config", "hidden": True},
    )
    folder: Path | None = field(
        default=None,
        metadata={
            "name": "Output folder",
            "help": "Directory for fingers analysis results.",
            "hidden": True,
        },
    )
    """Path to the results folder for fingers analysis."""

    def load(
        self,
        sec: dict,
        results: Path | None,
        roi_registry: RoiRegistry | None = None,
        color_embedding_registry: ColorEmbeddingRegistry | None = None,
        format_registry: "FormatRegistry | None" = None,
    ) -> "AnalysisFingersConfig":
        # Allow for two scenarios: single fingers or multiple fingers
        sub_sec = _get_section(sec, "fingers")

        self.data_selection = _get_key(
            sub_sec, "data_selection", required=False, default=None
        )

        raw_formats = _get_key(sub_sec, "formats", required=False, default=None)
        if raw_formats is None:
            self.formats = None
        else:
            if not isinstance(raw_formats, list):
                raise ValueError("analysis.fingers.formats must be a list.")
            if not all(isinstance(fmt, str) for fmt in raw_formats):
                raise ValueError("analysis.fingers.formats entries must be strings.")
            self.formats = [fmt.strip().lower() for fmt in raw_formats]
            _validate_format_keys(
                self.formats,
                format_registry,
                {"jpg", "png", "npz", "npy"},
                "[analysis.fingers].formats",
            )

        try:
            self.config = FingersConfig().load(
                sub_sec,
                roi_registry=roi_registry,
                color_embedding_registry=color_embedding_registry,
            )
        except KeyError:
            self.config = {}
            for key in sub_sec.keys():
                self.config[key] = FingersConfig().load(
                    _get_section(sub_sec, key),
                    roi_registry=roi_registry,
                    color_embedding_registry=color_embedding_registry,
                )
            try:
                self.config = {}
                for key in sub_sec.keys():
                    self.config[key] = FingersConfig().load(
                        _get_section(sub_sec, key),
                        roi_registry=roi_registry,
                        color_embedding_registry=color_embedding_registry,
                    )
            except KeyError as e:
                raise KeyError(
                    "Fingers config must be either a single or multiple fingers."
                ) from e

        folder = _get_key(sub_sec, "folder", required=False, type_=Path)
        if not folder:
            assert results is not None
            self.folder = results / "fingers"
        return self

    def error(self):
        raise ValueError("Use [analysis.fingers] in the config file to load fingers.")


@dataclass
class AnalysisCroppingConfig:
    data_selection: str | list[str] | None = field(
        default=None,
        metadata={
            "name": "Data selection",
            "help": "Registry key name(s) whose data is used for cropping.",
            "widget": "registry_key_list",
        },
    )
    """Name(s) of data registry entries to use for cropping."""
    formats: list[str] = field(
        default_factory=lambda: ["jpg"],
        metadata={
            "name": "Export formats",
            "help": (
                "Image formats to save for cropping. Only registry entries of type "
                "jpg, npz, png, or npy are offered."
            ),
            "widget": "format_key_list",
            "format_types": {"jpg", "npz", "png", "npy"},
        },
    )
    """Output formats for cropping images."""

    def load(
        self,
        sec: dict,
        format_registry: "FormatRegistry | None" = None,
    ) -> "AnalysisCroppingConfig":
        sub_sec = _get_section(sec, "cropping")

        self.data_selection = _get_key(
            sub_sec, "data_selection", required=False, default=None
        )

        raw_formats = _get_key(sub_sec, "formats", default=["jpg"], required=False)
        if not isinstance(raw_formats, list):
            raise ValueError("analysis.cropping.formats must be a list.")
        if not all(isinstance(fmt, str) for fmt in raw_formats):
            raise ValueError("analysis.cropping.formats entries must be strings.")
        self.formats = [fmt.strip().lower() for fmt in raw_formats]
        _validate_format_keys(
            self.formats,
            format_registry,
            {"jpg", "npz", "png", "npy"},
            "[analysis.cropping].formats",
        )
        return self


@dataclass
class AnalysisConfig:
    random_traverse: bool = field(
        default=False,
        metadata={
            "name": "Random traverse",
            "help": "Process images in random order instead of chronological.",
        },
    )
    """Whether to randomly traverse the data."""
    cropping: AnalysisCroppingConfig | None = field(
        default=None, metadata={"name": "Cropping"}
    )
    """Analysis cropping configuration."""
    segmentation: AnalysisSegmentationConfig | None = field(
        default=None, metadata={"name": "Segmentation"}
    )
    """Analysis segmentation configuration."""
    mass: AnalysisMassConfig | None = field(default=None, metadata={"name": "Mass"})
    """Analysis mass configuration."""
    volume: AnalysisVolumeConfig | None = field(
        default=None, metadata={"name": "Volume"}
    )
    """Analysis volume configuration."""
    fingers: AnalysisFingersConfig | None = field(
        default=None, metadata={"name": "Fingers"}
    )
    """Analysis fingers configuration."""
    thresholding: AnalysisThresholdingConfig | None = field(
        default=None, metadata={"name": "Thresholding"}
    )
    """Analysis thresholding configuration."""
    expert_knowledge: AnalysisExpertKnowledgeConfig = field(
        default_factory=AnalysisExpertKnowledgeConfig,
        metadata={"name": "Expert knowledge"},
    )
    """Expert knowledge constraints for selected scalar analysis fields."""

    def load(
        self,
        path: Path,
        data: Path | None,
        results: Path | None,
        roi_registry: RoiRegistry | None = None,
        format_registry: FormatRegistry | None = None,
        color_embedding_registry: ColorEmbeddingRegistry | None = None,
    ) -> "AnalysisConfig":
        sec = _get_section_from_toml(path, "analysis")

        self.random_traverse = _get_key(
            sec, "random_traverse", required=False, default=False, type_=bool
        )

        # Config to load analysis cropping
        try:
            self.cropping = AnalysisCroppingConfig().load(
                sec, format_registry=format_registry
            )
        except KeyError:
            warn("No analysis cropping found. Use [analysis.cropping].")
            self.cropping = AnalysisCroppingConfig()  # Default to empty cropping config

        # Config to load analysis segmentation
        try:
            self.segmentation = AnalysisSegmentationConfig().load(
                sec,
                results,
                color_embedding_registry=color_embedding_registry,
                format_registry=format_registry,
            )
        except KeyError:
            warn("No analysis segmentation found. Use [analysis.segmentation].")
            self.segmentation = None

        # Config to load analysis mass
        try:
            self.mass = AnalysisMassConfig().load(
                sec,
                results,
                roi_registry=roi_registry,
                color_embedding_registry=color_embedding_registry,
                format_registry=format_registry,
            )
        except KeyError:
            warn("No analysis mass found. Use [analysis.mass].")
            self.mass = None

        # Config to load analysis volume
        try:
            self.volume = AnalysisVolumeConfig().load(
                sec,
                results,
                roi_registry=roi_registry,
                format_registry=format_registry,
            )
        except KeyError:
            warn("No analysis volume found. Use [analysis.volume].")
            self.volume = None

        # Config to load analysis fingers
        try:
            self.fingers = AnalysisFingersConfig().load(
                sec,
                results,
                roi_registry=roi_registry,
                color_embedding_registry=color_embedding_registry,
                format_registry=format_registry,
            )
        except KeyError:
            warn("No analysis fingers found. Use [analysis.fingers].")
            self.fingers = None

        # Config to load analysis thresholding
        try:
            self.thresholding = AnalysisThresholdingConfig().load(
                sec,
                results,
                color_embedding_registry=color_embedding_registry,
                format_registry=format_registry,
            )
        except KeyError:
            warn("No analysis thresholding found. Use [analysis.thresholding].")
            self.thresholding = None

        # Config to load analysis expert knowledge. Missing section is a no-op default.
        try:
            self.expert_knowledge = AnalysisExpertKnowledgeConfig().load(
                sec, roi_registry=roi_registry
            )
        except KeyError:
            self.expert_knowledge = AnalysisExpertKnowledgeConfig()

        return self
