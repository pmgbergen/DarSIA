"""Expert-knowledge adapter for constraining scalar analysis fields by ROIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

import darsia
from darsia.utils.standard_images import roi_to_mask

if TYPE_CHECKING:
    from darsia.presets.workflows.config.analysis import AnalysisExpertKnowledgeConfig
    from darsia.presets.workflows.config.roi import RoiConfig
    from darsia.presets.workflows.config.roi_registry import RoiRegistry


@dataclass
class ExpertKnowledgeAdapter:
    """Adapter converting expert-knowledge ROI config into reusable image masks."""

    saturation_g_rois: dict[str, "RoiConfig"] = field(default_factory=dict)
    concentration_aq_rois: dict[str, "RoiConfig"] = field(default_factory=dict)
    _mask_cache: dict[tuple[str, tuple], np.ndarray] = field(
        default_factory=dict, init=False
    )

    @classmethod
    def from_config(
        cls,
        config: "AnalysisExpertKnowledgeConfig | None",
        roi_registry: "RoiRegistry | None",
    ) -> "ExpertKnowledgeAdapter":
        if config is None:
            return cls()

        saturation_rois = {}
        concentration_rois = {}

        if roi_registry is not None:
            if len(config.saturation_g) > 0:
                saturation_rois = roi_registry.resolve_rois(config.saturation_g)
            if len(config.concentration_aq) > 0:
                concentration_rois = roi_registry.resolve_rois(config.concentration_aq)

        return cls(
            saturation_g_rois=saturation_rois,
            concentration_aq_rois=concentration_rois,
        )

    def _signature(self, image: darsia.Image) -> tuple:
        metadata = image.metadata()
        dimensions = tuple(np.asarray(metadata.get("dimensions", []), dtype=float))
        origin = tuple(np.asarray(metadata.get("origin", []), dtype=float))
        indexing = str(metadata.get("indexing", ""))
        space_dim = int(metadata.get("space_dim", 2))
        return (
            tuple(int(v) for v in image.num_voxels),
            dimensions,
            origin,
            indexing,
            space_dim,
        )

    def _rois_for_mode(self, mode: str) -> dict[str, "RoiConfig"]:
        if mode == "saturation_g":
            return self.saturation_g_rois
        if mode == "concentration_aq":
            return self.concentration_aq_rois
        return {}

    def mask_for(self, image: darsia.Image, mode: str) -> np.ndarray | None:
        rois = self._rois_for_mode(mode)
        if len(rois) == 0:
            return None

        cache_key = (mode, self._signature(image))
        if cache_key not in self._mask_cache:
            mask_image = roi_to_mask([roi.roi for roi in rois.values()], image)
            self._mask_cache[cache_key] = mask_image.img.astype(bool)
        return self._mask_cache[cache_key]

    def apply(self, image: darsia.Image | None, mode: str) -> darsia.Image | None:
        if image is None:
            return None
        mask = self.mask_for(image, mode)
        if mask is None:
            return image

        constrained = image.copy()
        constrained.img[~mask] = 0.0
        return constrained

