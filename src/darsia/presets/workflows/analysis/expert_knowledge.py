"""Expert-knowledge adapter for constraining scalar analysis fields by ROIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

import darsia
from darsia.utils.standard_images import roi_to_mask

if TYPE_CHECKING:
    from darsia.presets.workflows.config.expert_knowledge import ExpertKnowledgeConfig
    from darsia.presets.workflows.config.roi import RoiConfig
    from darsia.presets.workflows.config.roi_registry import RoiRegistry


@dataclass
class ExpertKnowledgeAdapter:
    """Adapter converting expert-knowledge ROI config into reusable image masks."""

    saturation_g_rois: dict[str, "RoiConfig"] = field(default_factory=dict)
    concentration_aq_rois: dict[str, "RoiConfig"] = field(default_factory=dict)
    no_saturation_g_rois: dict[str, "RoiConfig"] = field(default_factory=dict)
    no_concentration_aq_rois: dict[str, "RoiConfig"] = field(default_factory=dict)
    _mask_cache: dict[tuple[str, tuple], np.ndarray] = field(
        default_factory=dict, init=False
    )

    @classmethod
    def from_config(
        cls,
        config: "ExpertKnowledgeConfig | None",
        roi_registry: "RoiRegistry | None",
    ) -> "ExpertKnowledgeAdapter":
        if config is None:
            return cls()

        saturation_rois = {}
        concentration_rois = {}
        no_saturation_rois = {}
        no_concentration_rois = {}

        if roi_registry is not None:
            if len(config.saturation_g) > 0:
                saturation_rois = roi_registry.resolve_rois(config.saturation_g)
            if len(config.concentration_aq) > 0:
                concentration_rois = roi_registry.resolve_rois(config.concentration_aq)
            if len(config.no_saturation_g) > 0:
                no_saturation_rois = roi_registry.resolve_rois(config.no_saturation_g)
            if len(config.no_concentration_aq) > 0:
                no_concentration_rois = roi_registry.resolve_rois(
                    config.no_concentration_aq
                )

        return cls(
            saturation_g_rois=saturation_rois,
            concentration_aq_rois=concentration_rois,
            no_saturation_g_rois=no_saturation_rois,
            no_concentration_aq_rois=no_concentration_rois,
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

    def _negative_rois_for_mode(self, mode: str) -> dict[str, "RoiConfig"]:
        if mode == "saturation_g":
            return self.no_saturation_g_rois
        if mode == "concentration_aq":
            return self.no_concentration_aq_rois
        return {}

    def mask_for(self, image: darsia.Image, mode: str) -> np.ndarray | None:
        """Boolean "keep" mask for `mode` - True where the field may stay non-zero.

        Combines two independent rules: a positive ROI union (only keep
        inside it, if configured) and a negative ROI union (always zero
        inside it, if configured). Where both are configured and overlap,
        the negative rule wins - the overlapping region is zeroed even
        though it's also inside a positive ROI.
        """
        positive_rois = self._rois_for_mode(mode)
        negative_rois = self._negative_rois_for_mode(mode)
        if len(positive_rois) == 0 and len(negative_rois) == 0:
            return None

        cache_key = (mode, self._signature(image))
        if cache_key not in self._mask_cache:
            if len(positive_rois) > 0:
                keep_mask = roi_to_mask(
                    [roi.roi for roi in positive_rois.values()], image
                ).img.astype(bool)
            else:
                keep_mask = np.ones(image.img.shape, dtype=bool)
            if len(negative_rois) > 0:
                exclude_mask = roi_to_mask(
                    [roi.roi for roi in negative_rois.values()], image
                ).img.astype(bool)
                keep_mask = keep_mask & ~exclude_mask
            self._mask_cache[cache_key] = keep_mask
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
