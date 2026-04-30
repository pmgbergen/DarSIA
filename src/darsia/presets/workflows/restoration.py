"""Factory for building restoration objects from workflow configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import logging
import darsia
from darsia.presets.workflows.config.restoration import RestorationConfig

if TYPE_CHECKING:
    from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)

class RestorationMaskFactory:
    """Factory for predefined restoration ignore masks."""

    def __init__(self, fluidflower: "Rig") -> None:
        self.fluidflower = fluidflower
        self._ignore_mask_builders = {
            "image_porosity": self._image_porosity_ignore_mask,
            "boolean_porosity": self._boolean_porosity_ignore_mask,
            "inner_labels": self._inner_labels_ignore_mask
        }

    def _image_porosity_ignore_mask(self) -> np.ndarray:
        return self.fluidflower.image_porosity.img <= 0

    def _boolean_porosity_ignore_mask(self) -> np.ndarray:
        return ~self.fluidflower.boolean_porosity.img.astype(bool)

    def _inner_labels_ignore_mask(self) -> np.ndarray:
        return ~self.fluidflower.inner_labels.img.astype(bool)

    def build_ignore_mask(self, mask_names: list[str]) -> np.ndarray | None:
        if len(mask_names) == 0:
            return None

        ignore_mask: np.ndarray | None = None
        for name in mask_names:
            if name not in self._ignore_mask_builders:
                raise ValueError(
                    f"Unknown restoration ignore mask '{name}'. "
                    f"Valid values are: {list(self._ignore_mask_builders.keys())}."
                )
            current_ignore_mask = self._ignore_mask_builders[name]()
            ignore_mask = (
                current_ignore_mask
                if ignore_mask is None
                else np.logical_or(ignore_mask, current_ignore_mask)
            )

        return ignore_mask


def build_restoration(
    restoration_config: RestorationConfig | None,
    fluidflower: "Rig",
) -> darsia.VolumeAveraging | darsia.TVD | None:
    """Build a restoration object from configuration.

    Args:
        restoration_config: Parsed restoration configuration, or None if no
            restoration section was present in the config file.
        fluidflower: The loaded rig instance (provides baseline and porosity).

    Returns:
        A callable restoration object, or None if no restoration is configured.

    """
    if restoration_config is None:
        logger.info(
            "No restoration configuration found. "
            "See in the template config file for options under [restoration]. "
            "Proceeding without restoration."
        )
        return None

    method = restoration_config.method

    if method is None:
        return None

    mask_factory = RestorationMaskFactory(fluidflower)
    ignore_mask = mask_factory.build_ignore_mask(restoration_config.ignore)
    active_mask = None if ignore_mask is None else (~ignore_mask).astype(float)

    if method == "volume_average":
        from darsia.presets.workflows.config.restoration import VolumeAveragingConfig

        options = restoration_config.options
        if not isinstance(options, VolumeAveragingConfig):
            options = VolumeAveragingConfig()
        rev_size = options.rev_size
        # Work on a copy as the porosity image is shared in the rig object.
        image_porosity = fluidflower.image_porosity.copy()
        if active_mask is not None:
            image_porosity.img *= active_mask
        restoration = darsia.VolumeAveraging(
            rev=darsia.REV(size=rev_size, img=fluidflower.baseline),
            mask=image_porosity,
        )

    elif method == "tvd":
        from darsia.presets.workflows.config.restoration import TVDConfig

        options = restoration_config.options
        if not isinstance(options, TVDConfig):
            options = TVDConfig()

        # Resolve porosity-based weight strings to actual arrays, forcing
        # "heterogeneous bregman" as the TVD method in those cases.
        tvd_method = options.method
        if isinstance(options.weight, str):
            if options.weight == "image_porosity":
                weight = fluidflower.image_porosity
            elif options.weight == "boolean_porosity":
                weight = fluidflower.boolean_porosity
            else:
                raise ValueError(
                    f"Unknown weight string '{options.weight}'. "
                    "Valid string values are 'image_porosity' and 'boolean_porosity'. "
                    "For a scalar weight, provide a numeric value in the config."
                )
            tvd_method = "heterogeneous bregman"
        else:
            weight = options.weight

        if isinstance(weight, darsia.Image):
            weight = weight.img

        if active_mask is not None:
            weight = np.multiply(weight, active_mask)
            tvd_method = "heterogeneous bregman"

        restoration = darsia.TVD(
            method=tvd_method,
            weight=weight,
            max_num_iter=options.max_num_iter,
            eps=options.eps,
            omega=options.omega,
            regularization=options.regularization,
            **options.kwargs,
        )

    else:
        raise NotImplementedError(f"Restoration method '{method}' not supported.")

    return restoration
