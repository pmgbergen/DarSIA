"""Factory for building restoration objects from workflow configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import darsia
from darsia.presets.workflows.config.restoration import RestorationConfig

if TYPE_CHECKING:
    from darsia.presets.workflows.rig import Rig


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
        return None

    method = restoration_config.method

    if method is None:
        return None

    elif method == "volume_average":
        from darsia.presets.workflows.config.restoration import VolumeAveragingConfig

        options = restoration_config.options
        if not isinstance(options, VolumeAveragingConfig):
            options = VolumeAveragingConfig()
        rev_size = options.rev_size
        image_porosity = fluidflower.image_porosity
        return darsia.VolumeAveraging(
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
            if options.weight == "porosity":
                weight = fluidflower.image_porosity
            elif options.weight == "boolean-porosity":
                weight = fluidflower.boolean_porosity
            else:
                raise ValueError(
                    f"Unknown weight string '{options.weight}'. "
                    "Valid string values are 'porosity' and 'boolean-porosity'. "
                    "For a scalar weight, provide a numeric value in the config."
                )
            tvd_method = "heterogeneous bregman"
        else:
            weight = options.weight

        return darsia.TVD(
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
