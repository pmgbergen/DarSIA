"""Factory for building restoration objects from workflow configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

import darsia
from darsia.presets.workflows.config.restoration import (
    RestorationConfig,
    RestorationStepConfig,
    TVDConfig,
    VolumeAveragingConfig,
)

if TYPE_CHECKING:
    from darsia.presets.workflows.processing.image_porosity import ImagePorosity
    from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


class RestorationSteps(darsia.CombinedModel):
    """Restoration steps applied in order, and the grid of their result."""

    def __init__(self, models: list, grid: darsia.Image) -> None:
        """
        Parameters
        ----------
        models : list
            Models of the steps, in the order of application.
        grid : darsia.Image
            Image on the grid of the restored images: the rig baseline after the
            resize steps (the baseline itself without them).

        """
        super().__init__(models)
        self.grid = grid
        """Image on the grid of the restored images."""


class RestorationMaskFactory:
    """Factory for predefined restoration ignore masks, on a given grid."""

    def __init__(
        self,
        fluidflower: "Rig",
        grid: darsia.Image,
        image_porosity: "ImagePorosity | None" = None,
    ) -> None:
        self.fluidflower = fluidflower
        self.grid = grid
        self.image_porosity = image_porosity
        self._ignore_mask_builders = {
            "image_porosity": self._image_porosity_ignore_mask,
            "boolean_porosity": self._boolean_porosity_ignore_mask,
            "inner_labels": self._inner_labels_ignore_mask,
        }

    def _image_porosity_ignore_mask(self) -> np.ndarray:
        return _required(self.image_porosity).map_on(self.grid).img <= 0

    def _boolean_porosity_ignore_mask(self) -> np.ndarray:
        boolean_porosity = _required(self.image_porosity).boolean_on(self.grid)
        return ~boolean_porosity.img.astype(bool)

    def _inner_labels_ignore_mask(self) -> np.ndarray:
        inner_labels = darsia.transfer(self.fluidflower.inner_labels, self.grid)
        return ~inner_labels.img.astype(bool)

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


def _required(image_porosity: "ImagePorosity | None") -> "ImagePorosity":
    if image_porosity is None:
        raise ValueError(
            "This restoration uses the image porosity, but none was given to "
            "build_restoration."
        )
    return image_porosity


def build_restoration(
    restoration_config: RestorationConfig | None,
    fluidflower: "Rig",
    image_porosity: "ImagePorosity | None" = None,
) -> RestorationSteps | None:
    """Build the restoration from configuration.

    The steps are built in order. A resize step changes the grid; the rig data of
    later steps (porosity, ignore masks, REV size in pixels) is taken on the new grid.

    Parameters
    ----------
    restoration_config : RestorationConfig | None
        Parsed restoration configuration, or None if no
        restoration section was present in the config file.
    fluidflower : 'Rig'
        The loaded rig instance (provides the baseline and inner labels).
    image_porosity : ImagePorosity, optional
        Stored image porosity (`processing.image_porosity.load_image_porosity`),
        read only if a step or ignore mask uses it.

    Returns
    -------
    RestorationSteps or None
        The active steps and the grid of their result, or None without active steps.

    """
    if restoration_config is None:
        logger.info(
            "No restoration configuration found. "
            "See in the template config file for options under [restoration]. "
            "Proceeding without restoration."
        )
        return None

    steps = [step for step in restoration_config.steps if step.active]
    if len(steps) == 0:
        return None

    grid = fluidflower.baseline
    models = []
    for step in steps:
        model, grid = _build_step(
            step, restoration_config.ignore, fluidflower, grid, image_porosity
        )
        models.append(model)
    return RestorationSteps(models, grid)


def _build_step(
    step: RestorationStepConfig,
    ignore: list[str],
    fluidflower: "Rig",
    grid: darsia.Image,
    image_porosity: "ImagePorosity | None" = None,
) -> tuple:
    """Model of one step on a grid, and the grid of its result."""
    if step.method == "resize":
        resize = step.resize_options
        if resize is None:
            raise ValueError("A resize step needs options (scale or target_shape).")
        if resize.mode == "scale":
            model = darsia.Resize(
                fx=resize.scale, fy=resize.scale, interpolation="inter_area"
            )
        else:
            model = darsia.Resize(shape=resize.target_shape, interpolation="inter_area")
        return model, model(grid)

    mask_factory = RestorationMaskFactory(fluidflower, grid, image_porosity)
    ignore_mask = mask_factory.build_ignore_mask(ignore)
    active_mask = None if ignore_mask is None else (~ignore_mask).astype(float)

    if step.method == "volume_average":
        volume_averaging = step.volume_averaging_options or VolumeAveragingConfig()
        if volume_averaging.mask == "image_porosity":
            # Work on a copy as the stored image porosity is shared.
            weight = _required(image_porosity).map_on(grid).copy()
        elif volume_averaging.mask == "boolean_porosity":
            weight = _required(image_porosity).boolean_on(grid).astype(float)
        else:
            weight = darsia.ones_like(grid, mode="voxels", dtype=float)
        if active_mask is not None:
            weight.img *= active_mask
        model = darsia.VolumeAveraging(
            rev=darsia.REV(size=volume_averaging.rev_size, img=grid),
            mask=weight,
        )
        return model, grid

    # TVD
    options = step.tvd_options or TVDConfig()

    # Resolve porosity-based weight strings to actual arrays, forcing
    # "heterogeneous bregman" as the TVD method in those cases.
    tvd_method = options.method
    if isinstance(options.weight, str):
        if options.weight == "image_porosity":
            weight = _required(image_porosity).map_on(grid)
        elif options.weight == "boolean_porosity":
            weight = _required(image_porosity).boolean_on(grid)
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

    model = darsia.TVD(
        method=tvd_method,
        weight=weight,
        max_num_iter=options.max_num_iter,
        eps=options.eps,
        omega=options.omega,
        regularization=options.regularization,
        **options.kwargs,
    )
    return model, grid
