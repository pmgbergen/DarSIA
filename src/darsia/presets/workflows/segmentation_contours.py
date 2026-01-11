"""Util for contour plots."""

import logging

import darsia
from darsia.utils.augmented_plotting import plot_contour_on_image
from darsia.presets.workflows.fluidflower_config import SegmentationConfig

logger = logging.getLogger(__name__)


class SegmentationContours:
    """Threshold based segmentation analysis."""

    def __init__(
        self, config: SegmentationConfig | dict[str, SegmentationConfig]
    ) -> None:
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = {"": config}

    def extract_mask(
        self, img: darsia.ScalarImage, thresholds: list[float]
    ) -> list[darsia.ScalarImage]:
        """Extract phase based on thresholding.

        Args:
            img: Signal to segment.
            label: Label to extract.

        Returns:
            darsia.Image: Segmented phase (boolean) image.

        """
        masks = []
        for i in range(len(thresholds)):
            lower = thresholds[i]
            next = i + 1
            upper = thresholds[next] if next < len(thresholds) else float("inf")
            mask = (img.img >= lower) & (img.img <= upper)
            masks.append(darsia.ScalarImage(img=mask, **img.metadata()))
        return masks

    def add_contours(
        self,
        img: darsia.Image,
        masks: list[darsia.ScalarImage],
        color: list[float],
        alpha: list[float],
        linewidth: int = 2,
    ) -> darsia.Image:
        """Add contours to image based on segmentation of mass.

        Args:
            img: Image to add contours to.
            masks: Mask as basis for contour extraction.

        Returns:
            Image with contours added.
        """
        contour_image = img.copy()

        for mask, alpha in zip(masks, alpha):
            contour_image = plot_contour_on_image(
                img=contour_image,
                mask=[mask],
                color=[color],
                alpha=[alpha],
                thickness=linewidth,
                return_image=True,
            )
        return contour_image

    def __call__(
        self,
        img,
        saturation_g: darsia.Image,
        concentration_aq: darsia.Image,
        mass: darsia.Image,
    ) -> darsia.Image:
        contour_img = img.copy()
        for segmentation_config in self.config.values():
            # Select values based on mode
            mode = segmentation_config.mode
            if mode == "saturation_g":
                values = saturation_g
            elif mode == "concentration_aq":
                values = concentration_aq
            elif mode == "mass":
                values = mass
            else:
                raise ValueError(f"Unknown label {mode} in segmentation config.")

            # Extract masks based on thresholds
            masks = self.extract_mask(values, segmentation_config.thresholds)

            # Add contours to image
            contour_img = self.add_contours(
                contour_img,
                masks,
                segmentation_config.color,
                segmentation_config.alpha,
                segmentation_config.linewidth,
            )
        return contour_img
