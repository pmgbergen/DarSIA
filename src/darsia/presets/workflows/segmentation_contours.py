"""Util for contour plots."""

import logging

import darsia
from darsia.utils.augmented_plotting import plot_contour_on_image

logger = logging.getLogger(__name__)


class SegmentationContours:
    """Threshold based segmentation analysis."""

    def __init__(
        self,
        thresholds: dict[str, list[list[float]]],
        colors: dict[str, list[float]] | None = None,
        alphas: dict[str, list[float]] | None = None,
        linewidth: int = 2,
    ):
        self.thresholds = thresholds
        self.colors = colors or {}
        self.alphas = alphas or {
            label: [1.0] * len(thresholds[label]) for label in thresholds
        }
        self.linewidth = linewidth

        # Safety checks
        assert set(self.thresholds.keys()) == set(self.colors.keys()), (
            "Thresholds and colors must have the same labels."
        )
        assert set(self.thresholds.keys()) == set(self.alphas.keys()), (
            "Thresholds and alphas must have the same labels."
        )
        assert all(
            len(self.thresholds[label]) == len(self.colors[label])
            for label in self.thresholds
        ), "Number of thresholds and colors must match for each label."
        assert all(
            len(self.thresholds[label]) == len(self.alphas[label])
            for label in self.thresholds
        ), "Number of thresholds and alphas must match for each label."

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
        colors: list[list[float]],
        alphas: list[float],
        masks: list[darsia.ScalarImage],
    ) -> darsia.Image:
        """Add contours to image based on segmentation of mass.

        Args:
            img: Image to add contours to.
            masks: Mask as basis for contour extraction.

        Returns:
            Image with contours added.
        """
        contour_image = img.copy()

        for color, mask, alpha in zip(colors, masks, alphas):
            contour_image = plot_contour_on_image(
                img=contour_image,
                mask=[mask],
                color=[color],
                alpha=[alpha],
                thickness=self.linewidth,
                return_image=True,
            )
        return contour_image

    def __call__(self, img, values: dict[str, darsia.Image]) -> darsia.Image:
        contour_img = img.copy()
        for label in values.keys():
            masks = self.extract_mask(values[label], self.thresholds[label])
            contour_img = self.add_contours(
                contour_img, self.colors[label], self.alphas[label], masks
            )
        contour_img.show()
        return contour_img
