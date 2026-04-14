"""Util for contour plots."""

import logging
from types import SimpleNamespace

import cv2
import numpy as np

import darsia
from darsia.presets.workflows.config.fluidflower_config import SegmentationConfig
from darsia.presets.workflows.mode_resolution import resolve_mode_image
from darsia.utils.augmented_plotting import plot_contour_on_image

logger = logging.getLogger(__name__)


def _compose_mass_analysis_result(
    saturation_g: darsia.Image | None,
    concentration_aq: darsia.Image | None,
    mass: darsia.Image | None,
):
    return SimpleNamespace(
        saturation_g=saturation_g,
        concentration_aq=concentration_aq,
        mass=mass,
        mass_g=None,
        mass_aq=None,
    )


class SimpleSegmentation:
    """Simple threshold based segmentation."""

    def __init__(self, mode: str, threshold: float) -> None:
        self.mode = mode
        """Type for segmentation."""
        self.threshold = threshold
        """Threshold for segmentation."""

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

    def __call__(
        self,
        img: darsia.Image,
        saturation_g: darsia.Image | None,
        concentration_aq: darsia.Image | None,
        mass: darsia.Image | None,
        mass_analysis_result=None,
        colorrange_config=None,
    ) -> darsia.Image:
        if mass_analysis_result is None:
            mass_analysis_result = _compose_mass_analysis_result(
                saturation_g, concentration_aq, mass
            )
        values = resolve_mode_image(
            self.mode,
            img,
            mass_analysis_result=mass_analysis_result,
            colorrange_config=colorrange_config,
        )

        # Extract masks based on thresholds
        mask = self.extract_mask(values, [self.threshold])[0]
        return mask


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
        thresholds: list[float],
        color: list[float],
        alpha: list[float],
        values_config,
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

        if values_config.show_values:
            contour_image = self.add_contour_values(
                contour_image=contour_image,
                masks=masks,
                thresholds=thresholds,
                values_config=values_config,
            )

        return contour_image

    @staticmethod
    def _boxes_overlap(
        box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
    ) -> bool:
        """Check overlap between two axis-aligned boxes (x0, y0, x1, y1)."""
        return not (
            box_a[2] <= box_b[0]
            or box_b[2] <= box_a[0]
            or box_a[3] <= box_b[1]
            or box_b[3] <= box_a[1]
        )

    @staticmethod
    def _format_threshold(value: float, value_format: str) -> str:
        """Format contour threshold value for plotting."""
        try:
            return value_format.format(value)
        except (ValueError, KeyError):
            return f"{value}"

    def _select_label_positions(
        self,
        contour: np.ndarray,
        min_distance_px: float,
        max_per_contour: int,
        density: float,
        existing_positions: list[tuple[int, int]],
        existing_boxes: list[tuple[int, int, int, int]],
        text: str,
        font_scale: float,
        thickness: int,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
        """Select non-overlapping positions for label text on one contour."""
        if len(contour) < 3 or max_per_contour <= 0:
            return [], []

        points = contour[:, 0, :] if contour.ndim == 3 else contour
        arc_length = cv2.arcLength(contour.astype(np.float32), closed=True)
        if arc_length < min_distance_px:
            return [], []

        target_count = int(np.ceil((arc_length / min_distance_px) * max(density, 0.0)))
        target_count = max(1, min(max_per_contour, target_count))
        stride = max(1, len(points) // target_count)

        selected_positions: list[tuple[int, int]] = []
        selected_boxes: list[tuple[int, int, int, int]] = []
        text_size, baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        for idx in range(0, len(points), stride):
            if len(selected_positions) >= max_per_contour:
                break
            x, y = int(points[idx][0]), int(points[idx][1])
            candidate_position = (x, y)

            # Enforce distance to previous labels.
            if any(
                np.hypot(x - px, y - py) < min_distance_px
                for px, py in existing_positions + selected_positions
            ):
                continue

            # Estimate box from text anchor (bottom-left for cv2.putText).
            x0 = x
            y0 = y - text_size[1] - baseline
            x1 = x + text_size[0]
            y1 = y + baseline
            candidate_box = (x0, y0, x1, y1)
            if any(
                self._boxes_overlap(candidate_box, box)
                for box in existing_boxes + selected_boxes
            ):
                continue

            selected_positions.append(candidate_position)
            selected_boxes.append(candidate_box)

        return selected_positions, selected_boxes

    def add_contour_values(
        self,
        contour_image: darsia.Image,
        masks: list[darsia.ScalarImage],
        thresholds: list[float],
        values_config,
    ) -> darsia.Image:
        """Add threshold labels near contours."""
        base = contour_image.img.astype(np.uint8)
        overlay = np.copy(base)

        alpha = max(0.0, min(1.0, values_config.value_alpha))
        font_scale = max(0.1, values_config.value_size)
        min_distance_px = max(1.0, values_config.value_min_distance_px)
        max_per_contour = max(0, values_config.value_max_per_contour)
        density = max(0.0, values_config.value_density)
        configured_color = list(values_config.value_color or [])
        if len(configured_color) < 3:
            configured_color = configured_color + [255] * (3 - len(configured_color))
        text_color = tuple(int(np.clip(c, 0, 255)) for c in configured_color[:3])
        text_thickness = max(1, int(round(1.2 * font_scale)))

        used_positions: list[tuple[int, int]] = []
        used_boxes: list[tuple[int, int, int, int]] = []

        for mask, threshold in zip(masks, thresholds):
            binary = mask.img_as(bool).img.astype(np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            text = self._format_threshold(threshold, values_config.value_format)

            for contour in contours:
                selected_positions, selected_boxes = self._select_label_positions(
                    contour=contour,
                    min_distance_px=min_distance_px,
                    max_per_contour=max_per_contour,
                    density=density,
                    existing_positions=used_positions,
                    existing_boxes=used_boxes,
                    text=text,
                    font_scale=font_scale,
                    thickness=text_thickness,
                )
                for pos in selected_positions:
                    cv2.putText(
                        overlay,
                        text,
                        pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        text_color,
                        text_thickness,
                        cv2.LINE_AA,
                    )
                used_positions.extend(selected_positions)
                used_boxes.extend(selected_boxes)

        if alpha < 1.0:
            blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)
        else:
            blended = overlay

        return darsia.full_like(contour_image, blended, dtype=np.uint8)

    def __call__(
        self,
        img,
        saturation_g: darsia.Image | None,
        concentration_aq: darsia.Image | None,
        mass: darsia.Image | None,
        mass_analysis_result=None,
        colorrange_config=None,
    ) -> darsia.Image:
        contour_img = img.copy()
        if mass_analysis_result is None:
            mass_analysis_result = _compose_mass_analysis_result(
                saturation_g, concentration_aq, mass
            )
        for segmentation_config in self.config.values():
            values = resolve_mode_image(
                segmentation_config.mode,
                img,
                mass_analysis_result=mass_analysis_result,
                colorrange_config=colorrange_config,
            )

            # Extract masks based on thresholds
            masks = self.extract_mask(values, segmentation_config.thresholds)

            # Add contours to image
            contour_img = self.add_contours(
                contour_img,
                masks,
                segmentation_config.thresholds,
                segmentation_config.color,
                segmentation_config.alpha,
                segmentation_config.values,
                segmentation_config.linewidth,
            )
        return contour_img
