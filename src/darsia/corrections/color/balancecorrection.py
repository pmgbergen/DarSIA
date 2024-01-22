"""Module containing (white) balance correction functionality."""

import numpy as np
import skimage

import darsia


class BalanceCorrection(darsia.BaseCorrection):
    """Class for balance correction."""

    def __init__(
        self,
        base: darsia.Image,
        samples: list[tuple[slice, ...]],
        ref_sample: int = -1,
        show_plot: bool = False,
    ):
        """Initialize a balance correction.

        Args:
            base (darsia.Image): base image
            samples (list[tuple[slice,...]]): list of samples
            ref_sample (int): index of reference sample
            show_plot (bool): flag controlling whether plots of calibration are displayed.

        """
        self.samples = samples
        self.ref_sample = ref_sample

        # Fetch characteristic colors from samples
        characteristic_colors = darsia.extract_characteristic_data(
            signal=skimage.img_as_float(base.img),
            samples=samples,
            show_plot=show_plot,
        )

        # Pick reference color (to be the one in the center (assuming good lighting here)
        reference_color = characteristic_colors[ref_sample]

        # Scalar WB through solving local least-squares problems
        scaling = np.divide(
            np.sum(
                np.multiply(
                    np.outer(np.ones(len(characteristic_colors)), reference_color),
                    characteristic_colors,
                ),
                axis=1,
            ),
            np.sum(np.multiply(characteristic_colors, characteristic_colors), axis=1),
        )

        # Interpolate scaling to the full coordinate system
        x_coords = np.array(
            [
                base.coordinatesystem.coordinate(
                    darsia.make_voxel([sl[0].start, sl[1].start])
                )[0]
                for sl in samples
            ]
        )
        y_coords = np.array(
            [
                base.coordinatesystem.coordinate(
                    darsia.make_voxel([sl[0].start, sl[1].start])
                )[1]
                for sl in samples
            ]
        )
        self.local_wb_image = darsia.interpolate_to_image(
            [x_coords, y_coords, scaling], base
        )

    def correct_array(self, img: np.ndarray) -> np.ndarray:
        """Rescale an array using local WB.

        Args:
            img (np.ndarray): input image

        Returns:
            np.ndarray: corrected image

        """
        assert (
            img.shape[0:2] == self.local_wb_image.img.shape[0:2] and len(img.shape) == 3
        )
        img_wb = img.copy()
        for i in range(3):
            img_wb[..., i] = np.multiply(img_wb[..., i], self.local_wb_image.img)
        return img_wb
