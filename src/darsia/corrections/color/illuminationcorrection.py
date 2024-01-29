"""Module containing illumination correction functionality."""

from typing import Literal, Union

import numpy as np
import scipy
import scipy.optimize
import skimage

import darsia


class IlluminationCorrection(darsia.BaseCorrection):
    """Class for illumination correction."""

    def __init__(
        self,
        base: Union[darsia.Image, list[darsia.Image]],
        samples: list[tuple[slice, ...]],
        ref_sample: int = -1,
        filter: callable = lambda x: x,
        colorspace: Literal[
            "rgb", "rgb-scalar", "lab", "lab-scalar", "hsl", "hsl-scalar", "gray"
        ] = "hsl-scalar",
        interpolation: Literal["rbf", "quartic", "illumination"] = "quartic",
        show_plot: bool = False,
    ):
        """Initialize an illumination correction.

        Only the L-component is used for RGB-based correction, while the full
        RGB-based correction is used further.

        Args:
            base (darsia.Image): base image
            samples (list[tuple[slice,...]]): list of samples
            ref_sample (int): index of reference sample
            filter (callable): function to preprocess the signal before analysis, e.g.,
                Gaussian filter.
            colorspace (str): colorspace to use for analysis; defaults to "hsl-scalar".
            interpolation (str): interpolation method to use for scaling; defaults to
                "quartic".
            show_plot (bool): flag controlling whether plots of calibration are displayed.

        """
        # Cache input parameters
        if isinstance(base, darsia.Image):
            base = [base]
        num_base = len(base)
        num_samples = len(samples)

        # Convert image to requested format
        self.colorspace = colorspace.lower()
        if self.colorspace in ["rgb", "rgb-scalar"]:
            images = [skimage.img_as_float(base.img) for base in base]
        elif self.colorspace in ["lab"]:
            images = [
                skimage.img_as_float(base.to_trichromatic("LAB", return_image=True).img)
                for base in base
            ]
        elif self.colorspace in ["lab-scalar"]:
            images = [
                skimage.img_as_float(
                    base.to_trichromatic("LAB", return_image=True).img
                )[..., 0]
                for base in base
            ]
        elif self.colorspace in ["hsl"]:
            images = [
                skimage.img_as_float(base.to_trichromatic("HLS", return_image=True).img)
                for base in base
            ]
        elif self.colorspace in ["hsl-scalar"]:
            images = [
                skimage.img_as_float(
                    base.to_trichromatic("HLS", return_image=True).img
                )[..., 1]
                for base in base
            ]
        elif self.colorspace == "gray":
            images = [
                skimage.img_as_float(base.to_monochromatic("gray").img) for base in base
            ]
        else:
            raise ValueError(
                "Invalid method. Choose from 'rgb', 'lab', 'hsl(...-scalar)', 'gray'."
            )

        # Fetch characteristic colors from samples
        characteristic_colors = []
        reference_colors = []
        for image in images:
            colors = darsia.extract_characteristic_data(
                signal=image,
                samples=samples,
                filter=filter,
                show_plot=show_plot,
            )

            # Pick reference color and cache results
            reference_colors.append(np.outer(np.ones(num_samples), colors[ref_sample]))
            characteristic_colors.append(colors)

        # Determine local scaling values
        method_is_trichromatic = self.colorspace in ["rgb", "lab", "hsl"]
        color_components = 3 if method_is_trichromatic else 1

        def objective_function(scaling):
            """Objective function for least-squares problem."""
            scaling = (np.reshape(scaling, (num_samples, color_components)),)
            stacked_scaling = np.vstack(num_base * [scaling])
            stacked_characteristic_colors = np.vstack(characteristic_colors)
            stacked_reference_colors = np.vstack(reference_colors)
            return np.sum(
                (
                    np.multiply(stacked_scaling, stacked_characteristic_colors)
                    - stacked_reference_colors
                )
                ** 2
            )

        # Solve least-squares problem
        opt_result = scipy.optimize.minimize(
            objective_function,
            np.ones(num_samples * color_components),
            method="Powell",
            tol=1e-6,
            options={"maxiter": 1000, "disp": True},
        )

        scaling = np.reshape(opt_result.x, (num_samples, color_components))

        # Interpolate scaling to the full coordinate system
        # Implicitly assume that all base images have the same coordinate system
        x_coords = np.vstack(
            num_base
            * [
                np.array(
                    [
                        base[0].coordinatesystem.coordinate(
                            darsia.make_voxel([sl[0].start, sl[1].start])
                        )[0]
                        for sl in samples
                    ]
                )
            ]
        )
        y_coords = np.vstack(
            num_base
            * [
                np.array(
                    [
                        base[0].coordinatesystem.coordinate(
                            darsia.make_voxel([sl[0].start, sl[1].start])
                        )[1]
                        for sl in samples
                    ]
                )
            ]
        )

        # Interpolate the determined scaling and cache it - only the L-component of the
        # LAB-based analysis for RGB-based correction.
        if self.colorspace == "rgb":
            self.local_scaling = [
                darsia.interpolate_to_image(
                    [x_coords, y_coords, scaling[:, i]], base[0], method=interpolation
                )
                for i in range(3)
            ]
        elif self.colorspace == "lab":
            self.local_scaling = [
                darsia.interpolate_to_image(
                    [x_coords, y_coords, scaling[:, 0]], base[0], method=interpolation
                )
            ]
        elif self.colorspace == "hsl":
            self.local_scaling = [
                darsia.interpolate_to_image(
                    [x_coords, y_coords, scaling[:, 1]], base[0], method=interpolation
                )
            ]
        else:
            assert not method_is_trichromatic
            self.local_scaling = [
                darsia.interpolate_to_image(
                    [x_coords, y_coords, scaling[:, 0]],
                    base[0],
                    method=interpolation,
                )
            ]

    def correct_array(self, img: np.ndarray) -> np.ndarray:
        """Rescale an array using local WB.

        Args:
            img (np.ndarray): input image

        Returns:
            np.ndarray: corrected image

        """
        img_wb = img.copy()
        if img.shape[-1] == 1:
            raise NotImplementedError("Only color images are supported.")
        else:
            assert img.shape[-1] == 3
            for i in range(3):
                # NOTE: Only the "rgb" methodology employs a multi-component scaling.
                img_wb[..., i] = np.multiply(
                    img_wb[..., i],
                    self.local_scaling[i if self.colorspace == "rgb" else 0].img,
                )
        return img_wb
