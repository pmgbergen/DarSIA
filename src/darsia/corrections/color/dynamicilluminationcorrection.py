"""Module containing dynamic illumination correction functionality."""

from pathlib import Path
from typing import Literal

import numpy as np
import scipy
import scipy.optimize
import skimage

import darsia


class DynamicIlluminationCorrection(darsia.BaseCorrection):
    """Class for illumination correction."""

    def setup(
        self,
        base: darsia.Image,
        colorspace: Literal[
            "rgb", "rgb-scalar", "lab", "lab-scalar", "hsl", "hsl-scalar", "gray"
        ] = "hsl-scalar",
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
            rescale (bool): flag controlling whether scaling ensures max value 1

        """
        # Cache input parameters
        self.colorspace = colorspace.lower()

        # Define reference samples in the base image
        assistant = darsia.BoxSelectionAssistant(base)
        self.reference_samples = assistant()

        # Fetch and store reference colors
        self.reference_colors = self.extract_characteristic_colors(base)

        # Determine local scaling values
        method_is_trichromatic = self.colorspace in ["rgb", "lab", "hsl"]
        self.color_components = 3 if method_is_trichromatic else 1

    def extract_characteristic_colors(self, base: darsia.Image) -> np.ndarray:
        # Convert image to requested format
        if self.colorspace in ["rgb", "rgb-scalar"]:
            base_image = skimage.img_as_float(base.img)
        elif self.colorspace in ["lab"]:
            base_image = skimage.img_as_float(
                base.to_trichromatic("LAB", return_image=True).img
            )
        elif self.colorspace in ["lab-scalar"]:
            base_image = skimage.img_as_float(
                base.to_trichromatic("LAB", return_image=True).img
            )[..., 0]
        elif self.colorspace in ["hsl"]:
            base_image = skimage.img_as_float(
                base.to_trichromatic("HLS", return_image=True).img
            )
        elif self.colorspace in ["hsl-scalar"]:
            base_image = skimage.img_as_float(
                base.to_trichromatic("HLS", return_image=True).img
            )[..., 1]
        elif self.colorspace == "gray":
            base_image = skimage.img_as_float(base.to_monochromatic("gray").img)
        else:
            raise ValueError(
                "Invalid method. Choose from 'rgb', 'lab', 'hsl(...-scalar)', 'gray'."
            )

        # Fetch characteristic colors from samples
        colors = darsia.extract_characteristic_data(
            signal=base_image,
            mask=None,
            samples=self.reference_samples,
            filter=lambda x: x,
            show_plot=False,
        )

        return np.array(colors)

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

            colors = self.extract_characteristic_colors(img_wb)

            def objective_function(scaling: float):
                """Objective function for least-squares problem."""
                scaling = (np.reshape(scaling, (1, self.color_components)),)
                stacked_characteristic_colors = np.vstack(self.reference_colors)
                stacked_reference_colors = np.vstack(colors)
                return np.sum(
                    (
                        np.multiply(scaling, stacked_characteristic_colors)
                        - stacked_reference_colors
                    )
                    ** 2
                )

            # Solve least-squares problem
            opt_result = scipy.optimize.minimize(
                objective_function,
                np.ones(self.color_components),
                method="Powell",
                tol=1e-6,
                options={"maxiter": 1000, "disp": True},
            )

            # Shape and rescale
            scaling = np.reshape(opt_result.x, (1, self.color_components))
            print(f"Scaling factors: {scaling}")

            for i in range(3):
                # NOTE: Only the "rgb" methodology employs a multi-component scaling.
                img_wb[..., i] = np.multiply(
                    img_wb[..., i],
                    scaling[i if self.colorspace == "rgb" else 0],
                )
        return img_wb

    def save(self, path: Path) -> None:
        """Save the illumination correction to a file.

        Args:
            path (Path): path to the file

        """
        # Make sure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Store color space and local scaling images as npz files
        np.savez(
            path,
            class_name=type(self).__name__,
            config={
                "colorspace": self.colorspace,
                "samples": self.reference_samples,
                "reference_colors": self.reference_colors,
            },
        )
        print(f"Dynamic illumination correction saved to {path}.")

    def load(self, path: Path) -> None:
        """Load the illumination correction from a file.

        Args:
            path (Path): path to the file

        """
        # Make sure the file exists
        if not path.is_file():
            raise FileNotFoundError(f"File {path} not found.")

        # Load color space and local scaling images from npz file
        data = np.load(path, allow_pickle=True)["config"].item()
        self.colorspace = data["colorspace"]
        self.reference_samples = data["samples"]
        self.reference_colors = data["reference_colors"]
