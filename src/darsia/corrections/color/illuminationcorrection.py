"""Module containing illumination correction functionality."""

from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize
import skimage

import darsia


class IlluminationCorrection(darsia.BaseCorrection):
    """Class for illumination correction."""

    def setup(
        self,
        base: Union[darsia.Image, list[darsia.Image]],
        samples: list[tuple[slice, ...]],
        mask: Optional[np.ndarray] = None,
        ref_sample: int = -1,
        filter: callable = lambda x: x,
        colorspace: Literal[
            "rgb", "rgb-scalar", "lab", "lab-scalar", "hsl", "hsl-scalar", "gray"
        ] = "hsl-scalar",
        interpolation: Literal["rbf", "quartic", "illumination"] = "quartic",
        show_plot: bool = False,
        rescale: bool = False,
        log: Optional[Path] = None,
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
                mask=mask,
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

        # Shape and rescale
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

        if rescale:
            max_scaling = max(
                [
                    np.max(self.local_scaling[i].img)
                    for i in range(len(self.local_scaling))
                ]
            )
            for i in range(len(self.local_scaling)):
                self.local_scaling[i].img /= max_scaling

        if show_plot:
            # Plot the determined scaling
            fig, ax = plt.subplots()
            ax.imshow(self.local_scaling[0].img)
            ax.set_title("Scaling")
            # Add color bar
            fig.colorbar(
                ax.imshow(self.local_scaling[0].img),
                ax=ax,
                orientation="vertical",
                fraction=0.05,
            )
            plt.show()

        # Logging
        if log:

            # Create a directory for the log
            (log / "illumination_correction").mkdir(parents=True, exist_ok=True)

            # Log the original image, with patches and the determined scaling
            plt.figure("Log samples")
            # Plot the base image
            plt.imshow(base[0].img)
            # Overlay with ~mask
            if mask is not None:
                plt.imshow(mask, alpha=0.5)
            # Plot the patches as red boxes, and fill with the determined scaling
            for sample in samples:
                plt.plot(
                    [
                        sample[1].start,
                        sample[1].start,
                        sample[1].stop,
                        sample[1].stop,
                        sample[1].start,
                    ],
                    [
                        sample[0].start,
                        sample[0].stop,
                        sample[0].stop,
                        sample[0].start,
                        sample[0].start,
                    ],
                    "r-",
                )
                plt.fill_between(
                    [sample[1].start, sample[1].stop],
                    sample[0].start,
                    sample[0].stop,
                    color="r",
                    alpha=0.2,
                )
                plt.text(
                    0.5 * (sample[1].start + sample[1].stop),
                    0.5 * (sample[0].start + sample[0].stop),
                    f"{scaling[samples.index(sample)]}",
                    color="w",
                    ha="center",
                    va="center",
                    fontsize=5,
                )
            plt.savefig(log / "illumination_correction" / "samples.png")
            plt.close()

            # Log the before and after scaling in a side-by-side plot
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(base[0].img)
            ax[0].set_title("Original")
            ax[1].imshow(self.correct_array(base[0].img))
            ax[1].set_title("Corrected")
            ax[2].imshow(self.local_scaling[0].img, vmin=0, vmax=2)
            ax[2].set_title("Scaling")
            fig.colorbar(
                ax[2].imshow(self.local_scaling[0].img),
                ax=ax[2],
                orientation="vertical",
                fraction=0.05,
            )
            plt.savefig(log / "illumination_correction" / "scaling.png")
            plt.close()

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
                "local_scaling": self.local_scaling,
            },
        )
        print(f"Illumination correction saved to {path}.")

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
        if "colorspace" not in data or "local_scaling" not in data:
            raise ValueError("Invalid file format.")
        self.colorspace = data["colorspace"]
        self.local_scaling = data["local_scaling"]
