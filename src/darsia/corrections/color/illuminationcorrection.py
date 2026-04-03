"""Module containing illumination correction functionality."""

import logging
import time
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize
import skimage

import darsia
from darsia.presets.workflows.config.corrections import IlluminationCorrectionConfig

logger = logging.getLogger(__name__)


class IlluminationCorrection(darsia.BaseCorrection):
    """Class for illumination correction."""

    def select_random_samples(
        self,
        mask: darsia.Image | np.ndarray,
        config: IlluminationCorrectionConfig | None = None,
    ) -> list[tuple[slice, ...]]:
        # Fix random seed for reproducibility (TODO: make this configurable)
        np.random.seed(config.seed)

        # Find random patches, restricted to the masked regions
        width = config.width
        num_samples = config.num_samples

        larger_mask = np.zeros(
            (mask.shape[0] + width, mask.shape[1] + width), dtype=bool
        )
        larger_mask[: mask.shape[0], : mask.shape[1]] = mask

        indices = np.nonzero(mask if isinstance(mask, np.ndarray) else mask.img)
        moved_indices = tuple([indices[i] + width for i in range(len(indices))])
        test_moved_indices = larger_mask[moved_indices]
        restricted_indices = tuple(
            [indices[i][test_moved_indices] for i in range(len(indices))]
        )

        num_eligible_points = len(restricted_indices[0])
        if num_eligible_points == 0:
            logger.warning(
                """No eligible points for sampling found. Consider reducing the """
                """sample width, or increasing the masked area."""
            )
            return []
        random_ids = np.unique(
            (np.random.rand(num_samples) * num_eligible_points).astype(int)
        )
        sample_indices = np.transpose(
            tuple([restricted_indices[i][random_ids] for i in range(len(indices))])
        )

        samples = [
            (
                slice(sample[0], sample[0] + width, None),
                slice(sample[1], sample[1] + width, None),
            )
            for sample in sample_indices
        ]

        return samples

    def setup(
        self,
        base: darsia.Image | list[darsia.Image],
        sample_groups: list[list[tuple[slice, ...]]],
        mask: darsia.Image | np.ndarray | None = None,
        outliers: float = 0.0,
        filter: callable = lambda x: x,
        colorspace: Literal[
            "rgb", "rgb-scalar", "lab", "lab-scalar", "hsl", "hsl-scalar", "gray"
        ] = "hsl-scalar",
        interpolation: Literal["rbf", "quartic", "illumination"] = "quartic",
        bounds: tuple[float, float] = (0.5, 2.0),
        show_plot: bool = False,
        log: Path | None = None,
    ):
        """Initialize an illumination correction.

        Args:
            base: Image or list of images to use for correction.
            sample_groups: List of groups of samples, where each group is a list of slices
                defining the sample regions.
            mask: Mask to restrict sampling to certain areas (optional).
            outliers: Fraction of outliers to remove from samples (default: 0.0).
            filter: Callable to apply to the sampled colors (default: identity).
            colorspace: Colorspace to use for sampling and correction (default: "hsl-scalar").
            interpolation: Interpolation method for correction (default: "quartic").
            bounds: Bounds for the illumination correction factors (default: (0.5, 2.0)).
            show_plot: Whether to show diagnostic plots during setup (default: False).
            log: Path to save logs or diagnostic plots (optional).

        """
        # Cache input parameters
        if isinstance(base, darsia.Image):
            base = [base]

        # Convert image to requested format
        self.colorspace = colorspace.lower()
        images = self._convert_images(base)

        # Fetch characteristic colors from samples
        characteristic_colors = {}
        # reference_colors = []
        for group_id, samples in enumerate(sample_groups):
            for image_id, image in enumerate(images):
                colors = darsia.extract_characteristic_data(
                    signal=image,
                    mask=mask,
                    samples=samples,
                    filter=filter,
                    show_plot=show_plot,
                )
                characteristic_colors[(group_id, image_id)] = colors

        # Find skipped groups
        skipped_groups = []
        for group_id, samples in enumerate(sample_groups):
            if (
                sum(
                    len(characteristic_colors[(group_id, image_id)])
                    for image_id in range(len(images))
                )
                == 0
            ):
                skipped_groups.append(group_id)

        # Statistics.
        num_groups = len(sample_groups) - len(skipped_groups)
        num_samples = [
            len(s) for g, s in enumerate(sample_groups) if g not in skipped_groups
        ]

        # Determine local scaling values
        method_is_trichromatic = self.colorspace in ["rgb", "lab", "hsl"]
        color_components = 3 if method_is_trichromatic else 1

        # Coarse coordinates.
        coarse_base = darsia.resize(
            base[0], fx=0.1, fy=0.1, interpolation="inter_nearest"
        )

        # Mid voxels, coordinatesm and coarse voxels of samples.
        mid_voxels = []
        mid_coordinates = []
        mid_coarse_voxels = []
        for group_id, samples in enumerate(sample_groups):
            if group_id in skipped_groups:
                continue
            for sample in samples:
                mid_voxel = darsia.make_voxel(
                    [
                        (sample[0].start + sample[0].stop) // 2,
                        (sample[1].start + sample[1].stop) // 2,
                    ]
                )
                mid_voxels.append(mid_voxel)
                mid_coordinates.append(base[0].coordinatesystem.coordinate(mid_voxel))
                mid_coarse_voxels.append(
                    coarse_base.coordinatesystem.voxel(mid_coordinates[-1])
                )

        # Convert to arrays for later processing.
        mid_voxels = darsia.VoxelArray(mid_voxels)
        mid_coordinates = darsia.CoordinateArray(mid_coordinates)
        mid_coarse_voxels = darsia.VoxelArray(mid_coarse_voxels)

        # Cache coordinates for interpolation.
        x_coords = mid_coordinates[:, 0]
        y_coords = mid_coordinates[:, 1]

        def _interpolate_scaling(
            scaling_values: np.ndarray, base_image
        ) -> list[darsia.Image]:
            # Interpolate the determined scaling and cache it - only the L-component of the
            # LAB-based analysis for RGB-based correction.
            if self.colorspace == "rgb":
                local_scaling = [
                    darsia.interpolate_to_image(
                        [x_coords, y_coords, scaling_values[:, i]],
                        base_image,
                        method=interpolation,
                    )
                    for i in range(3)
                ]
            elif self.colorspace == "lab":
                local_scaling = [
                    darsia.interpolate_to_image(
                        [x_coords, y_coords, scaling_values[:, 0]],
                        base_image,
                        method=interpolation,
                    )
                ]
            elif self.colorspace == "hsl":
                local_scaling = [
                    darsia.interpolate_to_image(
                        [x_coords, y_coords, scaling_values[:, 1]],
                        base_image,
                        method=interpolation,
                    )
                ]
            else:
                assert not method_is_trichromatic
                local_scaling = [
                    darsia.interpolate_to_image(
                        [x_coords, y_coords, scaling_values[:, 0]],
                        base_image,
                        method=interpolation,
                    )
                ]
            return local_scaling

        def objective_function(scaling):
            """Objective function for least-squares problem."""

            # Preparation.
            reshaped_scaling = np.reshape(scaling, (-1, color_components))
            local_scaling = _interpolate_scaling(reshaped_scaling, coarse_base)

            # Fetch the effective scaling associated to the sampling points.
            effective_scaling = np.array(
                [
                    local_scaling[color_id].eval(mid_coordinates)
                    for color_id in range(color_components)
                ]
            ).reshape(-1, color_components)

            # Initialize residual.
            residual = 0

            # Part 1. Relative residual (rescaled colors=avg within each group).
            for group_id in range(num_groups):
                # Fetch scaling for the group and reshape.
                assert color_components == 1, (
                    """Only scalar methods are currently supported for the """
                    """optimization-based approach."""
                )
                group_scaling = effective_scaling[
                    sum(num_samples[:group_id]) : sum(num_samples[: group_id + 1])
                ]
                for image_id in range(len(images)):
                    # Fetch reference colors for the group and image.
                    group_colors = characteristic_colors[(group_id, image_id)]
                    if len(group_colors) == 0:
                        continue
                    stacked_characteristic_colors = np.vstack(group_colors)

                    # Pre compute the scaled colors.
                    assert stacked_characteristic_colors.shape == group_scaling.shape
                    rescaled_group_colors = np.multiply(
                        group_scaling, stacked_characteristic_colors
                    )

                    # Determine the average rescaled color and stack it to the reference color.
                    avg_color = np.mean(rescaled_group_colors, axis=0)
                    stacked_avg_color = np.outer(
                        np.ones(num_samples[group_id]), avg_color
                    )

                    # Operate on actual scaling
                    true_group_scaling = reshaped_scaling[
                        sum(num_samples[:group_id]) : sum(num_samples[: group_id + 1])
                    ]
                    true_rescaled_group_colors = np.multiply(
                        true_group_scaling, stacked_characteristic_colors
                    )
                    local_residuals = (
                        true_rescaled_group_colors - stacked_avg_color
                    ) ** 2

                    # Trim away smallest and largest fraction defined by `outliers`
                    sorted_residuals = np.sort(local_residuals, axis=0)
                    trim_amount = int(outliers * sorted_residuals.shape[0])
                    if trim_amount == 0:
                        # No trimming requested: use all residuals
                        residual += np.sum(sorted_residuals)
                    else:
                        residual += np.sum(
                            sorted_residuals[trim_amount:-trim_amount]
                        )

            return residual

        # Solve least-squares problem to find the optimal scaling values.
        num_vars = sum(num_samples) * color_components
        vector_bounds = [bounds] * num_vars

        logger.info("Starting optimization for illumination correction...")
        tic = time.time()
        opt_result = scipy.optimize.minimize(
            objective_function,
            np.ones(num_vars),
            bounds=vector_bounds,
            # method="Powell", # slower for large number of samples
            method="L-BFGS-B",
            tol=1e-6,
            options={"maxiter": 1000, "disp": True},
        )
        toc = time.time()
        logger.info(
            f"Finished optimization for illumination correction in {toc - tic} seconds."
        )
        reshaped_scaling = np.reshape(opt_result.x, (-1, color_components))
        self.local_scaling = _interpolate_scaling(reshaped_scaling, base[0])

        # Evaluate deviation for all samples (skip empty/skipped groups).
        for group_id, samples in enumerate(sample_groups):
            if group_id in skipped_groups:
                continue
            for image_id in range(len(images)):
                # Fetch the reference color for the sample.
                colors = characteristic_colors[(group_id, image_id)]
                if len(colors) == 0:
                    continue
                local_scaling = np.array(
                    [
                        self.local_scaling[i].eval(mid_coordinates)
                        for i in range(color_components)
                    ]
                )
                corrected_colors = np.multiply(
                    local_scaling.reshape(colors.shape), colors
                )

                # Log the deviation.
                original_deviation = np.linalg.norm(
                    colors - np.mean(colors, axis=0, keepdims=True)
                )
                corrected_deviation = np.linalg.norm(
                    np.array(corrected_colors)
                    - np.mean(corrected_colors, axis=0, keepdims=True)
                )
                logger.info(
                    f"""Deviation for group {group_id}, image {image_id}: """
                    f"""{original_deviation} -> {corrected_deviation}"""
                )

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
        if show_plot:
            plt.show()
        else:
            plt.close()

        # Log the original image, with patches and the determined scaling
        plt.figure("Log samples")
        # Plot the base image
        plt.imshow(base[0].img)
        # Overlay with ~mask
        if mask is not None:
            plt.imshow(mask if isinstance(mask, np.ndarray) else mask.img, alpha=0.5)
            plt.axis("off")

        # Plot the patches as red boxes, and fill with the determined scaling
        for samples in sample_groups:
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
                    (sample[1].start + sample[1].stop) // 2,
                    (sample[0].start + sample[0].stop) // 2,
                    f"{reshaped_scaling[samples.index(sample)]}",
                    color="w",
                    ha="center",
                    va="center",
                    fontsize=5,
                )

        # Logging
        if log:
            # Create a directory for the log
            (log / "illumination_correction").mkdir(parents=True, exist_ok=True)

            plt.savefig(log / "illumination_correction" / "samples.png", dpi=500)
        if show_plot:
            plt.show()
        else:
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
        if log:
            plt.savefig(log / "illumination_correction" / "scaling.png", dpi=500)
        if show_plot:
            plt.show()
        else:
            plt.close()

    def _convert_images(self, base_images: list[darsia.Image]) -> list[np.ndarray]:
        """Convert the base images to the requested colorspace and format.

        Conversion is based on attributes:
            - colorspace: the target colorspace for the conversion

        Args:
            base_images (list[darsia.Image]): list of base images

        Returns:
            list[np.ndarray]: list of converted images

        """
        if self.colorspace in ["rgb", "rgb-scalar"]:
            images = [skimage.img_as_float(base.img) for base in base_images]
        elif self.colorspace in ["lab"]:
            images = [
                skimage.img_as_float(base.to_trichromatic("LAB", return_image=True).img)
                for base in base_images
            ]
        elif self.colorspace in ["lab-scalar"]:
            images = [
                skimage.img_as_float(
                    base.to_trichromatic("LAB", return_image=True).img
                )[..., 0]
                for base in base_images
            ]
        elif self.colorspace in ["hsl"]:
            images = [
                skimage.img_as_float(base.to_trichromatic("HLS", return_image=True).img)
                for base in base_images
            ]
        elif self.colorspace in ["hsl-scalar"]:
            images = [
                skimage.img_as_float(
                    base.to_trichromatic("HLS", return_image=True).img
                )[..., 1]
                for base in base_images
            ]
        elif self.colorspace == "gray":
            images = [
                skimage.img_as_float(base.to_monochromatic("gray").img)
                for base in base_images
            ]
        else:
            raise ValueError(
                "Invalid method. Choose from 'rgb', 'lab', 'hsl(...-scalar)', 'gray'."
            )
        return images

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
        elif hasattr(self, "local_scaling"):
            assert img.shape[-1] == 3
            for i in range(3):
                # NOTE: Only the "rgb" methodology employs a multi-component scaling.
                img_wb[..., i] = np.multiply(
                    img_wb[..., i],
                    self.local_scaling[i if self.colorspace == "rgb" else 0].img,
                )
        else:
            logger.info("No local scaling determined. Returning original image.")
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
