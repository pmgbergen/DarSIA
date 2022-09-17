"""
Module containing class for translation analysis, relevant e.g. for
studying compaction of porous media.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.interpolate import RBFInterpolator

import daria


class TranslationAnalysis:
    """
    Class for translation analysis.
    """

    def __init__(
        self,
        img_src: daria.Image,
        N_patches: list[int],
        rel_overlap: float,
        translationEstimator: daria.TranslationEstimator,
    ) -> None:
        """
        Constructor for TranslationAnalysis.

        Args:
            img_src (daria.Image): base image
            N_patches (list of two int): number of patches in x and y direction
            rel_overlap (float): relative overal related to patch size in each direction
        """
        # Store parameters
        self.img_src = img_src
        self.N_patches = N_patches
        self.rel_overlap = rel_overlap
        self.translationEstimator = translationEstimator

        # Construct patches of the base image
        self.patches_src = daria.Patches(
            img_src, *self.N_patches, rel_overlap=self.rel_overlap
        )

        # Determine the centers of the patches, in Cartesian coordinates
        self.patch_centers_cartesian = self.patches_src.global_centers_cartesian

    def update_params(
        self, N_patches: Optional[list[int]] = None, rel_overlap: Optional[float] = None
    ) -> None:
        """
        Routine allowing to update parameters for creating patches.

        If any of the parameters is changed, a new patch of the base image is created.

        Args:
            N_patches (list of two int): number of patches in x and y direction
            rel_overlap (float): relative overal related to patch size in each direction
        """
        # Check if any update is needed
        need_update_N_patches = N_patches is not None and N_patches != self.N_patches
        need_update_rel_overlap = (
            rel_overlap is not None and rel_overlap != self.rel_overlap
        )

        # Update parameters if needed
        if need_update_N_patches:
            self.N_patches = N_patches

        if need_update_rel_overlap:
            self.rel_overlap = rel_overlap

        # Create new patches of the base image if any changes performed
        if need_update_N_patches or need_update_rel_overlap:
            self.patches_src = daria.Patches(
                self.img_src, *self.N_patches, rel_overlap=self.rel_overlap
            )

            # Determine the centers of the patches in Cartesian coordinates
            if need_update_N_patches:
                self.patch_centers_cartesian = self.patches_src.global_centers_cartesian

    def find_translation(
        self, img_dst: daria.Image, units: list[str] = ["metric", "pixel"]
    ) -> tuple:
        """
        Find translation map as translation required to match the base image to the
        provided image.

        The final translation map will be stored as callable function. And it allows
        various input and output spaces (metric vs. pixel).

        Args:
            img_dst (daria.Image): test image
            units (list of str): units for input (first entry) and output (second entry)
                of the resulting translation map; accepts either "metric" (default for both)
                or "pixel".

        Returns:
            Callable: translation map defined as interpolator
            bool: flag indicating on which patches the routine has been successful
        """
        # Assert correct units
        assert all([unit in ["metric", "pixel"] for unit in units])

        # Overall strategy:
        # 1. Determine translation on patches.
        # 2. Add potentially known boundary conditions.
        # 3. Create interpolator which is the main result.

        # ! ---- Step 1. Patch analysis.

        # Construct patches of the test image
        patches_dst = daria.Patches(
            img_dst, *self.N_patches, rel_overlap=self.rel_overlap
        )

        # Monitor success of finding a translation/homography for each patch
        have_translation = np.zeros(tuple(self.N_patches), dtype=bool)

        # Initialize containers for coordinates of the patches as well as the translation
        # Later to be used for interpolation.
        input_coordinates: list[np.ndarray] = []
        patch_translation_x: list[float] = []
        patch_translation_y: list[float] = []

        # Continue with investigating all patches. The main idea is to determine
        # a suitable patchwise (discontinuous) homography, and extract an effective
        # translation for each patch. The following procedure does not work
        # for all patches; those unsuccessful ones will be covered by interpolation
        # aferwards.

        # TODO adaptive refinement stategy - as safety measure! will require
        # some effort in how to access patches. try first with fixed params.

        # Loop over all patches.
        for i in range(self.N_patches[0]):
            for j in range(self.N_patches[1]):

                # Fetch patches of both source and destination image
                img_src = self.patches_src(i, j)
                img_dst = patches_dst(i, j)

                # Determine effective translation, operating on pixel coordinates using
                # reverse matrix indexing.
                (
                    translation,
                    intact_translation,
                ) = self.translationEstimator.find_effective_translation(
                    img_src.img, img_dst.img, None, None, plot_matches=False
                )

                # The above procedure to find a matching transformation is successful if in
                # any of the iterations, a translation has been found. If so, postprocess
                # and store the effective translation.
                if intact_translation:

                    # Flag success
                    have_translation[i, j] = True

                    # Fetch the center of the patch in metric units, which will be the input
                    # for later construction of the interpolator
                    center = self.patch_centers_cartesian[i, j]

                    # Convert to pixel units using reverse matrix indexing, if required
                    if units[0] == "pixel":
                        center = self.img_src.coordinatesystem.coordinateToPixel(
                            center, reverse=True
                        )

                    # Extract the effective displacement, stored in the constant part of
                    # the affine map, in pixel units, using reverse matrix indexing.
                    displacement = translation[:, -1]

                    # Convert to metric units if required - note that displacement is a
                    # vector, and that it is given using reverse matrix indexing
                    if units[1] == "metric":
                        displacement = self.img_src.coordinatesystem.pixelToCoordinateVector(
                            displacement, reverse=True
                        )

                    # Store the displacement for the centers of the patch.
                    # NOTE: In any case, the first and second components
                    # correspond to the x and y component.
                    input_coordinates.append(center)
                    patch_translation_x.append(displacement[0])
                    patch_translation_y.append(displacement[1])

        # ! ---- Step 2. Boundary conditions.

        # Fetch predetermined conditions (do not have to be on the boundary)
        extra_coordinates_x, extra_translation_x = self.bc_x(units)
        extra_coordinates_y, extra_translation_y = self.bc_y(units)

        # ! ---- Step 3. Interpolation.

        # Finally define separate interpolators for the translation in x and y
        # directions.
        self.interpolator_translation_x = RBFInterpolator(
            input_coordinates + extra_coordinates_x,
            patch_translation_x + extra_translation_x,
        )
        self.interpolator_translation_y = RBFInterpolator(
            input_coordinates + extra_coordinates_y,
            patch_translation_y + extra_translation_y,
        )

        # Convert interpolators to a callable displacement/translation map
        def translation(arg):
            return np.array(
                [
                    self.interpolator_translation_x(arg),
                    self.interpolator_translation_y(arg),
                ]
            )

        self.translation = translation

        # Store success
        self.have_translation = have_translation.copy()

        return self.translation, self.have_translation.any()

    def bc_x(self, units: list[str]) -> tuple:
        """
        Prescribed (boundary) conditions for the displacement in x direction.

        Can be overwritten. Here, tailored to FluidFlower scenarios, fix
        the displacement in x-direction at the vertical boundaries of the
        image.

        Args:
            units (list of str): "metric" or "pixel"

        Returns:
            list of np.ndarray: coordinates
            list of float: translation in x direction
        """
        # The loop over the boundary will depend on whether the coordinates
        # are interpreted as pixels or in metric units. Define the respective
        # sets here, and combine left and right vertical boundaries for simplicity,
        # as they will get assigned the same translation.
        vertical_boundary: list[np.ndarray] = []
        if units[0] == "metric":
            # Add the left vertical boundary
            vertical_boundary += [
                self.img_src.origo + np.array([0, y_pos])
                for y_pos in np.linspace(0, self.img_src.height, self.N_patches[1] + 1)
            ]
            # Add the right vertical boundary
            vertical_boundary += [
                self.img_src.origo + np.array([self.img_src.width, y_pos])
                for y_pos in np.linspace(0, self.img_src.height, self.N_patches[1] + 1)
            ]

        elif units[0] == "pixel":
            # Add the left vertical boundary - comply to reverse matrix indexing
            vertical_boundary += [
                np.array([0, y_pos])
                for y_pos in np.linspace(
                    0, self.img_src.num_pixels_height, self.N_patches[1] + 1
                )
            ]
            # Add the right vertical boundary - comply to reverse matrix indexing
            vertical_boundary += [
                np.array([self.img_src.num_pixels_width, y_pos])
                for y_pos in np.linspace(
                    0, self.img_src.num_pixels_height, self.N_patches[1] + 1
                )
            ]

        return vertical_boundary, len(vertical_boundary) * [0.0]

    def bc_y(self, units: list[str]) -> tuple:
        """
        Prescribed (boundary) conditions for the displacement in y direction.

        Args:
            units (list of str): "metric" or "pixel"

        Can be overwritten. Here, tailored to FluidFlower scenarios, fix
        the displacement in y-direction at the horizontal boundaries of the
        image.

        Returns:
            list of np.ndarray: coordinates
            list of float: translation in y direction
        """

        # The loop over the boundary will depend on whether the coordinates
        # are interpreted as pixels or in metric units. Define the respective
        # sets here, and combine left and right vertical boundaries for simplicity,
        # as they will get assigned the same translation.
        horizontal_boundary: list[np.ndarray] = []
        if units[0] == "metric":
            # Add the bottom horizontal boundary
            horizontal_boundary += [
                self.img_src.origo + np.array([x_pos, 0])
                for x_pos in np.linspace(0, self.img_src.width, self.N_patches[0] + 1)
            ]

        elif units[0] == "pixel":
            # Add the bottom horizontal boundary - comply to reverse matrix indexing
            horizontal_boundary += [
                np.array([x_pos, self.img_src.num_pixels_height])
                for x_pos in np.linspace(
                    0, self.img_src.num_pixels_width, self.N_patches[0] + 1
                )
            ]

        return horizontal_boundary, len(horizontal_boundary) * [0.0]

    def translate_centers(
        self, img_dst: daria.Image, plot_translation: bool = False
    ) -> None:
        """
        Translate centers of the base image.

        Args:
            plot_translation (bool): flag controlling whether the translation is
                plotted in terms of arrows on top of the base and test images
        """
        # Only continue if a translation has been already found
        assert self.have_translation.any()

        # Interpolate at patch centers
        input_arg = np.transpose(
            np.vstack(
                (
                    np.flipud(self.patch_centers_cartesian[:, :, 0].T).flatten(),
                    np.flipud(self.patch_centers_cartesian[:, :, 1].T).flatten(),
                )
            )
        )
        interpolated_patch_translation_x = self.interpolator_translation_x(input_arg)
        interpolated_patch_translation_y = self.interpolator_translation_y(input_arg)

        # Convert coordinates of patch centers to pixels - using the matrix indexing
        patch_centers_x_pixels = np.zeros(tuple(reversed(self.N_patches)), dtype=int)
        patch_centers_y_pixels = np.zeros(tuple(reversed(self.N_patches)), dtype=int)
        for j in range(self.N_patches[1]):
            for i in range(self.N_patches[0]):
                center = self.patch_centers_cartesian[i, j]
                pixel = self.img_src.coordinatesystem.coordinateToPixel(center)
                patch_centers_x_pixels[self.N_patches[1] - 1 - j, i] = pixel[1]
                patch_centers_y_pixels[self.N_patches[1] - 1 - j, i] = pixel[0]

        # Plot the interpolated translation
        if plot_translation:
            fig, ax = plt.subplots(1, num=1)
            ax.quiver(
                patch_centers_x_pixels,
                patch_centers_y_pixels,
                interpolated_patch_translation_x,
                interpolated_patch_translation_y,
                scale=2000,
                color="white",
            )
            ax.imshow(
                skimage.util.compare_images(
                    self.img_src.img, img_dst.img, method="blend"
                )
            )
            fig, ax = plt.subplots(1, num=2)
            ax.imshow(
                skimage.util.compare_images(
                    self.img_src.img,
                    skimage.transform.resize(
                        np.flipud(self.have_translation.T), self.img_src.img.shape
                    ),
                    method="blend",
                )
            )
            plt.show()

    def translate_image(
        self,
        img_src: daria.Image,
        reverse: bool = False,
        plot_transformation: bool = False,
    ) -> daria.Image:
        """
        Apply translation to an entire image by using piecwise perspective transformation.

        Args:
            img_dst (daria.Image): source image to be translated (not necessarily self.img_src)
            method (str): either "forward" or "reverse" for applying positive or negative
                displacement
            plot_transformation (bool): plot the transformed image

        Returns:
            daria.Image: translated image
        """

        # Construct patches of the base image without overlap
        patches_src = daria.Patches(img_src, *self.N_patches, rel_overlap=0.0)

        # Create piecewise perspective transform on the patches
        perspectiveTransform = daria.PiecewisePerspectiveTransform()
        transformed_img: daria.Image = perspectiveTransform.find_and_warp(
            patches_src, self.translation, reverse
        )

        return transformed_img
