"""
Module containing class for compaction analysis.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.interpolate import RBFInterpolator

import daria


class CompactionAnalysis:
    """
    Class for compaction analysis.
    """

    def __init__(
        self,
        img_src: daria.Image,
        N_patches: list[int],
        rel_overlap: float,
        translationEstimator: daria.TranslationEstimator,
    ) -> None:
        """
        Constructor for CompactionAnalysis.

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

        # Determine the centers of the patches
        self.patch_centers = self.patches_src.centers

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

            # Determine the centers of the patches
            if need_update_N_patches:
                self.patch_centers = self.patches_src.centers

    def find_compaction_map(self, img_dst: daria.Image, units: list[str]=["metric", "metric"]) -> tuple:
        """
        Find compaction map in order to match the base image to the provided image.

        The compaction will be measure in metric units.

        Args:
            img_dst (daria.Image): test image
            units (list of str): units for input (first entry) and output (second entry)
                of the resulting compaction map; accepts either "metric" (default for both)
                or "pixel".

        Returns:
            RBFInterpolator: compaction map as interpolator in space
            np.ndarray: flag indicating on which patches the routine has been successful
        """
        # Assert correct units
        assert all([unit in ["metric", "pixel"] for unit in units])

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

                # Determine effective translation measured in number of pixels
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

                    # Fetch the center of the patch in metric units, which will be the input for
                    # later construction of the interpolator
                    center = self.patch_centers[i,j]

                    # Convert to pixel units if required
                    if units[0] == "pixel":
                        # TODO
                        center = center

                    # Extract the effective displacement, stored in the constant part of
                    # the affine map, in pixel units
                    displacement = translation[:, -1]

                    # Convert to metric units if required
                    if units[1] == "metric":
                        # TODO
                        # Also take into account the flip, i.e., flip the direction with minus sign
                        displacement = displacement

                    print("test")
                    print(displacement)
                    print(self.patches_src.baseImg.coordinatesystem.pixelsToLength(displacement, "xy"))

                    # Store the displacement for the centers of the patch.
                    input_coordinates.append(center)
                    patch_translation_x.append(displacement[0])
                    patch_translation_y.append(displacement[1])

                    ## For debugging purposes only
                    # if False:

                    #    # Apply translation
                    #    (h, w) = img_dst.shape[:2]
                    #    aligned_img_src = cv2.warpAffine(img_src.img, translation, (w, h))

                    #    # Plot the original two images and the corrected ones
                    #    fig, ax = plt.subplots(2, 1)
                    #    ax[0].imshow(
                    #        skimage.util.compare_images(
                    #            img_src.img, img_dst.img, method="blend"
                    #        )
                    #    )
                    #    ax[1].imshow(
                    #        skimage.util.compare_images(
                    #            aligned_img_src, img_dst.img, method="blend"
                    #        )
                    #    )
                    #    plt.show()

        # Goal: Interpolate the effective translation in all patches

        # Supplement computed data with some boundary condition data.
        # For instance at the left and right boundary, there is no displacement
        # in x direction, while there is none in y-direction at the bottom.
        # Since this will result in two different data sets (for the coordinates)
        # a copy has to be made.
        input_coordinates_translation_x = input_coordinates.copy()
        input_coordinates_translation_y = input_coordinates.copy()

        # In order to still be able to plot the computed values without
        # the boundary data, also the translation data has to be copied.
        # FIXME: Copy needed? Only for plotting of displacement arrows.
        patch_translation_x_with_bc = patch_translation_x.copy()
        patch_translation_y_with_bc = patch_translation_y.copy()

        # TODO adapt according to unit[0] how to loop through the sets

        # Begin with known translation at the left and right boundary
        for y_loc in np.linspace(0, img_src.height, self.N_patches[1] + 1):
            # Left boundary patches
            input_coordinates_translation_x.append(img_src.origo + np.array([0, y_loc]))
            patch_translation_x_with_bc.append(0.0)

            # Right boundary patches
            input_coordinates_translation_x.append(img_src.origo + np.array([img_src.width, y_loc]))
            patch_translation_x_with_bc.append(0.0)

        # Also add known translation at the top and bottom
        for x_loc in np.linspace(0, img_src.width, self.N_patches[0] + 1):
            # Bottom boundary patches
            input_coordinates_translation_y.append(img_src.origo + np.array([x_loc, 0]))
            patch_translation_y_with_bc.append(0.0)

        # Finally define separate interpolators for the translation in x and y
        # directions.
        self.interpolator_translation_x = RBFInterpolator(
            np.array(input_coordinates_translation_x),
            patch_translation_x_with_bc,
        )
        self.interpolator_translation_y = RBFInterpolator(
            np.array(input_coordinates_translation_y),
            patch_translation_y_with_bc,
        )

        return (
            self.interpolator_translation_x,
            self.interpolator_translation_y,
            have_translation,
        )

    def find_translation(self, img_dst: daria.Image) -> None:
        """
        Compute compaction map as interpolator and apply to patch centers.

        Args:
            img_dst (daria.Image): test image

        Returns:
            np.ndarray: translation in x direction
            np.ndarray: translation in y direction
        """
        # Determine interpolators for both component of the translation
        (
            self.interpolator_translation_x,
            self.interpolator_translation_y,
            self.have_translation,
        ) = self.find_compaction_map(img_dst)

        def displacement(arg):
            return np.array([self.interpolator_translation_x(arg), self.interpolator_translation_y(arg)])

        self.displacement = displacement

    def translate_centers(self, img_dst: daria.Image, plot_translation: bool = False) -> None:
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
                    np.flipud(self.patch_centers[:, :, 0].T).flatten(),
                    np.flipud(self.patch_centers[:, :, 1].T).flatten(),
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
                center = self.patch_centers[i, j]
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
