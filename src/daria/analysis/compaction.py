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

    def find_compaction_map(self, img_dst: daria.Image):
        """
        Find compaction map in order to match the base image to the provided image.

        Args:
            img_dst (daria.Image): test image

        Returns:
            RBFInterpolator: compaction map as interpolator in space
            np.ndarray: flag indicating on which patches the routine has been successful
        """

        # Construct patches of the test image
        patches_dst = daria.Patches(
            img_dst, *self.N_patches, rel_overlap=self.rel_overlap
        )

        # Monitor success of finding a translation/homography for each patch
        have_translation = np.zeros(tuple(self.N_patches), dtype=bool)

        # Initialize containers for coordinates of the patches as well as the translation
        # Later to be used for interpolation.
        x_patch: list[float] = []
        y_patch: list[float] = []
        patch_translation_x: list[float] = []
        patch_translation_y: list[float] = []

        # Continue with investigating all patches. The main idea is to determine
        # a suitable patchwise (discontinuous) homography, and extract an effective
        # translation for each patch. The following procedure does not work
        # for all patches; those unsuccessful ones will be covered by interpolation
        # aferwards.

        # Loop over all patches.
        for i in range(self.N_patches[0]):
            for j in range(self.N_patches[1]):

                # TODO adaptive refinement stategy - as safety measure! will require
                # some effort in how to access patches. try first with fixed params.

                # TODO just probe in many patches, track success and apply interpolation
                # in the end, under the assumption of a continuous process - should be
                # true in the FluidFlower without fault evolution

                # Fetch patches of both source and desctination image
                img_src = self.patches_src(i, j)
                img_dst = patches_dst(i, j)

                # Determine effective translation
                (
                    translation,
                    intact_translation,
                ) = self.translationEstimator.find_effective_translation(
                    img_src.img, img_dst.img, None, None, plot_matches=False
                )

                # The above procedure to find a matching transformation is successful if in
                # any of the iterations, a translation has been found.
                if intact_translation:

                    # Flag success
                    have_translation[i, j] = True

                    # Extract the effective displacement, stored in the constant part of
                    # the affine map.
                    displacement = translation[:, -1]

                    # Store the displacement for the centers of the patch
                    x_patch.append(self.patch_centers[i, j, 0])
                    y_patch.append(self.patch_centers[i, j, 1])
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
        x_patch_for_translation_x = x_patch.copy()
        x_patch_for_translation_y = x_patch.copy()
        y_patch_for_translation_x = y_patch.copy()
        y_patch_for_translation_y = y_patch.copy()

        # In order to still be able to plot the computed values without
        # the boundary data, also the translation data has to be copied.
        patch_translation_x_with_bc = patch_translation_x.copy()
        patch_translation_y_with_bc = patch_translation_y.copy()

        # Begin with known translation at the left and right boundary
        for y_loc in np.linspace(0, img_src.height, self.N_patches[1] + 1):
            # Left boundary patches
            x_patch_for_translation_x.append(img_src.origo[0])
            y_patch_for_translation_x.append(img_src.origo[1] + y_loc)
            patch_translation_x_with_bc.append(0.0)

            # Right boundary patches
            x_patch_for_translation_x.append(img_src.origo[0] + img_src.width)
            y_patch_for_translation_x.append(img_src.origo[1] + y_loc)
            patch_translation_x_with_bc.append(0.0)

        # Also add known translation at the top and bottom
        for x_loc in np.linspace(0, img_src.width, self.N_patches[0] + 1):
            # Bottom boundary patches
            x_patch_for_translation_y.append(img_src.origo[0] + x_loc)
            y_patch_for_translation_y.append(img_src.origo[1])
            patch_translation_y_with_bc.append(0.0)

        # Finally define separate interpolators for the translation in x and y
        # directions.
        self.interpolator_translation_x = RBFInterpolator(
            np.array([x_patch_for_translation_x, y_patch_for_translation_x]).T,
            patch_translation_x_with_bc,
        )
        self.interpolator_translation_y = RBFInterpolator(
            np.array([x_patch_for_translation_y, y_patch_for_translation_y]).T,
            patch_translation_y_with_bc,
        )

        return (
            self.interpolator_translation_x,
            self.interpolator_translation_y,
            have_translation,
        )

    def find_translation(self, img_dst: daria.Image, plot_translation: bool = False):
        """
        Compute compaction map as interpolator and apply to patch centers.

        Args:
            img_dst (daria.Image): test image
            plot_translation (bool): flag controlling whether the translation is
                plotted in terms of arrows on top of the base and test images

        Returns:
            np.ndarray: translation in x direction
            np.ndarray: translation in y direction
        """
        # TODO: Add possibility to evaluate the interppolators in the corners of the patches

        # Determine interpolators for both component of the translation
        (
            interpolator_translation_x,
            interpolator_translation_y,
            have_patch_translation,
        ) = self.find_compaction_map(img_dst)

        # Interpolate at patch centers
        input_arg = np.transpose(
            np.vstack(
                (
                    np.flipud(self.patch_centers[:, :, 0].T).flatten(),
                    np.flipud(self.patch_centers[:, :, 1].T).flatten(),
                )
            )
        )
        interpolated_patch_translation_x = interpolator_translation_x(input_arg)
        interpolated_patch_translation_y = interpolator_translation_y(input_arg)

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
                        np.flipud(have_patch_translation.T), self.img_src.img.shape
                    ),
                    method="blend",
                )
            )
            plt.show()
