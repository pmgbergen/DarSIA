"""
Module containing class for translation analysis, relevant e.g. for
studying compaction of porous media.
"""

from typing import Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.interpolate import RBFInterpolator

import darsia


class TranslationAnalysis:
    """
    Class for translation analysis.
    """

    def __init__(
        self,
        base: darsia.Image,
        N_patches: list[int],
        rel_overlap: float,
        translationEstimator: darsia.TranslationEstimator,
        mask: Optional[darsia.Image] = None,
    ) -> None:
        """
        Constructor for TranslationAnalysis.

        It allows to determine the translation of any image to a given baseline image
        in order to provide a best-possible match, based on feature detection.

        Args:
            base (darsia.Image): baseline image; it serves as fixed point in the analysis,
                which is relevant if a series of translations is analyzed. Furthermore, the
                baseline image provides all reference values as the coordinate system, e.g.
            N_patches (list of two int): number of patches in x and y direction
            rel_overlap (float): relative overal related to patch size in each direction
        """
        # Store parameters
        self.N_patches = N_patches
        self.rel_overlap = rel_overlap
        self.translationEstimator = translationEstimator

        # Construct patches of the base image
        self.update_base(base)

        # Initialize translation with zero, allowing summation of translation
        def zero_translation(arg):
            return np.transpose(np.zeros_like(arg))

        self.translation = zero_translation
        self.have_translation = np.zeros(tuple(self.N_patches), dtype=bool)

        # Cache mask
        if mask is None:
            self.mask_base = darsia.Image(
                np.ones(base.img.shape[:2], dtype=bool),
                width=base.width,
                height=base.height,
            )
        else:
            self.mask_base = mask.copy()

    # TOOD add update_base methods similar to other tools

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
            self.N_patches = cast(list[int], N_patches)

        if need_update_rel_overlap:
            self.rel_overlap = cast(float, rel_overlap)

        # Update the patches of the base image accordingly.
        if need_update_N_patches or need_update_rel_overlap:
            self.update_base_patches()

    def update_base(self, base: darsia.Image) -> None:
        """Update baseline image.

        Args:
            base (darsia.Image): baseline image
        """
        self.base = base
        self.update_base_patches()

    def update_base_patches(self) -> None:
        """Update patches of baseline."""
        # Create new patches of the base image.
        self.patches_base = darsia.Patches(
            self.base, *self.N_patches, rel_overlap=self.rel_overlap
        )

    def load_image(
        self, img: darsia.Image, mask: Optional[darsia.Image] = None
    ) -> None:
        """Load an image to be inspected in futher analysis.

        Args:
            img (darsia.Image): test image
        """
        self.img = img

        # Cache mask
        if mask is None:
            self.mask_img = darsia.Image(
                np.ones(img.img.shape[:2], dtype=bool),
                width=img.width,
                height=img.height,
            )
        else:
            self.mask_img = mask.copy()

        # TODO, apply patching here already? why not?

    def find_translation(self, units: list[str] = ["pixel", "pixel"]) -> tuple:
        # TODO ideally this method should not require units; only the application routine does.
        """
        Find translation map as translation from image to baseline image such that
        these match as best as possible, measure on features.

        The final translation map will be stored as callable function. And it allows
        various input and output spaces (metric vs. pixel).

        Args:
            units (list of str): units for input (first entry) and output (second entry)
                ranges of the resulting translation map; accepts either "metric"
                or "pixel".
            mask (np.ndarray, optional): boolean mask marking all pixels to be considered;
                all if mask is None (default).

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
        patches_img = darsia.Patches(
            self.img, *self.N_patches, rel_overlap=self.rel_overlap
        )

        # Monitor success of finding a translation/homography for each patch
        have_translation = np.zeros(tuple(self.N_patches), dtype=bool)

        # Initialize containers for coordinates of the patches as well as the translation
        # Later to be used for interpolation.
        input_coordinates: list[np.ndarray] = []
        patch_translation_x: list[float] = []
        patch_translation_y: list[float] = []

        # Fetch patch centers
        patch_centers_cartesian = self.patches_base.global_centers_cartesian

        # Continue with investigating all patches. The main idea is to determine
        # a suitable patchwise (discontinuous) homography, and extract an effective
        # translation for each patch. The following procedure does not work
        # for all patches; those unsuccessful ones will be covered by interpolation
        # aferwards.

        # TODO adaptive refinement stategy - as safety measure! will require
        # some effort in how to access patches. try first with fixed params.

        # Create patches of masks
        patches_mask_base = darsia.Patches(
            self.mask_base, *self.N_patches, rel_overlap=self.rel_overlap
        )
        patches_mask_img = darsia.Patches(
            self.mask_img, *self.N_patches, rel_overlap=self.rel_overlap
        )

        # Loop over all patches.
        for i in range(self.N_patches[0]):
            for j in range(self.N_patches[1]):

                # Fetch patches of both source and destination image
                patch_base = self.patches_base(i, j)
                patch_img = patches_img(i, j)

                # Fetch corresponding patches of mask
                patch_mask_base = patches_mask_base(i, j)
                patch_mask_img = patches_mask_img(i, j)

                # Determine effective translation from input to baseline image, operating on
                # pixel coordinates and using reverse matrix indexing.
                (
                    translation,
                    intact_translation,
                ) = self.translationEstimator.find_effective_translation(
                    patch_img.img,
                    patch_base.img,
                    None,
                    None,
                    mask_src=patch_mask_img.img,
                    mask_dst=patch_mask_base.img,
                    plot_matches=False,
                )

                # The above procedure to find a matching transformation is successful if in
                # any of the iterations, a translation has been found. If so, postprocess
                # and store the effective translation.
                if intact_translation:

                    # Flag success
                    have_translation[i, j] = True

                    # Fetch the center of the patch in metric units, which will be the input
                    # for later construction of the interpolator
                    center = patch_centers_cartesian[i, j]

                    # Convert to pixel units using reverse matrix indexing, if required
                    if units[0] == "pixel":
                        center = self.base.coordinatesystem.coordinateToPixel(
                            center, reverse=True
                        )

                    # Extract the effective displacement, stored in the constant part of
                    # the affine map, in pixel units, using reverse matrix indexing.
                    displacement = translation[:, -1]

                    # Convert to metric units if required - note that displacement is a
                    # vector, and that it is given using reverse matrix indexing
                    if units[1] == "metric":
                        displacement = (
                            self.base.coordinatesystem.pixelToCoordinateVector(
                                displacement, reverse=True
                            )
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
        def translation_callable(arg):
            return np.array(
                [
                    self.interpolator_translation_x(arg),
                    self.interpolator_translation_y(arg),
                ]
            )

        self.translation = translation_callable

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
                self.base.origin + np.array([0, y_pos])
                for y_pos in np.linspace(0, self.base.height, self.N_patches[1] + 1)
            ]
            # Add the right vertical boundary
            vertical_boundary += [
                self.base.origin + np.array([self.base.width, y_pos])
                for y_pos in np.linspace(0, self.base.height, self.N_patches[1] + 1)
            ]

        elif units[0] == "pixel":
            # Add the left vertical boundary - comply to reverse matrix indexing
            vertical_boundary += [
                np.array([0, y_pos])
                for y_pos in np.linspace(
                    0, self.base.num_pixels_height, self.N_patches[1] + 1
                )
            ]
            # Add the right vertical boundary - comply to reverse matrix indexing
            vertical_boundary += [
                np.array([self.base.num_pixels_width, y_pos])
                for y_pos in np.linspace(
                    0, self.base.num_pixels_height, self.N_patches[1] + 1
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
                self.base.origin + np.array([x_pos, 0])
                for x_pos in np.linspace(0, self.base.width, self.N_patches[0] + 1)
            ]

        elif units[0] == "pixel":
            # Add the bottom horizontal boundary - comply to reverse matrix indexing
            horizontal_boundary += [
                np.array([x_pos, self.base.num_pixels_height])
                for x_pos in np.linspace(
                    0, self.base.num_pixels_width, self.N_patches[0] + 1
                )
            ]

        return horizontal_boundary, len(horizontal_boundary) * [0.0]

    def return_patch_translation(
        self, reverse: bool = True, units: str = "metric"
    ) -> np.ndarray:
        """
        Translate patch centers of the test image.

        Args:
            reverse (bool): flag whether the translation is understood as from the
                test image to the baseline image, or reversed. The default is the
                former latter.
            units (list of str): "metric" or "pixel"

        Returns:
            np.ndarray: deformation in patch centers
        """
        # Only continue if a translation has been already found
        # assert self.have_translation.any()

        # Interpolate at patch centers (pixel coordinates with reverse matrix indexing)
        patch_centers = self.patches_base.global_centers_reverse_matrix
        patch_centers_shape = patch_centers.shape
        patch_centers = patch_centers.reshape((-1, 2))
        interpolated_patch_translation = self.translation(patch_centers)

        # Flip, if required
        if reverse:
            interpolated_patch_translation *= -1.0

        # Collect patch_translation using standard matrix indexing - have to flip components
        patch_translation = np.vstack(
            (interpolated_patch_translation[1], interpolated_patch_translation[0])
        ).T

        # Convert units if needed and provide in metric units
        if units == "metric":
            patch_translation = self.base.coordinatesystem.pixelToCoordinateVector(
                patch_translation
            )

        # Return in patch format
        return patch_translation.reshape(patch_centers_shape)

    def plot_translation(
        self,
        reverse: bool = True,
        scaling: float = 1.0,
        mask: Optional[darsia.Image] = None,
    ) -> None:
        """
        Translate centers of the test image and plot in terms of displacement arrows.
        """
        # Fetch the patch centers
        patch_centers = self.patches_base.global_centers_reverse_matrix.reshape((-1, 2))

        # Determine patch translation in matrix ordering (and with flipped y-direction
        # to comply with the orientation of the y-axis in imaging.
        patch_translation = self.return_patch_translation(
            reverse, units="pixel"
        ).reshape((-1, 2))
        patch_translation_y = patch_translation[:, 0]
        patch_translation_x = patch_translation[:, 1]

        # For visual purposes, since the row-axis / y-axis has a negative orientation,
        # scale the translation in y-direction.
        patch_translation_y *= -1.0

        # Restrict to values covered by mask
        if mask is not None:
            # Determine active and inactive set
            assert mask.img.dtype == bool
            num_patches = patch_centers.shape[0]
            active_set = np.ones(num_patches, dtype=bool)
            for i in range(num_patches):
                # Fetch the position
                pixel_pos_patch = patch_centers[i, :2]
                coord_pos_patch = self.base.coordinatesystem.pixelToCoordinate(
                    pixel_pos_patch, reverse=True
                )
                # Find pixel coordinate in mask
                pixel_pos_mask = mask.coordinatesystem.coordinateToPixel(
                    coord_pos_patch
                )
                if not mask.img[pixel_pos_mask[0], pixel_pos_mask[1]]:
                    active_set[i] = False
            inactive_set = np.logical_not(active_set)

            # Damp translation in inexactive set
            patch_translation_x[inactive_set] = 0
            patch_translation_y[inactive_set] = 0

            # Deactive unmasked points
            active_patch_centers = patch_centers[active_set]
            active_patch_translation_x = patch_translation_x[active_set]
            active_patch_translation_y = patch_translation_y[active_set]

        else:
            active_patch_centers = patch_centers
            active_patch_translation_x = patch_translation_x
            active_patch_translation_y = patch_translation_y

        c = np.sqrt(
            np.power(active_patch_translation_x, 2)
            + np.power(active_patch_translation_y, 2)
        )

        # Plot the interpolated translation
        fig, ax = plt.subplots(1, num=1)
        ax.quiver(
            active_patch_centers[:, 0],
            active_patch_centers[:, 1],
            active_patch_translation_x * scaling,
            active_patch_translation_y * scaling,
            c,
            scale=1000,
            alpha=0.5,
            # color="white",
            cmap="viridis",
        )
        ax.imshow(
            self.base.img
            # skimage.util.compare_images(self.base.img, self.img.img, method="blend")
        )
        plt.figure("Deformation arrow")
        plt.quiver(
            active_patch_centers[:, 0],
            active_patch_centers[:, 1],
            active_patch_translation_x * scaling,
            active_patch_translation_y * scaling,
            c,
            scale=1000,
            alpha=0.5,
            # color="white",
            cmap="viridis",
        )
        plt.imshow(
            self.base.img
            # skimage.util.compare_images(self.base.img, self.img.img, method="blend")
        )

        # For storing, uncomment:
        # plt.savefig("deformation.svg", format="svg", dpi=1000)
        # translation_length = np.max(np.sqrt(np.power(active_patch_translation_x, 2) + np.power(active_patch_translation_y, 2)))
        # translation_length_SI = self.patches_base.base.coordinatesystem
        # print(f"max length: {translation_length * self.patches_base.base.dx}, {self.patches_base.base.dx}, {self.patches_base.base.dy}")
        # plt.figure("length")
        # plt.imshow(np.sqrt(np.power(active_patch_translation_x, 2) + np.power(active_patch_translation_y, 2)).reshape(1,-1) * self.patches_base.base.dx)
        # cbar = plt.colorbar()
        # cbar.set_ticks(np.linspace(0, translation_length * self.patches_base.base.dx, 2))

        plt.figure("Success")
        plt.imshow(
            skimage.util.compare_images(
                self.base.img,
                skimage.transform.resize(
                    np.flipud(self.have_translation.T), self.base.img.shape
                ),
                method="blend",
            )
        )
        #plt.savefig("success.svg", format="svg", dpi=1000)

        # Plot deformation in number of pixels
        plt.figure("deformation x pixels")
        plt.title("Deformation in x-direction in pixels")
        plt.imshow(patch_translation_x.reshape(self.N_patches[1], self.N_patches[0]))
        plt.colorbar()
        plt.figure("deformation y pixels")
        plt.title("Deformation in y-direction in pixels")
        plt.imshow(patch_translation_y.reshape(self.N_patches[1], self.N_patches[0]))
        plt.colorbar()

        # Plot deformation in meters
        plt.figure("deformation x meters")
        plt.title("Deformation in x-direction in meters")
        plt.imshow(
            patch_translation_x.reshape(self.N_patches[1], self.N_patches[0])
            * self.base.dx
        )
        plt.colorbar()
        plt.figure("deformation y meters")
        plt.title("Deformation in y-direction in meters")
        plt.imshow(
            patch_translation_y.reshape(self.N_patches[1], self.N_patches[0])
            * self.base.dy
        )
        plt.colorbar()
        plt.show()

    def translate_image(
        self,
        reverse: bool = True,
    ) -> darsia.Image:
        """
        Apply translation to an entire image by using piecwise perspective transformation.

        Args:
            reverse (bool): flag whether the translation is understood as from the
                test image to the baseline image, or reversed. The default is the
                latter.

        Returns:
            darsia.Image: translated image
        """

        # Segment the test image into cells by patching without overlap
        patches = darsia.Patches(self.img, *self.N_patches, rel_overlap=0.0)

        # Create piecewise perspective transform on the patches
        perspectiveTransform = darsia.PiecewisePerspectiveTransform()
        import time

        tic = time.time()
        transformed_img: darsia.Image = perspectiveTransform.find_and_warp(
            patches, self.translation, reverse
        )
        print(f"find and warp takes {time.time() - tic}")

        return transformed_img

    def __call__(
        self,
        img: darsia.Image,
        reverse: bool = False,
        mask: Optional[darsia.Image] = None,
    ) -> darsia.Image:
        """
        Standard workflow, starting with loading the test image, finding
        the translation required to match the baseline image,
        and then apply the translation.

        Args:
            img (darsia.Image): test image, to be matched with the baseline image

        Returns:
            darsia.Image: translated image
        """
        self.load_image(img, mask)
        import time

        tic = time.time()
        self.find_translation()
        print(f"find takes: {time.time() - tic}")
        transformed_img = self.translate_image(reverse)

        return transformed_img

    def deduct_translation_analysis(
        self, translation_analysis  #: TranslationAnalysis
    ) -> None:
        """
        Overwrite translation analysis by deducting from external one.
        (Re)defines the interpolation object.

        Args:
            translation_analysis (darsia.TranslationAnalysis): translation analysis
                holding an interpolation object.
        """

        # ! ---- Step 1. Patch analysis.

        # Initialize containers for coordinates of the patches as well as the translation
        # Later to be used for interpolation.
        input_coordinates: list[np.ndarray] = []
        patch_translation_x: list[float] = []
        patch_translation_y: list[float] = []

        # Fetch patch centers
        patch_centers_cartesian = self.patches_base.global_centers_cartesian

        # Loop over all patches.
        for i in range(self.N_patches[0]):
            for j in range(self.N_patches[1]):

                # Fetch the center of the patch in metric units, which will be the input
                # for later construction of the interpolator
                center = patch_centers_cartesian[i, j]

                # TODO?
                # Convert to pixel units using reverse matrix indexing, if required
                if True:
                    center = self.base.coordinatesystem.coordinateToPixel(
                        center, reverse=True
                    )

                # Evaluate translation provided by the external translation analysis
                displacement = translation_analysis.translation(center.reshape(-1, 2))

                # Store the displacement for the centers of the patch.
                # NOTE: In any case, the first and second components
                # correspond to the x and y component.
                input_coordinates.append(center)
                patch_translation_x.append(displacement[0])
                patch_translation_y.append(displacement[1])

        # ! ---- Step 2. Boundary conditions.

        # Fetch predetermined conditions (do not have to be on the boundary)
        units = ["pixel", "pixel"]  # TODO?
        extra_coordinates_x, extra_translation_x = self.bc_x(units)
        extra_coordinates_y, extra_translation_y = self.bc_y(units)

        # Define new interpolation objects
        self.interpolator_translation_x = RBFInterpolator(
            input_coordinates + extra_coordinates_x,
            patch_translation_x + extra_translation_x,
        )
        self.interpolator_translation_y = RBFInterpolator(
            input_coordinates + extra_coordinates_y,
            patch_translation_y + extra_translation_y,
        )

        # Convert interpolators to a callable displacement/translation map
        def translation_callable(arg):
            return np.array(
                [
                    self.interpolator_translation_x(arg),
                    self.interpolator_translation_y(arg),
                ]
            )

        self.translation = translation_callable

    def add_translation_analysis(
        self, translation_analysis  #: TranslationAnalysis
    ) -> None:
        """
        Add another translation analysis to the existing one.
        Modifies the interpolation object by redefinition.

        Args:
            translation_analysis (darsia.TranslationAnalysis): Translation analysis holding
                an interpolation object.
        """

        # ! ---- Step 1. Patch analysis.

        # Initialize containers for coordinates of the patches as well as the translation
        # Later to be used for interpolation.
        input_coordinates: list[np.ndarray] = []
        patch_translation_x: list[float] = []
        patch_translation_y: list[float] = []

        # Fetch patch centers
        patch_centers_cartesian = self.patches_base.global_centers_cartesian

        # Loop over all patches.
        for i in range(self.N_patches[0]):
            for j in range(self.N_patches[1]):

                # Fetch the center of the patch in metric units, which will be the input
                # for later construction of the interpolator
                center = patch_centers_cartesian[i, j]
                pixel_center = self.base.coordinatesystem.coordinateToPixel(
                    center, reverse=True
                )

                # Find displaced center
                pixel_displacement = self.translation(pixel_center.reshape(-1, 2))
                displaced_pixel_center = pixel_center + pixel_displacement.reshape(
                    pixel_center.shape
                )

                # Determine the additional displacement
                additional_pixel_displacement = translation_analysis.translation(
                    displaced_pixel_center.reshape(-1, 2)
                )

                # Total displacement is sum and it is the effective displacement of
                # the combined maps
                total_pixel_displacement = (
                    pixel_displacement + additional_pixel_displacement
                )

                # Store the displacement for the centers of the patch.
                # NOTE: In any case, the first and second components
                # correspond to the x and y component.
                input_coordinates.append(pixel_center)
                patch_translation_x.append(total_pixel_displacement[0])
                patch_translation_y.append(total_pixel_displacement[1])

        # ! ---- Step 2. Boundary conditions.

        # Fetch predetermined conditions (do not have to be on the boundary)
        units = ["pixel", "pixel"]  # TODO?
        extra_coordinates_x, extra_translation_x = self.bc_x(units)
        extra_coordinates_y, extra_translation_y = self.bc_y(units)

        # Define new interpolation objects
        self.interpolator_translation_x = RBFInterpolator(
            input_coordinates + extra_coordinates_x,
            patch_translation_x + extra_translation_x,
        )
        self.interpolator_translation_y = RBFInterpolator(
            input_coordinates + extra_coordinates_y,
            patch_translation_y + extra_translation_y,
        )

        # Convert interpolators to a callable displacement/translation map
        def translation_callable(arg):
            return np.array(
                [
                    self.interpolator_translation_x(arg),
                    self.interpolator_translation_y(arg),
                ]
            )

        self.translation = translation_callable
