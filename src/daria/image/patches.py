from math import ceil, sqrt

import numpy as np

import daria as da


class Patches:
    """
    Class for Patched image

    Contains an array of patches of a prescribed full image.

    Attributes:
        baseImg (da.Image) = full daria image
        num_patches_x (int) = number of patches in the horizondal direction
        num_patches_y (int) = number of patches in the vertical direction
        images (np.ndarray)= array of patches of the original image

    NOTE: A Cartesian ordering is used to refer to specific patches, i.e.,
        The lower left patch will have the patch coordinate (0,0), while the
        top right patch will have the patch coordinate
        (num_patches_x-1, num_patches_y-1).

    Methods:
        assemble (da.Image): reassembles the patches to a new image
    """

    def __init__(self, im: da.Image, *args, **kwargs) -> None:
        """
        Constructor for Patches class.

        Arguments:
            im (da.Image): the Image that is to be patched
            *args: information on how many patches are created;
                        eiher one can give the amount of patches as one argument
                        (needs to be a square number),
                        or one can give the number of patches in x, and y direction
                        as two separate arguments, respectively.
            **kwargs: optional keyword arguments:
                "patch_overlap" (int): overlap of each patch in each direction of a rectangular
                    patch, given in metric units (same as for im); default value is 0.
        """

        # Instance of base image and patch_overlap
        self.baseImg = im
        self.patch_overlap = kwargs.pop("patch_overlap", 0)  # TODO rename to overlap

        # Define number of patches in each direction
        if len(args) == 1:
            self.num_patches_x = int(sqrt(args[0]))
            self.num_patches_y = int(sqrt(args[0]))

        elif len(args) == 2:
            self.num_patches_x = int(args[0])
            self.num_patches_y = int(args[1])

        else:
            raise ValueError("Provide either a single or two arguments.")

        # Define base width and height of each patch (without overlap)...
        # ... in metric units
        patch_width = self.baseImg.width / self.num_patches_x
        patch_height = self.baseImg.height / self.num_patches_y
        # ... and in numbers of pixles
        patch_pixels_width = ceil(patch_width / self.baseImg.dx)
        patch_pixels_height = ceil(patch_height / self.baseImg.dy)

        # Convert the overlap from metric to numbers of pixels
        overlap_pixels_width = ceil(self.patch_overlap / self.baseImg.dx)
        overlap_pixels_height = ceil(self.patch_overlap / self.baseImg.dy)

        # TODO consider new class Patch which stores all the interesting info?

        # Extract patches (images and rois) and use a Cartesian ordering for patch
        # coordinates
        images_and_rois: list[list[tuple]] = [
            [
                da.extractROI(
                    self.baseImg,
                    [
                        max(0, i * patch_width - self.patch_overlap),
                        min(
                            self.baseImg.width,
                            (i + 1) * patch_width + self.patch_overlap,
                        ),
                    ],
                    [
                        max(0, j * patch_height - self.patch_overlap),
                        min(
                            self.baseImg.height,
                            (j + 1) * patch_height + self.patch_overlap,
                        ),
                    ],
                    return_roi=True,
                )
                for j in range(self.num_patches_y)
            ]
            for i in range(self.num_patches_x)
        ]

        # Extract the images
        self.images: list[list[da.Image]] = [
            [images_and_rois[i][j][0] for j in range(self.num_patches_y)]
            for i in range(self.num_patches_x)
        ]

        # Extract the rois
        # TODO rename? do we also require rois without overlap?
        self.rois_with_overlap: list[list[tuple]] = [
            [images_and_rois[i][j][1] for j in range(self.num_patches_y)]
            for i in range(self.num_patches_x)
        ]
        # TODO self.rois_without_overlap = None

        # Store centers (x,y) of each patch in meters
        self.centers = [
            [
                np.array([(i + 0.5) * patch_width, (j + 0.5) * patch_height])
                for j in range(self.num_patches_y)
            ]
            for i in range(self.num_patches_x)
        ]

        # Define partition of unity later to be used as weighting masks for blending in
        # x-direction. Distinguish between th three different cases: (i) left border,
        # (ii) internal, and (iii) right border.
        self.weight_x = {
            "left": np.hstack(
                (
                    np.ones(
                        patch_pixels_width - ceil(overlap_pixels_width / 2), dtype=float
                    ),
                    np.linspace(1, 0, overlap_pixels_width),
                    np.zeros(ceil(overlap_pixels_width / 2), dtype=float),
                )
            ),
            "internal": np.hstack(
                (
                    np.zeros(ceil(overlap_pixels_width / 2), dtype=float),
                    np.linspace(0, 1, overlap_pixels_width),
                    np.ones(
                        patch_pixels_width - 2 * ceil(overlap_pixels_width / 2),
                        dtype=float,
                    ),
                    np.linspace(1, 0, overlap_pixels_width),
                    np.zeros(ceil(overlap_pixels_width / 2), dtype=float),
                )
            ),
            "right": np.hstack(
                (
                    np.zeros(ceil(overlap_pixels_width / 2), dtype=float),
                    np.linspace(0, 1, overlap_pixels_width),
                    np.ones(
                        patch_pixels_width - ceil(overlap_pixels_width / 2), dtype=float
                    ),
                )
            ),
        }

        # Analogously, define the weighting in y-direction.
        # NOTE: The weight has to be defined consistently with the conventional pixel ordering
        # of images, i.e., the first pixel lies at the top.
        self.weight_y = {
            "top": np.hstack(
                (
                    np.ones(
                        patch_pixels_height - ceil(overlap_pixels_height / 2),
                        dtype=float,
                    ),
                    np.linspace(1, 0, overlap_pixels_height),
                    np.zeros(ceil(overlap_pixels_height / 2), dtype=float),
                )
            ),
            "internal": np.hstack(
                (
                    np.zeros(ceil(overlap_pixels_height / 2), dtype=float),
                    np.linspace(0, 1, overlap_pixels_height),
                    np.ones(
                        patch_pixels_height - 2 * ceil(overlap_pixels_height / 2),
                        dtype=float,
                    ),
                    np.linspace(1, 0, overlap_pixels_height),
                    np.zeros(ceil(overlap_pixels_height / 2), dtype=float),
                )
            ),
            "bottom": np.hstack(
                (
                    np.zeros(ceil(overlap_pixels_height / 2), dtype=float),
                    np.linspace(0, 1, overlap_pixels_height),
                    np.ones(
                        patch_pixels_height - ceil(overlap_pixels_height / 2),
                        dtype=float,
                    ),
                )
            ),
        }

    def position(self, i: int, j: int) -> tuple[str]:
        """
        Determine positioning of patch wrt. boundary or internal patches
        in both x- and y-direction.

        Args:
            i (int): patch coordinate in x-direction
            j (int): patch coordinate in y-direction

        NOTE: The patch coordinates employ the Cartesian ordering.

        Returns:
            str: "left" or "right" if the patch is touching the left or right boundary
                of the image; otherwise "internal"
            str: "top" or "bottom" if the patch is touching the top or bottom boundary
                of the image; otherwise "internal"
        """
        # Determine horizontal position (x-direction)
        if i == 0:
            horizontal_position: str = "left"
        elif i == self.num_patches_x - 1:
            horizontal_position = "right"
        else:
            horizontal_position = "internal"

        # Determine vertical position (y-direction)
        if j == 0:
            vertical_position: str = "bottom"
        elif i == self.num_patches_y - 1:
            vertical_position = "top"
        else:
            vertical_position = "internal"

        return horizontal_position, vertical_position

    def __call__(self, i: int, j: int) -> np.ndarray:
        """Return patch with patch coordinates (i,j)."""
        return self.images[i][j]

    def set_image(self, img: np.ndarray, i: int, j: int) -> None:
        """
        Update the image of a patch.
        """
        # TODO redundant if this class only contains patches (a new object type)
        assert self.images[i][j].img.shape == img.shape
        self.images[i][j].img = img.copy()

    def assemble(self) -> da.Image:
        """
        Reassembles the patches into a new daria image.
        """
        # Create temporary image-array and add the lower left-corner patch to it
        im_tmp_x = self.images[0][0].img

        # Add patches along the bottom to the temporary image to create the
        # lower part of the image
        for i in range(self.num_patches_x - 1):
            im_tmp_x = np.c_["1,0,0", im_tmp_x, self.images[i + 1][0].img]

        # Create actual temporary image
        im_tmp = im_tmp_x

        # Repeate the process and create "image-strips" that are added to the
        # top of the temporary image
        for j in range(self.num_patches_y - 1):
            im_tmp_x = self.images[0][j + 1].img
            for i in range(self.num_patches_x - 1):
                im_tmp_x = np.c_["1,0,0", im_tmp_x, self.images[i + 1][j + 1].img]
            im_tmp = np.r_["0,1,0", im_tmp_x, im_tmp]

        # TODO review
        print(im_tmp.shape, self.baseImg.img.shape)

        # Return the resulting daria image
        return da.Image(
            im_tmp,
            origo=self.baseImg.origo,
            width=self.baseImg.width,
            height=self.baseImg.height,
        )

    def blend_and_assemble(self, update_img: bool = False) -> da.Image:
        """
        Reassembles taking into account the overlap as well.
        On the overlap, a convex combination is used for
        smooth blending.

        Args:
            update_img (bool): flag controlling whether the base image will be updated
                with the assembled image; default set to False

        Returns:
            daria.image: assembled image as daria image
        """
        # The procedure is as follows. The image is reassembled, row by row.
        # Each row is reconstructed by concatenation. Overlapping regions
        # have to handled separately by a weighted sum (convex combination).
        # For this, the above defined partition of unity will be used.
        # Then, rows are combined in a similar manner, but now entire rows are
        # concatenated, and added using the partition of unity.

        # Allocate memory for resassembled image
        assembled_img = np.zeros_like(self.baseImg.img, dtype=self.baseImg.img.dtype)

        # Loop over patches
        for j in range(self.num_patches_y):

            # Allocate memory for row j
            assembled_row_j = np.zeros_like(
                (self.images[0][j].num_pixels_height, self.img.num_pixels_width),
                dtype=self.img.img.dtype,
            )

            # Assemble row j by suitable weighted combination of the patches in row j
            for i in range(self.num_patches_x):

                # Determine active pixel range
                roi = self.rois_with_overlap[i][j]
                roi_x = roi[1]

                # Fetch patch
                img_i_j = self.images[i][j].img[roi_x]

                # Fetch weight
                weight_i_j = self.weight_x[self.position(i, j)[0]]

                # Add weighted patch at the active pixel range
                assembled_row_j[:, roi_x] += np.multiply(img_i_j, weight_i_j)

            # Anologous procedure, but now on row-level and not single-patch level.

            # Determine active pixel range. NOTE: roi[0] still contains the relevant
            # pixel range, relevant for addressing the base image.
            roi_y = roi[0]

            # Determine size of roi in y-direction
            roi_y_size = roi_y.stop - roi_y.start

            # Fetch weight
            weight_j = self.weight_y[self.position(i, j)[1]]

            # Add weighted row at the active pixel range
            assembled_img[roi_y, :] += np.multiply(
                assembled_row_j, weight_j.reshape(roi_y_size, 1)
            )

        # Make sure the newly assembled image is compatible with the original base image
        assert assembled_img.shape == self.baseImg.img.shape

        # Define resulting daria image
        da_assembled_img = da.Image(
            img=assembled_img,
            origo=self.baseImg.origo,
            width=self.baseImg.width,
            height=self.baseImg.height,
        )

        # Update the base image if required
        if update_img:
            self.baseImg = da_assembled_img.copy()

        return da_assembled_img

    def reassemble(self, update_img: bool = False) -> da.Image:
        """
        Reassembles without taking into account the overlap.

        Args:
            update_img (bool): flag controlling whether the base image will be updated
                with the assembled image; default set to False

        Returns:
            daria.image: assembled image as daria image
        """

        # Initialize empty row of the final image. It will be used
        # to assemble the patches row by row.
        assembled_img = np.zeros(
            (0, *self.baseImg.img.shape[1:]), dtype=self.baseImg.img.dtype
        )

        # Create "image-strips" that are assembled by concatenation
        for j in range(self.num_patches_y):
            # Initialize the row of the final image with the first patch image.
            assembled_row_j = self.images[0][j].img
            # And assemble the remainder of the row by concatenation.
            for i in range(1, self.num_patches_x):
                assembled_row_j = np.hstack((assembled_row_j, self.images[i][j].img))
            # Concatenate the row and the current image
            assembled_img = np.vstack((assembled_row_j, assembled_img))

        # Make sure that the resulting image has the same resolution
        assert assembled_img.shape == self.baseImg.img.shape

        # Define resulting daria image
        da_assembled_img = da.Image(
            img=assembled_img,
            origo=self.baseImg.origo,
            width=self.baseImg.width,
            height=self.baseImg.height,
        )

        # Update the base image if required
        if update_img:
            self.baseImg = da_assembled_img.copy()

        return da_assembled_img
