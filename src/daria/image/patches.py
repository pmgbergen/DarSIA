from math import ceil, sqrt

import numpy as np

import daria as da
import skimage


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
        patch_pixels_width = self.baseImg.coordinatesystem.lengthToPixels(
            patch_width, "x"
        )
        patch_pixels_height = self.baseImg.coordinatesystem.lengthToPixels(
            patch_height, "y"
        )
        # TODO rename and put pixels to the back

        # Convert the overlap from metric to numbers of pixels
        overlap_pixels_width = self.baseImg.coordinatesystem.lengthToPixels(
            self.patch_overlap, "x"
        )
        overlap_pixels_height = self.baseImg.coordinatesystem.lengthToPixels(
            self.patch_overlap, "y"
        )

        # Some abbreviation for better overview
        nh = self.baseImg.num_pixels_height
        pw = patch_pixels_width
        ph = patch_pixels_height
        ow = overlap_pixels_width
        oh = overlap_pixels_height
        cw = ceil(overlap_pixels_width / 2)
        ch = ceil(overlap_pixels_height / 2)
        off_w = 0 if cw == ow / 2.0 else 1
        off_h = 0 if ch == oh / 2.0 else 1

        # TODO if this works, change the convention of numbering of patches. and use the same as for images.

        # Define pixel-based ROIs - with overlap
        self.rois: list[list[tuple]] = [
            [
                (
                    slice(max(nh - (j + 1) * ph - oh, 0), nh - j * ph + oh),
                    slice(max(i * pw - ow, 0), (i + 1) * pw + ow),
                )
                for j in range(self.num_patches_y)
            ]
            for i in range(self.num_patches_x)
        ]

        # Define relative pixel-based ROIs corresponding to the area - without overlap
        self.relative_rois_without_overlap: list[list[tuple]] = [
            [
                (
                    slice(
                        0 if j == self.num_patches_y - 1 else oh,
                        ph if j == self.num_patches_y - 1 else ph + oh,
                    ),
                    slice(0 if i == 0 else ow, pw if i == 0 else pw + ow),
                )
                for j in range(self.num_patches_y)
            ]
            for i in range(self.num_patches_x)
        ]

        # Extract images with overlap
        self.images = [
            [
                da.extractROIPixel(self.baseImg, self.rois[i][j])
                for j in range(self.num_patches_y)
            ]
            for i in range(self.num_patches_x)
        ]

        # TODO consider new class Patch which stores all the interesting info?

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
        self.weight_x = {}

        # Corresponding to the most left patches
        self.weight_x["left"] = np.hstack(
            (
                np.ones(pw - cw, dtype=float),
                np.linspace(1, 0, ow),
                np.zeros(cw, dtype=float),
            )
        )

        # Corresponding to the internal patches
        self.weight_x["internal"] = np.hstack(
            (
                np.zeros(cw, dtype=float),
                np.linspace(0, 1, ow),
                np.ones(pw - 2 * cw, dtype=float),
                np.linspace(1, 0, ow),
                np.zeros(cw, dtype=float),
            )
        )

        # Have to take into account that not all patches have perfect size, and small
        # adjustments have to be made at the right boundary patches. Thus, compute the
        # so far total width occupied - goal: determine the amount of pixels which are
        # left for true 1 values in the weight corresponding to the most right patch.
        marked_width = (
            # left patch
            pw
            - cw
            + ow
            + off_w
            # all internal patches
            + (self.num_patches_x - 2) * (pw - 2 * cw + ow + off_w)
        )

        # Corresponding to the most right patches
        self.weight_x["right"] = np.hstack(
            (
                np.zeros(cw, dtype=float),
                np.linspace(0, 1, ow),
                np.ones(self.baseImg.num_pixels_width - marked_width, dtype=float),
            )
        )

        # Analogously, define the weighting in y-direction.
        # NOTE: The weight has to be defined consistently with the conventional pixel ordering
        # of images, i.e., the first pixel lies at the top.
        self.weight_y = {}

        # Corresponding to the bottom patches
        self.weight_y["bottom"] = np.hstack(
            (
                np.zeros(ch, dtype=float),
                np.linspace(0, 1, oh),
                np.ones(ph - ch, dtype=float),
            )
        )

        # Corresponding to the internal patches
        self.weight_y["internal"] = np.hstack(
            (
                np.zeros(ch, dtype=float),
                np.linspace(0, 1, oh),
                np.ones(ph - 2 * ch, dtype=float),
                np.linspace(1, 0, oh),
                np.zeros(ch, dtype=float),
            )
        )

        # Analogously to weight_x, have to determine the marked height
        marked_height = (
            # bottom patch
            (ph - ch + oh + off_h)
            # all internal patches
            + (self.num_patches_y - 2) * (ph - 2 * ch + oh + off_h)
        )

        # Corresponding to the top patches
        self.weight_y["top"] = np.hstack(
            (
                np.ones(self.baseImg.num_pixels_height - marked_height, dtype=float),
                np.linspace(1, 0, oh),
                np.zeros(ch, dtype=float),
            )
        )

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
        elif j == self.num_patches_y - 1:
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
        assert self.images[i][j].img.shape == img.shape
        self.images[i][j].img = img.copy()

    def assemble(self, update_img: bool = False) -> da.Image:
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
            rel_roi = self.relative_rois_without_overlap[0][j]
            assembled_row_j = self.images[0][j].img[rel_roi]

            # And assemble the remainder of the row by concatenation.
            for i in range(1, self.num_patches_x):
                rel_roi = self.relative_rois_without_overlap[i][j]
                assembled_row_j = np.hstack(
                    (assembled_row_j, self.images[i][j].img[rel_roi])
                )

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
        assembled_img = np.zeros_like(self.baseImg.img, dtype=float)

        # Loop over patches
        for j in range(self.num_patches_y):

            # Allocate memory for row j
            shape = [self.images[0][j].num_pixels_height, *self.baseImg.img.shape[1:]]
            assembled_row_j = np.zeros(tuple(shape), dtype=float)

            # Assemble row j by suitable weighted combination of the patches in row j
            for i in range(self.num_patches_x):

                # Determine active pixel range
                roi = self.rois[i][j]
                roi_x = roi[1]

                # Fetch patch, and convert to float
                img_i_j = skimage.util.img_as_float(self.images[i][j].img)

                # Fetch weight and convert to tensor
                weight_i_j = self.weight_x[self.position(i, j)[0]]
                # Convert to tensor
                weight_i_j = weight_i_j.reshape(1, np.size(weight_i_j), 1)

                # Add weighted patch at the active pixel range
                assembled_row_j[:, roi_x] += np.multiply(img_i_j, weight_i_j)

            # Anologous procedure, but now on row-level and not single-patch level.

            # Determine active pixel range. NOTE: roi[0] still contains the relevant
            # pixel range, relevant for addressing the base image.
            roi_y = roi[0]

            # Fetch weight
            weight_j = self.weight_y[self.position(i, j)[1]]
            # Convert to tensor
            weight_j = weight_j.reshape(np.size(weight_j), 1, 1)

            # Add weighted row at the active pixel range
            assembled_img[roi_y, :] += np.multiply(assembled_row_j, weight_j)

        # Make sure the newly assembled image is compatible with the original base image
        assert assembled_img.shape == self.baseImg.img.shape

        # Convert final image to uint8 format
        assembled_img = skimage.util.img_as_ubyte(assembled_img)

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
