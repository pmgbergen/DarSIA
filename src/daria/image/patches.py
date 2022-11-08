from math import ceil, sqrt
from typing import cast

import numpy as np
import skimage

import daria as da


class Patches:
    """
    Class for Patched image

    Contains an array of patches of a prescribed full image.

    Attributes:
        base (da.Image) = full daria image
        num_patches_x (int) = number of patches in the horizondal direction
        num_patches_y (int) = number of patches in the vertical direction
        images (np.ndarray)= array of patches of the original image

    NOTE: A Cartesian indexing is used to refer to specific patches, i.e.,
        The lower left patch will have the patch coordinate (0,0), while the
        top right patch will have the patch coordinate
        (num_patches_x-1, num_patches_y-1).

    Methods:
        assemble (da.Image): reassembles the patches to a new image
    """

    def __init__(self, img: da.Image, *args, **kwargs) -> None:
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
                "rel_overlap" (int): relative overlap of each patch (in relation to patch size)
                    in each direction of a rectangular patch; default value is 0.
        """

        # Instance of base image and relative_overlap
        self.base = img
        self.relative_overlap = kwargs.pop("rel_overlap", 0.0)

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
        patch_width_metric = self.base.width / self.num_patches_x
        patch_height_metric = self.base.height / self.num_patches_y
        # ... and in numbers of pixles
        patch_width_pixels = cast(
            int, self.base.coordinatesystem.lengthToPixels(patch_width_metric, "x")
        )
        patch_height_pixels = cast(
            int, self.base.coordinatesystem.lengthToPixels(patch_height_metric, "y")
        )

        # Determine the overal in metric lengths
        overlap_width_metric = self.relative_overlap * patch_width_metric
        overlap_height_metric = self.relative_overlap * patch_height_metric

        # Convert the overlap from metric to numbers of pixels
        overlap_width_pixels = cast(
            int, self.base.coordinatesystem.lengthToPixels(overlap_width_metric, "x")
        )
        overlap_height_pixels = cast(
            int, self.base.coordinatesystem.lengthToPixels(overlap_height_metric, "y")
        )

        # Some abbreviation for better overview
        nh = self.base.num_pixels_height
        pw = patch_width_pixels
        ph = patch_height_pixels
        ow = overlap_width_pixels
        oh = overlap_height_pixels
        cw = ceil(overlap_width_pixels / 2)
        ch = ceil(overlap_height_pixels / 2)
        off_w = 0 if cw == ow / 2.0 else 1
        off_h = 0 if ch == oh / 2.0 else 1

        self.nh = nh
        self.pw = pw
        self.ph = ph
        self.ow = ow
        self.oh = oh
        self.cw = cw
        self.ch = ch
        self.off_w = off_w
        self.off_h = off_h

        # Define pixel-based ROIs - with overlap
        # NOTE: While patches use the Cartesian indexing, the ROIs address
        # images with conventional matrix indexing, i.e., (x,y) vs (row,col).
        # Thus, the standard conversion has to be applied: Flip arguments and flip
        # the orientation of the vertical component.
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

        # Define relative pixel-based ROIs corresponding to the area - without overlap.
        self.relative_rois_without_overlap: list[list[tuple]] = [
            [
                (
                    slice(0, ph) if j == self.num_patches_y else slice(oh, ph + oh),
                    slice(0, pw) if i == 0 else slice(ow, pw + ow),
                )
                for j in range(self.num_patches_y)
            ]
            for i in range(self.num_patches_x)
        ]

        # Extract images with overlap
        self.images = [
            [
                da.extractROIPixel(self.base, self.rois[i][j])
                for j in range(self.num_patches_y)
            ]
            for i in range(self.num_patches_x)
        ]

        # Store centers (x,y) of each patch in global Cartesian coordinates and metric units
        self.global_centers_cartesian = np.array(
            [
                [
                    self.base.origo
                    + np.array(
                        [
                            (i + 0.5) * patch_width_metric,
                            (j + 0.5) * patch_height_metric,
                        ]
                    )
                    for j in range(self.num_patches_y)
                ]
                for i in range(self.num_patches_x)
            ]
        )

        # Convert coordinates of patch centers to pixels - using the matrix indexing
        self.global_centers_reverse_matrix = np.array(
            [
                [
                    np.flip(
                        self.base.coordinatesystem.coordinateToPixel(
                            self.global_centers_cartesian[i, self.num_patches_y - 1 - j]
                        )
                    )
                    for i in range(self.num_patches_x)
                ]
                for j in range(self.num_patches_y)
            ],
            dtype=int,
        )

        # Store corners of all patches in various formats, but keep the order:
        # top_left, bottom_left, bottom_right, top_right

        # Corners in global Cartesian coordinates, using metric units
        self.global_corners_cartesian = np.array(
            [
                [
                    np.array(
                        [
                            [i * patch_width_metric, (j + 1) * patch_height_metric],
                            [i * patch_width_metric, j * patch_height_metric],
                            [(i + 1) * patch_width_metric, j * patch_height_metric],
                            [
                                (i + 1) * patch_width_metric,
                                (j + 1) * patch_height_metric,
                            ],
                        ]
                    )
                    + self.base.origo[np.newaxis, :]
                    for j in range(self.num_patches_y)
                ]
                for i in range(self.num_patches_x)
            ]
        )

        # Corners in global pixel coordinates, using reverse matrix indexing
        self.global_corners_reverse_matrix = np.array(
            [
                [
                    np.array(
                        [
                            [i * pw, nh - (j + 1) * ph],
                            [i * pw, nh - j * ph],
                            [(i + 1) * pw, nh - j * ph],
                            [(i + 1) * pw, nh - (j + 1) * ph],
                        ]
                    )
                    for j in range(self.num_patches_y)
                ]
                for i in range(self.num_patches_x)
            ],
            dtype=int,
        )

        # Corners in local pixel coordinates, using reverse matrix indexing
        self.local_corners_reverse_matrix = np.array(
            [
                [
                    np.array([[0, 0], [0, ph], [pw, ph], [pw, 0]])
                    for j in range(self.num_patches_y)
                ]
                for i in range(self.num_patches_x)
            ],
            dtype=int,
        )

        # Define flag (will be turned on when running _prepare_weights)
        self.weights_defined = False

    # TODO the interpolation may be not required after all. Keep it for now, but it may
    # disappear afterall.

    def _prepare_weights(self):
        """
        Auxiliary setup method for defining weights to be used in blend_and_assemble.
        """

        # Fetch some abbreviations
        pw = self.pw
        ph = self.ph
        ow = self.ow
        oh = self.oh
        cw = self.cw
        ch = self.ch
        off_w = self.off_w
        off_h = self.off_h

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

        if pw - 2 * cw <= 0:
            raise ValueError("Overlap chosen to large")

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
                np.ones(self.base.num_pixels_width - marked_width, dtype=float),
            )
        )

        # Analogously, define the weighting in y-direction.
        # NOTE: The weight has to be defined consistently with the conventional matrix
        # indexing of images, i.e., the first pixel lies at the top.
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
                np.ones(self.base.num_pixels_height - marked_height, dtype=float),
                np.linspace(1, 0, oh),
                np.zeros(ch, dtype=float),
            )
        )

        # Mark flag that weights are defined.
        self.weights_defined = True

    def position(self, i: int, j: int) -> tuple[str, str]:
        """
        Determine positioning of patch wrt. boundary or internal patches
        in both x- and y-direction.

        Args:
            i (int): patch coordinate in x-direction
            j (int): patch coordinate in y-direction

        NOTE: The patch coordinates employ the Cartesian indexing, i.e., (x,y).

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

    def __call__(self, i: int, j: int) -> da.Image:
        """
        Return patch with Cartesian patch coordinates (i,j).

        Args:
            i (int): x-coordinate of the patch
            j (int): y-coordinate of the patch

        Returns:
            daria.Image: image of the patch
        """
        return self.images[i][j]

    def set_image(self, img: np.ndarray, i: int, j: int) -> None:
        """
        Update the image of a patch.

        Args:
            img (np.ndarray): image array
            i (int): x-coordinate of the patch
            j (int): y-coordinate of the patch
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
        # to assemble the patches 'row' by 'row' (here row is not
        # meant as for images, as patches use Cartesian coordinates).
        assembled_img = np.zeros(
            (0, *self.base.img.shape[1:]), dtype=self.base.img.dtype
        )

        # Create "image-strips" that are assembled by concatenation
        for j in range(self.num_patches_y):

            # Initialize the row with y coordinate j of the final image
            # with the first patch image.
            rel_roi = self.relative_rois_without_overlap[0][j]
            assembled_y_j = self.images[0][j].img[rel_roi]

            # And assemble the remainder of the row by concatenation
            # over the patches in x-direction with same y coordinate (=j).
            for i in range(1, self.num_patches_x):
                rel_roi = self.relative_rois_without_overlap[i][j]
                assembled_y_j = np.hstack(
                    (assembled_y_j, self.images[i][j].img[rel_roi])
                )

            # Concatenate the row and the current image
            assembled_img = np.vstack((assembled_y_j, assembled_img))

        # Make sure that the resulting image has the same resolution
        assert assembled_img.shape == self.base.img.shape

        # Define resulting daria image
        da_assembled_img = da.Image(
            img=assembled_img,
            origo=self.base.origo,
            width=self.base.width,
            height=self.base.height,
            color_space=self.base.colorspace,
        )

        # Update the base image if required
        if update_img:
            self.base = da_assembled_img.copy()

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
        # Require weights. Define if needed.
        if not self.weights_defined:
            self._prepare_weights()

        # The procedure is as follows. The image is reassembled, row by row.
        # Each row is reconstructed by concatenation. Overlapping regions
        # have to handled separately by a weighted sum (convex combination).
        # For this, the above defined partition of unity will be used.
        # Then, rows are combined in a similar manner, but now entire rows are
        # concatenated, and added using the partition of unity.

        # Allocate memory for resassembled image
        assembled_img = np.zeros_like(self.base.img, dtype=float)

        # Loop over patches
        for j in range(self.num_patches_y):

            # Allocate memory for row  with y-coordinate j.
            shape = [self.images[0][j].num_pixels_height, *self.base.img.shape[1:]]
            assembled_y_j = np.zeros(tuple(shape), dtype=float)

            # Assemble the row with y-coordinate j by a suitable weighted combination
            # of the patches in row j
            for i in range(self.num_patches_x):

                # Determine the active pixel range
                roi = self.rois[i][j]
                roi_x = roi[1]

                # Fetch patch, and convert to float
                img_i_j = skimage.img_as_float(self.images[i][j].img)

                # Fetch weight and convert to tensor
                weight_i_j = self.weight_x[self.position(i, j)[0]]
                # Convert to tensor
                weight_i_j = weight_i_j.reshape(1, np.size(weight_i_j), 1)

                # Add weighted patch at the active pixel range
                assembled_y_j[:, roi_x] += np.multiply(img_i_j, weight_i_j)

            # Anologous procedure, but now on row-level and not single-patch level.

            # Determine active pixel range. NOTE: roi[0] still contains the relevant
            # pixel range, relevant for addressing the base image.
            roi_y = roi[0]

            # Fetch weight
            weight_j = self.weight_y[self.position(i, j)[1]]
            # Convert to tensor
            weight_j = weight_j.reshape(np.size(weight_j), 1, 1)

            # Add weighted row at the active pixel range
            assembled_img[roi_y, :] += np.multiply(assembled_y_j, weight_j)

        # Make sure the newly assembled image is compatible with the original base image
        assert assembled_img.shape == self.base.img.shape

        # Convert final image to uint8 format
        if self.base.original_dtype == np.uint8:
            assembled_img = skimage.img_as_ubyte(assembled_img)
        elif self.base.original_dtype == np.uint16:
            assembled_img = skimage.img_as_uint(assembled_img)

        # Define resulting daria image
        da_assembled_img = da.Image(
            img=assembled_img,
            origo=self.base.origo,
            width=self.base.width,
            height=self.base.height,
            color_space=self.base.colorspace,
        )

        # Update the base image if required
        if update_img:
            self.base = da_assembled_img.copy()

        return da_assembled_img
