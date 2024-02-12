from __future__ import annotations

from math import ceil

import numpy as np
import skimage

import darsia


class Patches:
    """
    Class for patched image.

    Contains an array of patches of a prescribed full image.

    Attributes:
        base (darsia.Image) = full darsia image
        num_patches (list) = list with number of patches in each dimension
        patches (np.ndarray)= array of patches of the original image

    NOTE: A standard matrix indexing is used to refer to specific patches, i.e.,
        The left, upper (front) patch will have the patch coordinate (0,0)
        ((0,0,0) in 3d).

    """

    def __init__(self, img: darsia.Image, num_patches: list[int], **kwargs) -> None:
        """
        Constructor for Patches class.

        Arguments:
            img (darsia.Image): image to be patched.
            num_patches (list of int): number of patches in each dimension, using the same
                convention on the order of element access as for 'img', incl. time access.
                If length of num_patches is equal to spatial dimension, the entire time slab
                is kept as one.
            **kwargs: optional keyword arguments:
                "rel_overlap" (int): relative overlap of each patch in spatial dimensions
                    (in relation to patch size) in each direction of a rectangular patch;
                    default value is 0.
                "abs_time_overlap" (int)" Absolute overlap of each patch in temporal dimension;
                    default value is 0.

        """

        # ! ---- Instance of base image, incl. relevant meta information
        self.base = img
        """Underlying base image."""

        # FIXME
        if self.base.space_dim == 3:
            raise NotImplementedError("3d patches are not tested yet!")
        if self.base.time_dim == 1:
            raise NotImplementedError(
                "Patches of space time images are not tested yet!"
            )

        # ! ---- Key properties for defining patches

        self.num_patches: list[int] = num_patches
        """Number of patches in row, col, (depth, time)."""

        self.num_active_spatial_axes = min(len(self.num_patches), self.base.space_dim)
        """Active dimenion of patching. Remaining dimensions will not be patched."""

        self.relative_space_overlap = kwargs.get(
            "rel_overlap", 0.0
        )  # TODO call rel_space_overlap.
        """Relative overlap in each spatial dimension."""

        self.absolute_time_overlap: int = kwargs.get("abs_time_overlap", 0)
        """Absolute overlap in temporal dimension."""

        # Deduct spatial dimensions of each patch without overlap.
        patch_dimensions_metric = [
            self.base.dimensions[i] / self.num_patches[i]
            for i in range(self.num_active_spatial_axes)
        ]

        indexing = self.base.indexing

        patch_dimensions_voxels = [
            self.base.coordinatesystem.num_voxels(
                length=patch_dimensions_metric[i],
                axis=darsia.to_cartesian_indexing(i, indexing),
            )
            for i in range(self.num_active_spatial_axes)
        ]

        # Determine the absolute overlap.
        overlap_metric = [
            self.relative_space_overlap * patch_dimensions_metric[i]
            for i in range(self.num_active_spatial_axes)
        ]

        overlap_voxels = [
            self.base.coordinatesystem.num_voxels(
                length=overlap_metric[i], axis=darsia.to_cartesian_indexing(i, indexing)
            )
            for i in range(self.num_active_spatial_axes)
        ]

        # ! ---- Some abbreviation for better overview
        nv = self.base.num_voxels
        pv = patch_dimensions_voxels
        ov = overlap_voxels
        cv = [ceil(overlap / 2) for overlap in overlap_voxels]
        off = [0 if c == o / 2.0 else 1 for c, o in zip(cv, ov)]

        self.nv = nv
        self.pv = pv
        self.ov = ov
        self.cv = cv
        self.off = off

        # NOTE: In the following, all patch access is using matrix indexing.

        # ! ---- Single patches

        # ROIs with overlap
        if self.base.space_dim == 2:

            self.rois: list[list[tuple]] = [
                [
                    (
                        slice(max(i * pv[0] - ov[0], 0), (i + 1) * pv[0] + ov[0]),
                        slice(max(j * pv[1] - ov[1], 0), (j + 1) * pv[1] + ov[1]),
                    )
                    for j in range(self.num_patches[1])
                ]
                for i in range(self.num_patches[0])
            ]

        elif self.base.space_dim == 3:
            raise NotImplementedError

        # Relative pixel-based ROIs corresponding to the area - without overlap.
        if self.base.space_dim == 2:

            self.relative_rois_without_overlap: list[list[tuple]] = [
                [
                    (
                        slice(0, pv[0]) if i == 0 else slice(ov[0], pv[0] + ov[0]),
                        slice(0, pv[1]) if j == 0 else slice(ov[1], pv[1] + ov[1]),
                    )
                    for j in range(self.num_patches[1])
                ]
                for i in range(self.num_patches[0])
            ]

        elif self.base.space_dim == 3:
            raise NotImplementedError

        # Extract images with overlap
        if self.base.space_dim == 2:

            self.patches = [
                [
                    self.base.subregion(self.rois[i][j])
                    for j in range(self.num_patches[1])
                ]
                for i in range(self.num_patches[0])
            ]

        elif self.base.space_dim == 3:
            raise NotImplementedError

        # ! ---- Coordinates of patch centers

        # Store centers of each patch in global Cartesian coordinates and metric units
        if self.base.space_dim == 2:
            self.global_centers_cartesian = np.array(
                [
                    [
                        self.base.origin
                        + np.array(
                            [
                                (j + 0.5) * patch_dimensions_metric[1],
                                -(i + 0.5) * patch_dimensions_metric[0],
                            ]
                        )
                        for j in range(self.num_patches[1])
                    ]
                    for i in range(self.num_patches[0])
                ]
            )
        elif self.base.space_dim == 3:
            raise NotImplementedError

        # Coordinates of patch centers in voxels, using matrix indexing
        if self.base.space_dim == 2:

            self.global_centers_voxels = np.array(
                [
                    [
                        self.base.coordinatesystem.voxel(
                            self.global_centers_cartesian[i, j]
                        )
                        for j in range(self.num_patches[1])
                    ]
                    for i in range(self.num_patches[0])
                ],
                dtype=int,
            )

        elif self.base.space_dim == 3:
            raise NotImplementedError

        # ! ---- Coordinates of patch corners

        # Store corners of all patches in various formats, but keep the order:
        # top_left, bottom_left, bottom_right, top_right

        # Corners in global Cartesian coordinates, using metric units
        if self.base.space_dim == 2:

            self.global_corners_cartesian = np.array(
                [
                    [
                        np.array(
                            [
                                [
                                    j * patch_dimensions_metric[1],
                                    -i * patch_dimensions_metric[0],
                                ],
                                [
                                    j * patch_dimensions_metric[1],
                                    -(i + 1) * patch_dimensions_metric[0],
                                ],
                                [
                                    (j + 1) * patch_dimensions_metric[1],
                                    -(i + 1) * patch_dimensions_metric[0],
                                ],
                                [
                                    (j + 1) * patch_dimensions_metric[1],
                                    -i * patch_dimensions_metric[0],
                                ],
                            ]
                        )
                        + self.base.origin[np.newaxis, :]
                        for j in range(self.num_patches[1])
                    ]
                    for i in range(self.num_patches[0])
                ]
            )

        elif self.base.space_dim == 3:
            raise NotImplementedError

        # Corners in global voxel coordinates.
        if self.base.space_dim == 2:
            self.global_corners_voxels = np.array(
                [
                    [
                        np.array(
                            [
                                [i * pv[0], j * pv[1]],
                                [min(nv[0], (i + 1) * pv[0]), j * pv[1]],
                                [
                                    min(nv[0], (i + 1) * pv[0]),
                                    min(nv[1], (j + 1) * pv[1]),
                                ],
                                [i * pv[0], min(nv[1], (j + 1) * pv[1])],
                            ]
                        )
                        for j in range(self.num_patches[1])
                    ]
                    for i in range(self.num_patches[0])
                ],
                dtype=int,
            )
        elif self.base.space_dim == 3:
            raise NotImplementedError

        # Corners in local voxel coordinates.
        if self.base.space_dim == 2:

            self.local_corners_voxels = np.array(
                [
                    [
                        np.array(
                            [
                                [0, 0],
                                [min(nv[0], (i + 1) * pv[0]) - i * pv[0], 0],
                                [
                                    min(nv[0], (i + 1) * pv[0]) - i * pv[0],
                                    min(nv[1], (j + 1) * pv[1]) - j * pv[1],
                                ],
                                [0, min(nv[1], (j + 1) * pv[1]) - j * pv[1]],
                            ]
                        )
                        for j in range(self.num_patches[1])
                    ]
                    for i in range(self.num_patches[0])
                ],
                dtype=int,
            )

        elif self.base.space_dim == 3:
            raise NotImplementedError

        # Define flag (will be turned on when running _prepare_weights)
        self.weights_defined = False

    # ! ---- I/O

    def __call__(self, *args) -> darsia.Image:
        """
        Return patch with index (i,j).

        Args:
            args (tuple of int): index of a patch

        Returns:
            darsia.Image: image of the indexed patch

        """
        assert len(args) == self.base.space_dim

        if self.base.space_dim == 2:

            i, j = args
            return self.patches[i][j]

        elif self.base.space_dim == 3:
            raise NotImplementedError

    def set_image(self, img: np.ndarray, *args) -> None:
        """
        Update the image of a patch.

        Args:
            img (np.ndarray): image array
            args (tuple of int): index of the patch

        """
        assert len(args) == self.num_active_spatial_axes

        if self.base.space_dim == 2:

            i, j = args
            assert self.patches[i][j].img.shape == img.shape
            self.patches[i][j].img = img.copy()

        elif self.base.space_dim == 3:
            raise NotImplementedError

    # ! ---- (Re-)assembly of patched image.

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
            + (self.num_patches[0] - 2) * (pw - 2 * cw + ow + off_w)
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
            + (self.num_patches[1] - 2) * (ph - 2 * ch + oh + off_h)
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
        # TODO is this used? rm?
        # Determine horizontal position (x-direction)
        if i == 0:
            horizontal_position: str = "left"
        elif i == self.num_patches[0] - 1:
            horizontal_position = "right"
        else:
            horizontal_position = "internal"

        # Determine vertical position (y-direction)
        if j == 0:
            vertical_position: str = "bottom"
        elif j == self.num_patches[1] - 1:
            vertical_position = "top"
        else:
            vertical_position = "internal"

        return horizontal_position, vertical_position

    def assemble(self, update_img: bool = False) -> darsia.Image:
        """
        Reassembles without taking into account the overlap.

        Args:
            update_img (bool): flag controlling whether the base image will be updated
                with the assembled image; default set to False

        Returns:
            darsia.image: assembled image as darsia image
        """

        # TODO naturally extends to 3d?

        # Initialize empty row of the final image. It will be used
        # to assemble the patches 'row' by 'row' (here row is not
        # meant as for images, as patches use Cartesian coordinates).
        assembled_img = np.zeros(
            (0, *self.base.img.shape[1:]), dtype=self.base.img.dtype
        )
        print(assembled_img)

        # Create "image-strips" that are assembled by concatenation
        for j in range(self.num_patches[0]):

            # Initialize the row with y coordinate j of the final image
            # with the first patch image.
            rel_roi = self.relative_rois_without_overlap[j][0]
            assembled_y_j = self.patches[j][0].img[rel_roi]
            # And assemble the remainder of the row by concatenation
            # over the patches in x-direction with same y coordinate (=j).
            for i in range(1, self.num_patches[1]):
                rel_roi = self.relative_rois_without_overlap[j][i]
                assembled_y_j = np.hstack(
                    (assembled_y_j, self.patches[j][i].img[rel_roi])
                )

            # Concatenate the row and the current image
            assembled_img = np.vstack((assembled_img, assembled_y_j))

        # Make sure that the resulting image has the same resolution
        assert assembled_img.shape == self.base.img.shape

        # Define resulting darsia image
        da_assembled_img = darsia.Image(
            img=assembled_img,
            metadata=self.base.metadata(),
        )

        # Update the base image if required
        if update_img:
            self.base = da_assembled_img.copy()

        return da_assembled_img

    def blend_and_assemble(self, update_img: bool = False) -> darsia.Image:
        """
        Reassembles taking into account the overlap as well.
        On the overlap, a convex combination is used for
        smooth blending.

        Args:
            update_img (bool): flag controlling whether the base image will be updated
                with the assembled image; default set to False

        Returns:
            darsia.Image: assembled image as darsia image

        """

        # TODO naturally extends to 3d?

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
        for j in range(self.num_patches[1]):

            # Allocate memory for row  with y-coordinate j.
            shape = [self.patches[0][j].num_pixels_height, *self.base.img.shape[1:]]
            assembled_y_j = np.zeros(tuple(shape), dtype=float)

            # Assemble the row with y-coordinate j by a suitable weighted combination
            # of the patches in row j
            for i in range(self.num_patches[0]):

                # Determine the active pixel range
                roi = self.rois[i][j]
                roi_x = roi[1]

                # Fetch patch, and convert to float
                img_i_j = skimage.img_as_float(self.patches[i][j].img)

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

        # Define resulting darsia image
        da_assembled_img = darsia.Image(
            img=assembled_img,
            **self.base.metadata(),
        )

        # Update the base image if required
        if update_img:
            self.base = da_assembled_img.copy()

        return da_assembled_img
