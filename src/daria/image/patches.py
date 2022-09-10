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
        self.patch_overlap = kwargs.pop("patch_overlap", 0)

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
        The method will run regardless of whether patch overlap has
        been applied or not, but it is only functioning as intended
        without patch overlap.
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
            im_tmp_x = self.images[0, j + 1].img
            for i in range(self.num_patches_x - 1):
                im_tmp_x = np.c_["1,0,0", im_tmp_x, self.images[i + 1, j + 1].img]
            im_tmp = np.r_["0,1,0", im_tmp_x, im_tmp]

        # Make the daria image from the array
        full_img = da.Image(
            im_tmp,
            origo=self.baseImg.origo,
            width=self.baseImg.width,
            height=self.baseImg.height,
        )
        return full_img
