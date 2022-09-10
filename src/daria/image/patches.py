from math import sqrt

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

        # Define width and height of each patch (before patch overlap is applied)
        patch_width = im.width / self.num_patches_x
        patch_height = im.height / self.num_patches_y

        # Extract patches - use Cartesian sense for ordering patches, i.e., e.g.,
        # lower left patch is stored under [0][0].
        self.images = [
            [
                da.extractROI(
                    self.baseImg,
                    [
                        max(0, i * patch_width - self.patch_overlap),
                        min(im.width, (i + 1) * patch_width + self.patch_overlap),
                    ],
                    [
                        max(0, j * patch_height - self.patch_overlap),
                        min(im.height, (j + 1) * patch_height + self.patch_overlap),
                    ],
                )
                for j in range(self.num_patches_y)
            ]
            for i in range(self.num_patches_x)
        ]

    def __call__(self, i: int, j: int) -> np.ndarray:
        """Return patch (j,i) of the base image."""
        return self.images[i][j]

    def assemble(self) -> da.Image:
        """
        Reassembles the patches into a new daria image.
        The method will run regardless of whether patch overlap has
        been applied or not, but it is only functioning as intended
        without patch overlap.
        """

        # Create temporary image-array and add the lower left-corner patch to it
        im_tmp_x = self.images[0, 0].img

        # Add patches along the bottom to the temporary image to create the
        # lower part of the image
        for i in range(self.num_patches_x - 1):
            im_tmp_x = np.c_["1,0,0", im_tmp_x, self.images[i + 1, 0].img]

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
