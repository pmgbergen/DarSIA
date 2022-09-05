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

        arguments:
            im (da.Image): the Image that is to be patched
            *args: information regarding how many patches we should have:
                        eiher one can give the amount of patches as one argument
                        (needs to be a square number),
                        or one can give the number of patches in x, and y direction
                        as two separate arguments, respectively.
            **kwargs: checks for patch_overlay argument. If it is provided an overlay
                      of length = patch_overlay is provided to the patches.

        """

        # Instance of base image and patch_overlay
        self.baseImg = im
        if "patch_overlay" in kwargs:
            self.patch_overlay = kwargs["patch_overlay"]
        else:
            self.patch_overlay = 0

        # Define number of patches in each direction
        if len(args) == 1:
            self.num_patches_x = int(sqrt(args[0]))
            self.num_patches_y = int(sqrt(args[0]))

        if len(args) == 2:
            self.num_patches_x = int(sqrt(args[0]))
            self.num_patches_y = int(sqrt(args[1]))

        # Define width and height of each patch (before patch overlay is applied)
        width = im.width / self.num_patches_x
        height = im.height / self.num_patches_y

        # Create empty array where the patches will be contained.
        self.images = np.empty(
            shape=(self.num_patches_x, self.num_patches_y), dtype=da.Image
        )

        # Create patches. The overlay has been hardcoded, and therefore it looks a bit messy.
        # Corners
        self.images[0, 0] = da.extractROI(
            self.baseImg,
            [0, width + self.patch_overlay],
            [0, height + self.patch_overlay],
        )
        self.images[self.num_patches_x - 1, self.num_patches_y - 1] = da.extractROI(
            self.baseImg,
            [
                (self.num_patches_x - 1) * width - self.patch_overlay,
                self.num_patches_x * width,
            ],
            [
                (self.num_patches_y - 1) * height - self.patch_overlay,
                self.num_patches_y * height,
            ],
        )
        self.images[0, self.num_patches_y - 1] = da.extractROI(
            self.baseImg,
            [0, width + self.patch_overlay],
            [
                (self.num_patches_y - 1) * height - self.patch_overlay,
                self.num_patches_y * height,
            ],
        )
        self.images[self.num_patches_x - 1, 0] = da.extractROI(
            self.baseImg,
            [
                (self.num_patches_x - 1) * width - self.patch_overlay,
                self.num_patches_x * width,
            ],
            [0, height + self.patch_overlay],
        )

        # Top and botton patches
        for i in range(1, self.num_patches_x - 1):
            self.images[i, 0] = da.extractROI(
                self.baseImg,
                [i * width - self.patch_overlay, (i + 1) * width + self.patch_overlay],
                [0, height + self.patch_overlay],
            )
            self.images[i, self.num_patches_y - 1] = da.extractROI(
                self.baseImg,
                [i * width - self.patch_overlay, (i + 1) * width + self.patch_overlay],
                [
                    (self.num_patches_y - 1) * height - self.patch_overlay,
                    height * self.num_patches_y,
                ],
            )

        # Left and right patches
        for j in range(1, self.num_patches_y - 1):
            self.images[0, j] = da.extractROI(
                self.baseImg,
                [0, width + self.patch_overlay],
                [
                    j * height - self.patch_overlay,
                    (j + 1) * height + self.patch_overlay,
                ],
            )
            self.images[self.num_patches_x - 1, j] = da.extractROI(
                self.baseImg,
                [
                    width * (self.num_patches_x - 1) - self.patch_overlay,
                    width * self.num_patches_x,
                ],
                [
                    j * height - self.patch_overlay,
                    (j + 1) * height + self.patch_overlay,
                ],
            )

        # Internal patches
        for j in range(1, self.num_patches_y - 1):
            for i in range(1, self.num_patches_x - 1):
                self.images[i, j] = da.extractROI(
                    self.baseImg,
                    [
                        i * width - self.patch_overlay,
                        (i + 1) * width + self.patch_overlay,
                    ],
                    [
                        j * height - self.patch_overlay,
                        (j + 1) * height + self.patch_overlay,
                    ],
                )

    def assemble(self) -> da.Image:
        """
        Reassembles the patches into a new daria image.
        The method will run regardless of whether patch overlay has
        been applied or not, but it is only functioning as intended
        without patch overlay.
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
