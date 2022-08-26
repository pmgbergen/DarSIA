from math import sqrt

import numpy as np

import daria as da


class Patches:
    def __init__(self, im: da.Image, *args):
        self.baseImg = im

        if len(args) == 1:
            self.num_patches_x = int(sqrt(args[0]))
            self.num_patches_y = int(sqrt(args[0]))

        if len(args) == 2:
            self.num_patches_x = int(sqrt(args[0]))
            self.num_patches_y = int(sqrt(args[1]))

        width = im.width / self.num_patches_x
        height = im.height / self.num_patches_y

        self.images = np.empty(
            shape=(self.num_patches_x, self.num_patches_y), dtype=da.Image
        )

        for j in range(self.num_patches_y):
            for i in range(self.num_patches_x):
                im_tmp = da.extractROI(
                    im, [i * width, (i + 1) * width], [j * height, (j + 1) * height]
                )
                self.images[i, j] = im_tmp

    def assemble(self):
        im_tmp_x = self.images[0, 0].img
        print(im_tmp_x)
        for i in range(self.num_patches_x - 1):
            im_tmp_x = np.c_["1,0,0", im_tmp_x, self.images[i + 1, 0].img]
        im_tmp = im_tmp_x
        for j in range(self.num_patches_y - 1):
            im_tmp_x = self.images[0, j + 1].img
            for i in range(self.num_patches_x - 1):
                im_tmp_x = np.c_["1,0,0", im_tmp_x, self.images[i + 1, j + 1].img]
            im_tmp = np.r_["0,1,0", im_tmp_x, im_tmp]
        full_img = da.Image(
            im_tmp,
            origo=self.baseImg.origo,
            width=self.baseImg.width,
            height=self.baseImg.height,
        )
        return full_img


# np.c_

# og np.r_
