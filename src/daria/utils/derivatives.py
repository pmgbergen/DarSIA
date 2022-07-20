from daria import Image
import numpy as np


# Functions for forward and backward differences of images in greyscale, with 0 Neumann boundary conditions
def backward_diff_x(im):
    img = im.astype(float)
    return np.c_[img, img[:, -1]][:, 1:] - img


def forward_diff_x(im):
    img = im.astype(float)
    return img - np.c_[img[:, 0], img][:, :-1]


def laplace_x(im):
    return 0.5 * (
        forward_diff_x(backward_diff_x(im)) + backward_diff_x(forward_diff_x(im))
    )


def backward_diff_y(im):
    img = im.astype(float)
    return np.r_[img, [img[-1, :]]][1:, :] - img


def forward_diff_y(im):
    img = im.astype(float)
    return img - np.r_[[img[0, :]], img][:-1, :]


def laplace_y(im):
    return 0.5 * (
        forward_diff_y(backward_diff_y(im)) + backward_diff_y(forward_diff_y(im))
    )


def laplace(im):
    return laplace_x(im) + laplace_y(im)
