from __future__ import annotations


from daria import Image
import numpy as np


# Functions for forward and backward differences of images in greyscale, with 0 Neumann boundary conditions
def backward_diff_x(im: np.ndarray) -> np.ndarray:
    return np.diff(im, axis=1, append = im[:,-1:])

def forward_diff_x(im):
    return np.diff(im, axis=1, prepend = im[:,:1])

def backward_diff_y(im):
    return np.diff(im, axis=0, append = im[-1:,:])

def forward_diff_y(im):
    return np.diff(im, axis=0, prepend = im[:1,:])

def laplace_x(im):
    return 0.5 * (
        forward_diff_x(backward_diff_x(im)) + backward_diff_x(forward_diff_x(im))
    )

def laplace_y(im):
    return 0.5 * (
        forward_diff_y(backward_diff_y(im)) + backward_diff_y(forward_diff_y(im))
    )

def laplace(im):
    return laplace_x(im) + laplace_y(im)
