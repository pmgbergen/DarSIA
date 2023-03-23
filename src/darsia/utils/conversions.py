from __future__ import annotations

import cv2
import numpy as np

import darsia as da


def BGR2RGB(img: da.Image) -> da.Image:
    """Conversion between opencv (cv2) which has BGR-formatting and scikit-image (skimage)
    which has RGB. The same command works in both directions.

    Arguments:
        img (np.ndarray): input image

    Returns:
        np.ndarray: converted image.
    """

    img_new = img.copy()
    img_new.img = cv2.cvtColor(img_new.img, cv2.COLOR_BGR2RGB)
    return img_new


def BGR2GRAY(img: da.Image) -> da.Image:
    """Creates a grayscale darsia Image from a BGR ones

    Arguments:
        img (da.Image): input image

    Returns:
        da.Image: converged image
    """
    img_gray = img.copy()
    img_gray.img = cv2.cvtColor(img_gray.img, cv2.COLOR_BGR2GRAY)
    return img_gray


def BGR2RED(img: da.Image) -> da.Image:
    """Creates a redscale darsia Image from a BGR ones

    Arguments:
        img (da.Image): input image

    Returns:
        da.Image: converged image
    """
    img_red = img.copy()
    if img.colorspace == "BGR":
        img_red.img = img_red.img[:, :, 2]
    elif img.colorspace == "RGB":
        img_red.img = img_red.img[:, :, 0]
    return img_red


def BGR2GREEN(img: da.Image) -> da.Image:
    """Creates a greenscale darsia Image from a BGR one

    Arguments:
        img (da.Image): input image

    Returns:
        da.Image: converged image
    """
    img_green = img.copy()
    if img.colorspace == "BGR":
        img_green.img = img_green.img[:, :, 1]
    elif img.colorspace == "RGB":
        img_green.img = img_green.img[:, :, 1]
    return img_green


def BGR2BLUE(img: da.Image) -> da.Image:
    """Creates a bluescale darsia Image from a BGR ones

    Arguments:
        img (da.Image): input image

    Returns:
        da.Image: converged image
    """
    img_blue = img.copy()
    if img.colorspace == "BGR":
        img_blue.img = img_blue.img[:, :, 0]
    elif img.colorspace == "RGB":
        img_blue.img = img_blue.img[:, :, 2]
    return img_blue
