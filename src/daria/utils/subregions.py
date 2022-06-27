from daria import Image
import numpy as np

# Extracts region based on physical coordinates
def extractROI(img: Image, x: list, y: list) -> Image:
    im = img.img
    x_pix = [
        round((x[0] - img.origo[0]) / img.dx),
        round((x[1] - img.origo[0]) / img.dx),
    ]
    y_pix = [
        round((y[0] - img.origo[1]) / img.dy),
        round((y[1] - img.origo[1]) / img.dy),
    ]
    im = im[
        (img.shape[0] - y_pix[1]) : (img.shape[0] - y_pix[0]),
        x_pix[0] : x_pix[1],
    ]
    o = [x[0], y[0]]
    w = x[1] - x[0]
    h = y[1] - y[0]
    return Image(im, o, w, h)


# Extracts region based on pixels
def extractROIPixel(img: Image, x: list, y: list) -> Image:
    im = img.img
    im = im[
        (img.shape[0] - y[1]) : (img.shape[0] - y[0]),
        x[0] : x[1],
    ]
    o = [img.origo[0] + x[0] * img.dx, img.origo[1] + y[0] * img.dy]
    w = ((x[1] - x[0]) / img.shape[0]) * img.width
    h = ((y[1] - y[0]) / img.shape[1]) * img.height

    return Image(im, o, w, h)
