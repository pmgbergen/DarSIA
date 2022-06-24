from daria import Image
import numpy as np


# Need to take a lot of care here. The images are oriented from top to bottom, and then left to right.
# This might be the golden standard in computer science, but I still think that we should make everything intuitive
#  for the physisists and mathematicians and try to conform to classical x,y(,z) coordinate systems.

# Should extract region based on physical coordinates
def extractSubRegion(img: Image) -> Image:
    pass


# Should extract region based on pixels
def extractSubRegionByPixel(img: Image, x: list, y: list) -> Image:
    im = img.img
    im = im[(img.shape[0] - y[1]) : (img.shape[0] - y[0]), x[0] : x[1]]
    o = [img.origo[0] + x[0] * img.dx, img.origo[1] + y[0] * img.dy]
    w = ((x[1] - x[0]) / img.shape[0]) * img.width
    h = ((y[1] - y[0]) / img.shape[1]) * img.height

    return Image(im, o, w, h)
