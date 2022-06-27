# Image class. Should contain information about origo, size, and actual image data at the very least.
import cv2
import numpy as np


class Image:

    # input img can either be a path to the image file or a numpy array.
    def __init__(self, img, origo=[0, 0], width=1, height=1, depth=0, dim=2):
        if type(img) == np.ndarray:
            self.img = img
        elif type(img) == str:
            self.img = cv2.imread(img)
        self.origo = origo
        self.width = width
        self.height = height
        self.shape = self.img.shape
        self.dim = dim
        self.dx = self.width / self.shape[1]
        self.dy = self.height / self.shape[0]
        if self.dim == 3:
            self.depth = depth
            self.dz = self.depth / self.shape[2]

    # Write image to file. Here, the BGR-format is used. Image path, name and format can be changed by passing them as strings to the method.
    def write(self, name="img", path="images/modified/", format=".jpg"):
        cv2.imwrite(path + name + format, self.img)
        print("Image saved as: " + path + name + format)
