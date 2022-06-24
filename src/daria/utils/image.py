# Image class. Should contain information about origo, size, and actual image data at the very least.
import cv2
import numpy as np


class Image:

    # Image should as of now be given in BGR, which is the openCV standard.
    def __init__(self, img: np.array, origo=[0, 0], width=1, height=1, depth=0, dim=2):
        self.img = img
        self.origo = origo
        self.width = width
        self.height = height
        self.shape = self.img.shape
        self.dim = dim
        self.dx = self.width / self.shape[0]
        self.dy = self.height / self.shape[1]
        if self.dim == 3:
            self.depth = depth
            self.dz = self.depth / self.shape[2]

    # Figure out how to have two declarators.

    # When in the presence of internet, figure out how to access the name of the class and feed it to the save file.
    def write(self, name="im", path="images/modified/", format=".jpg"):
        cv2.imwrite(path + name + format, self.img)
        print(path + name + format)
