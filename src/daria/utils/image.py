# Image class. Should contain information about origo, size, and actual image data at the very least.
import cv2
import numpy as np


class Image:

    # input img can either be a path to the image file or a numpy array.
    def __init__(
        self,
        img,
        origo=[0, 0],
        width=1,
        height=1,
        depth=0,
        dim=2,
        read_metadata_from_file=False,
        metadata_path="-",
    ):
        if type(img) == np.ndarray:
            self.img = img
        elif type(img) == str:
            self.img = cv2.imread(img)
            self.imgpath = img
        else:
            raise Exception(
                "Invalid image data. Please provide either a path to an image or an image array."
            )
        if read_metadata_from_file:
            self.create_metadata_from_file(metadata_path)
        else:
            self.origo = origo
            self.width = width
            self.height = height
            self.dim = dim
        self.shape = self.img.shape
        self.dx = self.width / self.shape[1]
        self.dy = self.height / self.shape[0]
        if self.dim == 3:
            self.depth = depth
            self.dz = self.depth / self.shape[2]

    # There might be a cleaner way to do this. Then again, it works.
    def create_metadata_from_file(self, path):
        if path == "-":
            pl = self.imgpath.split("/")
            name = pl[2].split(".")[0]
            path = pl[0] + "/metadata/" + name + ".txt"
        f = open(path, "r")
        md_dict = {}
        for line in f:
            key, value = line.split(":")
            md_dict[key] = value
        # The origo numbers are hardcoded, might be a better solution
        origo_nums = md_dict["Origo"].replace("[", "").replace("]", "").split(",")
        self.origo = [float(origo_nums[0]), float(origo_nums[1])]
        self.width = float(md_dict["Width"])
        self.height = float(md_dict["Height"])
        self.dim = float(md_dict["Dimension"])

    # Write image to file. Here, the BGR-format is used. Image path, name and format can be changed by passing them as strings to the method.
    def write(
        self,
        name="img",
        path="images/modified/",
        format=".jpg",
        save_metadata=True,
        metadata_path="images/metadata/",
    ):
        cv2.imwrite(path + name + format, self.img)
        print("Image saved as: " + path + name + format)
        if save_metadata:
            f = open(metadata_path + name + ".txt", "w")
            f.write("Origo: " + str(self.origo) + "\n")
            f.write("Width: " + str(self.width) + "\n")
            f.write("Height: " + str(self.height) + "\n")
            f.write("Dimension: " + str(self.dim) + "\n")
            f.close()
