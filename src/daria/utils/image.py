# Image class. Should contain information about origo, size, and actual image data at the very least.
import cv2
import numpy as np

# USE **kwargs in contructor

# Base image class. Contains the actual image, as well as base properties such as position in global coordinates, width and height.
# One can either pass in metadata (origo, width and height, amongst other entities) by passing them directly to the constructor
# or through a metadata-file (default is to pass metadata diretly to the constructor, and if metadatafile is requires the "read_metadata_from_file"
# variable has to be passed as True).
# Image can either be passed in as numpy array or a path to a file (this is automatically detected through the input).
# Base functionality such as saving to image-file, and drawing a grid is provided in the class.
class Image:

    # Input img can either be a path to the image file or a numpy array.
    def __init__(
        self,
        img,
        origo=[
            0,
            0,
        ],  # Origo is the position of the lower left corner of the image in the global picture.
        width=1,  # Width of the image
        height=1,  # Height of the image
        depth=0,  # Depth of image if 3D is required
        dim=2,  # Dimension of the image date
        read_metadata_from_file=False,  # Needs to be passed as True is the metadata is read from file.
        metadata_path="-",  # Path to metadata. Should be changed if metadata is not placed in the "images/metadata"-folder and taking the same name as the image. If image is not passed as a path one also needs to pass it as a path.
    ) -> None:
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
    def create_metadata_from_file(self, path) -> None:
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
    ) -> None:
        cv2.imwrite(path + name + format, self.img)
        print("Image saved as: " + path + name + format)
        if save_metadata:
            f = open(metadata_path + name + ".txt", "w")
            f.write("Origo: " + str(self.origo) + "\n")
            f.write("Width: " + str(self.width) + "\n")
            f.write("Height: " + str(self.height) + "\n")
            f.write("Dimension: " + str(self.dim) + "\n")
            f.close()

    # Draws a grid on the image and writes it.
    def draw_grid(
        self,
        DX=100,
        DY=100,
        color=(0, 0, 125),
        thickness=9,
        name="img-grid",
        path="images/modified/",
        format=".jpg",
    ) -> None:
        num_h_lines = round((self.shape[0] + self.origo[1] / self.dy) / DY)
        num_v_lines = round((self.shape[1] + self.origo[0] / self.dx) / DX)
        gridimg = self.img
        for l in range(num_h_lines):
            gridimg = cv2.line(
                gridimg,
                (
                    round(-self.origo[0] / self.dx),
                    round(l * DY),
                ),
                (round(self.shape[1]), round(l * DY)),
                color,
                thickness,
            )
        for k in range(num_v_lines):
            gridimg = cv2.line(
                gridimg,
                (
                    round(k * DX - self.origo[0] / self.dx),
                    round(0),
                ),
                (
                    round(k * DX - self.origo[0] / self.dx),
                    round(self.shape[0] + self.origo[1] / self.dy),
                ),
                color,
                thickness,
            )
        cv2.imwrite(path + name + format, gridimg)
