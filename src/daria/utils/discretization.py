from daria import Image
import numpy as np

class Discretization:

    def __init__(self, image: Image) -> None:
        self.image = image
        self.im = image.img


    #Forward difference in the x-direction
    def dxf(self) -> np.ndarray:
        dxf =