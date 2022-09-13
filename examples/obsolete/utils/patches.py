import daria as da
import numpy as np


im = da.Image("images/originals/Baseline.jpg")
patch = da.Patches(im, 16, patch_overlay = 0.1)

im.show()
patch.images[0, 0].show()
patch.images[0, 1].show()
patch.images[1, 0].show()
patch.images[1, 1].show()

reassembled = patch.assemble()
reassembled.show()