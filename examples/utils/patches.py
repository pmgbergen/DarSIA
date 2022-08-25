import daria as da
import numpy as np


im = da.Image("images/originals/Baseline.jpg")
patch = da.Patches(im, 4)


im.show()
patch.images[0, 0].show()
patch.images[0, 1].show()
patch.images[1, 0].show()
patch.images[1, 1].show()

reassembled = patch.assemble()
reassembled.show()

# a = np.array([[[1.0, 1.1], [2.0, 2.1]], [[3.0, 3.1], [4.0, 4.1]]])
# b = np.array([[[5.0, 5.1], [6.0, 6.1]], [[7.0, 7.1], [8.0, 8.1]]])
# print(np.c_[a[:], b[:]])
# print(a)
# print(b)
