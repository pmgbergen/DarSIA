import daria as da
import cv2

preimarr = cv2.imread("images/originals/Baseline.jpg", 0)
preimage = da.Image(preimarr, (0, 0), 250, 180)
preimage.resize(0.1, 0.1)
reg_image = da.tv_denoising(preimage, 0.5, 1, da.cg, verbose=True)
reg_image.show()
preimage.show()
