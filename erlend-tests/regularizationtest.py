import daria as da
import cv2


# Reading greyscale image to DarIA image
pre_im_array = cv2.imread("images/originals/Baseline.jpg", 0)
pre_im = da.Image(pre_im_array, (0, 0), 250, 180)

# Resizing image to make regularization more computationally efficient
pre_im.resize(0.1, 0.1)

# Apply  TV Denoising
reg_im05 = da.tv_denoising(pre_im, 0.5, 1, verbose=True)
reg_im1 = da.tv_denoising(pre_im, 1, 1, verbose=True)
reg_im2 = da.tv_denoising(pre_im, 2, 1, verbose=True)

# Show images
pre_im.show()
reg_im05.show()
reg_im1.show()
reg_im2.show()
