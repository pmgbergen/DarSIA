import daria as da
import cv2

# Read curved im
curved_im_pre = cv2.imread("images/originals/NonCurvCorrected.JPG")

# Create curvature corrected im
im_pre = da.curvature_correction(curved_im_pre)

# Show images
curved_im = da.Image(curved_im_pre)
im = da.Image(im_pre)

curved_im.show()
im.show()
