import cv2
import daria.corrections.color as dacl

# -------- Convert the image into linear RGB color space

# Fetch image, in BGR
img_BGR = cv2.imread("../images/fluidflower/Baseline.jpg")

# Convert to RGB (int)
# img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

# -------- Match colors and apply correction to original picture using the nonlinear RGB colour space

# Apply colour correction based on color checker
colorcorrection = dacl.ColorCorrection()

# Need to specify a ROI which contains the color checker
roi_cc = (slice(100, 600), slice(350, 600))

# Apply color correction
corrected_baseline_RGB = colorcorrection.adjust(img_RGB, roi_cc, verbosity=True, whitebalancing=True)

# -------- Plot images pre and post correction, and store both images

cv2.imwrite("colour-correction/baseline.jpg", img_BGR)
cv2.namedWindow("original baseline", cv2.WINDOW_NORMAL)
cv2.imshow("original baseline", img_BGR)
cv2.waitKey(0)

# Convert to BGR - for plotting
corrected_baseline_BGR = cv2.cvtColor(corrected_baseline_RGB, cv2.COLOR_RGB2BGR)
cv2.imwrite("colour-correction/baseline_corrected.jpg", corrected_baseline_BGR)
cv2.namedWindow("corrected baseline", cv2.WINDOW_NORMAL)
cv2.imshow("corrected baseline", corrected_baseline_BGR)
cv2.waitKey(0)
