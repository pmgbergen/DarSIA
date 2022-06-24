import daria as dia
import cv2


img = cv2.imread("images/originals/Profilbilde.jpg")
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
cv2.imwrite("images/modified/Profilbilde05.jpg", img)
img = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)
cv2.imwrite("images/modified/Profilbilde1.jpg", img)
