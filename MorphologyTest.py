import numpy as np
import sys
import cv2

image = cv2.imread('img/binary.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found.")
    sys.exit()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilation_result = cv2.dilate(image, kernel)
erosion_result = cv2.erode(image, kernel)
opening_result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closing_result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('test_result/dilation_result.png', dilation_result)
cv2.imwrite('test_result/erosion_result.png', erosion_result)
cv2.imwrite('test_result/opening_result.png', opening_result)
cv2.imwrite('test_result/closing_result.png', closing_result)

print("Complete")