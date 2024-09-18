import cv2
import numpy as np
import sys

def empty(v):
    pass

image = cv2.imread('img/lena.bmp', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found.")
    sys.exit()
cv2.namedWindow('result',cv2.WINDOW_NORMAL)

cv2.namedWindow('trackBar')
cv2.createTrackbar('window_size', 'trackBar', 15, 200, empty)
cv2.createTrackbar('const', 'trackBar', 2, 50, empty)

while True:
    window_size = cv2.getTrackbarPos('window_size', 'trackBar')
    const = cv2.getTrackbarPos('const', 'trackBar')
    if window_size <= 1: 
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1

    print(window_size, const)


    binary_image = cv2.adaptiveThreshold(
        image,
        255,  # max value
        cv2.ADAPTIVE_THRESH_MEAN_C,  # adaptive method
        cv2.THRESH_BINARY,  # thresholding type
        window_size,  # block size (must be odd)
        const  # constant subtracted from the mean
    )
    
    # Show the result
    cv2.imshow("result", binary_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()