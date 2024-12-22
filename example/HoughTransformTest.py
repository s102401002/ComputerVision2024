from skimage import data
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
src= data.camera()
dst= cv.Canny(src, 50, 200, None, 3)
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta= lines[i][0][1]
        a= math.cos(theta)
        b= math.sin(theta)
        x0= a* rho
        y0= b* rho
        pt1= (int(x0+ 1000*(-b)), int(y0+ 1000*(a)))
        pt2 = (int(x0-1000*(-b)), int(y0-1000*(a)))
        cv.line(cdst, pt1, pt2, (255,0,0), 1)
plt.imshow(cdst)
plt.show()