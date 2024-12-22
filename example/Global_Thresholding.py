import cv2
import numpy as np
def global_thresholding(hist, total_pixels):
    for t in range(1, 255):
        w0 = np.sum(hist[:t]) / total_pixels
        w1 = np.sum(hist[t:]) / total_pixels
        if w0 == 0 or w1 == 0:
            continue
        mean0 = np.sum(np.arange(0, t) * hist[:t]) / np.sum(hist[:t])
        mean1 = np.sum(np.arange(t, 256) * hist[t:]) / np.sum(hist[t:])
        variance0 = np.sum(((np.arange(0, t) - mean0) ** 2) * hist[:t]) / np.sum(hist[:t])
        variance1 = np.sum(((np.arange(t, 256) - mean1) ** 2) * hist[t:]) / np.sum(hist[t:])
        likelihood = w0 * np.log(variance0) + w1 * np.log(variance1)
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            threshold = t
    return threshold
image = cv2.imread('img/lena.bmp', cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
thresholding = global_thresholding(hist, image.size)
ret, otsu_threshold = cv2.threshold(
    image,
    thresholding, # 使用Otsu時要忽略的值
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
while not ( cv2.waitKey(0) & 0xFF == ord('q')):
    cv2.imshow("result", otsu_threshold)
