import cv2

image = cv2.imread('img/lena.bmp', cv2.IMREAD_GRAYSCALE)
ret, otsu_threshold = cv2.threshold(
    image,
    0, # 使用Otsu時要忽略的值
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
while not ( cv2.waitKey(0) & 0xFF == ord('q')):
    cv2.imshow("result", otsu_threshold)
