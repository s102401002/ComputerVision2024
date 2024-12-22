import numpy as np
import sys
import cv2

image = cv2.imread('img/lena.bmp', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found.")
    sys.exit()

output = cv2.Sobel(image, ddepth = -1 , dx = 1, dy = 0, ksize = 5, scale = 2)
cv2.imwrite('sobel_result.png', output)
print("complete")
# img 來源影像
# dx 針對 x 軸抓取邊緣
# dy 針對 y 軸抓取邊緣
# ddepth 影像深度，設定 -1 表示使用圖片原本影像深度
# ksize 運算區域大小，預設 1 ( 必須是正奇數 )
# scale 縮放比例常數，預設 1 ( 必須是正奇數 )