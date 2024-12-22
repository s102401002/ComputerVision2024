import numpy as np
import cv2
import math
def my_Gaussianfilter_processing(image, kernel_size, sigma):
    padding = kernel_size // 2
    height, width = image.shape
    image = image.astype(np.float32)
    kernel = np.empty((kernel_size, kernel_size))
    sum = 0.0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - padding
            y = j - padding
            kernel[i][j] = np.exp(-(x ** 2 + y ** 2) / 2 / sigma ** 2) / 2 / np.pi / (sigma ** 2)
            sum += kernel[i][j]
    kernel /= sum
    result = np.zeros_like(image)
    
    for i in range(0,height):
        for j in range(0,width):
            window = np.empty((kernel_size, kernel_size))
            for i2 in range(i - padding, i + padding + 1):
                for j2 in range(j - padding, j + padding + 1):
                    if i2 < 0 or j2 < 0 or i2 >= height or j2 >= width:
                        window[i2 - i + padding][j2 - j + padding] = image[i][j]
                    else:
                        window[i2 - i + padding][j2 - j + padding] = image[i2][j2]
            window *= kernel
            result[i][j] = np.sum(window)
    return result
def my_Canny_edge_detection(image, kernel_size, sigma):
    image = my_Gaussianfilter_processing(image, kernel_size, sigma)
    kernel_size = 3
    padding = kernel_size // 2
    height, width = image.shape
    image = image.astype(np.float32)
    kernelX = [[-1, -2, -1], [0, 0, 0], [1, 2, 1] ]
    MX = np.zeros_like(image)
    for i in range(0,height):
        for j in range(0,width):
            window = np.empty((kernel_size, kernel_size))
            for i2 in range(i - padding, i + padding + 1):
                for j2 in range(j - padding, j + padding + 1):
                    if i2 < 0 or j2 < 0 or i2 >= height or j2 >= width:
                        window[i2 - i + padding][j2 - j + padding] = image[i][j]
                    else:
                        window[i2 - i + padding][j2 - j + padding] = image[i2][j2]
            window *= kernelX
            MX[i][j] = np.sum(window)

    kernelY = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ]
    MY = np.zeros_like(image)
    for i in range(0,height):
        for j in range(0,width):
            window = np.empty((kernel_size, kernel_size))
            for i2 in range(i - padding, i + padding + 1):
                for j2 in range(j - padding, j + padding + 1):
                    if i2 < 0 or j2 < 0 or i2 >= height or j2 >= width:
                        window[i2 - i + padding][j2 - j + padding] = image[i][j]
                    else:
                        window[i2 - i + padding][j2 - j + padding] = image[i2][j2]
            window *= kernelY
            MY[i][j] = np.sum(window)
    M = np.zeros_like(image)
    result = np.zeros_like(image)
    for i in range(0,height):
        for j in range(0,width):
            M[i][j] = np.sqrt(MX[i][j]**2 + MY[i][j] ** 2)
    for i in range(0,height):
        for j in range(0,width):
            window = np.empty((kernel_size, kernel_size))
            for i2 in range(i - padding, i + padding + 1):
                for j2 in range(j - padding, j + padding + 1):
                    if i2 < 0 or j2 < 0 or i2 >= height or j2 >= width:
                        window[i2 - i + padding][j2 - j + padding] = M[i][j]
                    else:
                        window[i2 - i + padding][j2 - j + padding] = M[i2][j2]
            theta = math.atan2(MY[i][j], MX[i][j])/math.pi*180
            if (theta >= -157.5 and theta <= 157.5) or (theta >= -22.5 and theta <= 22.5):
                if (window[1][1] > window[0][1] and window[1][1] > window[2][1]):
                    result[i][j] = M[i][j]
                else:
                    result[i][j] = 0
            elif (theta >= 112.5 and theta <= 157.5) or (theta >= -67.5 and theta <= -22.5):
                if (window[1][1] > window[0][2] and window[1][1] > window[2][0]):
                    result[i][j] = M[i][j]
                else:
                    result[i][j] = 0
            elif (theta >= 67.5 and theta <= 112.5) or (theta >= -112.5 and theta <= -67.5):
                if (window[1][1] > window[1][0] and window[1][1] > window[1][2]):
                    result[i][j] = M[i][j]
                else:
                    result[i][j] = 0
            else:
                if (window[1][1] > window[0][0] and window[1][1] > window[2][2]):
                    result[i][j] = M[i][j]
                else:
                    result[i][j] = 0
    TL, TH = 20, 60
    for i in range(0,height):
        for j in range(0,width):
            if(result[i][j] < TL):
                result[i][j] = 0
            if(result[i][j] > TL and result[i][j] < TH):
                window = np.empty((kernel_size, kernel_size))
                for i2 in range(i - padding, i + padding + 1):
                    for j2 in range(j - padding, j + padding + 1):
                        if i2 < 0 or j2 < 0 or i2 >= height or j2 >= width:
                            window[i2 - i + padding][j2 - j + padding] = result[i][j]
                        else:
                            window[i2 - i + padding][j2 - j + padding] = result[i2][j2]
                flag = False
                for i2 in range(kernel_size):
                    for j2 in range(kernel_size):
                        if window[i2][j2] > TH:
                            flag = True
                if flag == False:
                    result[i][j] = 0
    return result
def compute_r_table(template_edges):
    rows, cols = template_edges.shape
    r_table = {}
    yc = rows / 2
    xc = cols / 2
    for y in range(rows):
        for x in range(cols):
            if template_edges[y, x] > 0: 
                dx = x - xc
                dy = y - yc
                angle = math.atan2(dy, dx)
                # r = math.hypot(dx, dy) # sqrt(x*x + y*y)
                angle = math.degrees(angle) # converts an angle from radians to degrees.
                if angle not in r_table:
                    r_table[angle] = []
                r_table[angle].append([dx, dy])
    return r_table
def generalized_hough_transform(reference, template):
    """
    實現廣義 Hough 轉換算法,檢測圖像中的目標形狀。
    """
    # 1. 邊緣檢測
    # edges = my_Canny_edge_detection(template, kernel_size=3, sigma=2)
    edges = cv2.Canny(template, 50, 150)
    r_table = compute_r_table(edges)

    # 2. 在參考圖像上進行邊緣檢測
    reference_edges = cv2.Canny(reference, 50, 150)
    rows, cols = reference_edges.shape

    # 3. 創建一個累加器陣列
    step = 1 # 角度的累加速度
    accumulator = np.zeros((rows, cols, 360/ step), dtype=np.int32)

    # 4. 對參考圖像中的邊緣點進行投票
    for y in range(rows):
        for x in range(cols):
            if reference_edges[y, x] == 255:  # 邊緣點
                # 根據 R-table 投票
                for phi, vectors in r_table.items():
                    for v in vectors:
                        for theta_idx, theta in enumerate(range(0, 360, step)):
                            theta = np.deg2rad(theta)
                            dx = v[0]
                            dy = v[1]
                            xc = int(x - dx * np.cos(theta) + dy * np.sin(theta))
                            yc = int(y - dx * np.sin(theta) - dy * np.cos(theta))
                            # 確保投票在圖像範圍內
                            if 0 <= xc < cols and 0 <= yc < rows:
                                accumulator[yc, xc, theta_idx] += 1

    return accumulator 

# 示例用法
if __name__ == "__main__":
    reference = cv2.imread('img/Refernce.png', cv2.IMREAD_GRAYSCALE)
    template = cv2.imread('img/Template.png', cv2.IMREAD_GRAYSCALE)
    
    result = generalized_hough_transform(reference, template)

    cv2.imwrite('hw3_result/test.png', result)
    