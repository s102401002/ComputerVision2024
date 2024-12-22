import numpy as np
import math
import sys
import cv2    

def my_Sobel_edge_detection(image):
    kernel_size = 3
    padding = kernel_size // 2
    height, width = image.shape
    image = image.astype(np.float32)
    kernelX = [[-1, -2, -1], [0, 0, 0], [1, 2, 1] ]
    resultX = np.zeros_like(image)
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
            resultX[i][j] = np.sum(window)

    kernelY = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ]
    resultY = np.zeros_like(image)
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
            resultY[i][j] = np.sum(window)
    result = np.zeros_like(image)
    for i in range(0,height):
        for j in range(0,width):
            result[i][j] = np.sqrt(resultX[i][j]**2 + resultY[i][j] ** 2)
    return result

def my_Prewitt_edge_detection(image):
    kernel_size = 3
    padding = kernel_size // 2
    height, width = image.shape
    image = image.astype(np.float32)
    kernelX = [[-1, -1, -1], [0, 0, 0], [1, 1, 1] ]
    resultX = np.zeros_like(image)
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
            resultX[i][j] = np.sum(window)

    kernelY = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1] ]
    resultY = np.zeros_like(image)
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
            resultY[i][j] = np.sum(window)
    result = np.zeros_like(image)
    for i in range(0,height):
        for j in range(0,width):
            result[i][j] = np.sqrt(resultX[i][j]**2 + resultY[i][j] ** 2)
    return result

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


def my_Dilation(image, kernel):
    kernel_height, kernel_width = len(kernel), len(kernel[0])
    paddingX = kernel_height // 2
    paddingY = kernel_width // 2
    height, width, color = image.shape
    result = np.empty((height, width))
    for i in range(0,height):
        for j in range(0,width):
            window = np.empty((kernel_height, kernel_width))
            for i2 in range(i - paddingX, i + paddingX + 1):
                for j2 in range(j - paddingY, j + paddingY + 1):
                    if i2 < 0 or j2 < 0 or i2 >= height or j2 >= width:
                        window[i2 - i + paddingX][j2 - j + paddingY] = image[i][j][0]
                    else:
                        window[i2 - i + paddingX][j2 - j + paddingY] = image[i2][j2][0]
            val = 0
            for i2 in range(kernel_height):
                if val == 255:
                    break
                for j2 in range(kernel_width):
                    if val == 255:
                        break
                    if window[i2][j2] == 255 and kernel[i2][j2] == 255:
                        val = 255
            result[i][j] = val
    return result
    
def my_Erosion(image, kernel):
    kernel_height, kernel_width = len(kernel), len(kernel[0])
    paddingX = kernel_height // 2
    paddingY = kernel_width // 2
    height, width, color = image.shape
    result = np.empty((height, width))
    for i in range(0,height):
        for j in range(0,width):
            window = np.empty((kernel_height, kernel_width))
            for i2 in range(i - paddingX, i + paddingX + 1):
                for j2 in range(j - paddingY, j + paddingY + 1):
                    if i2 < 0 or j2 < 0 or i2 >= height or j2 >= width:
                        window[i2 - i + paddingX][j2 - j + paddingY] = image[i][j][0]
                    else:
                        window[i2 - i + paddingX][j2 - j + paddingY] = image[i2][j2][0]
            val = 255
            for i2 in range(kernel_height):
                for j2 in range(kernel_width):
                    if window[i2][j2] != 255:
                        val = 0
            result[i][j] = val
    return result

def my_Opening(image, kernel):
    result = my_Erosion(image, kernel)
    height, width = result.shape
    temp_image = np.empty((height, width, 1))
    for i in range(height):
        for j in range(width):
            temp_image[i][j][0] = result[i][j]
    result = my_Dilation(temp_image, kernel)
    return result

def my_Closing(image, kernel):
    result = my_Dilation(image, kernel)
    height, width = result.shape
    temp_image = np.empty((height, width, 1))
    for i in range(height):
        for j in range(width):
            temp_image[i][j][0] = result[i][j]
    result = my_Erosion(temp_image, kernel)
    return result

if __name__ == "__main__":
    image = cv2.imread('img/lena.bmp', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        sys.exit()
    image2 = cv2.imread('img/binary.png',cv2.IMREAD_COLOR)
    if image2 is None:
        print("Error: Image not found.")
        sys.exit()
    # Sobel_result = my_Sobel_edge_detection(image=image)
    # Prewitt_result = my_Prewitt_edge_detection(image=image)
    kernel = [[255] * 3 for _ in range (3)]
    # Dilation_result = my_Dilation(image2, kernel)
    # Erosion_result = my_Erosion(image2, kernel)
    # Opening_result = my_Opening(image2, kernel)
    # Closing_result = my_Closing(image2, kernel)
    Canny_result = my_Canny_edge_detection(image, kernel_size=3, sigma=2)
    # cv2.imwrite('hw2_result/sobel_result.png', Sobel_result)
    # cv2.imwrite('hw2_result/prewitt_result.png', Prewitt_result)
    # cv2.imwrite('hw2_result/dilation_result.png', Dilation_result)
    # cv2.imwrite('hw2_result/erosion_result.png', Erosion_result)
    # cv2.imwrite('hw2_result/opening_result.png', Opening_result)
    # cv2.imwrite('hw2_result/closing_result.png', Closing_result)
    cv2.imwrite('hw2_result/canny_result.png', Canny_result)
    print("Complete.")