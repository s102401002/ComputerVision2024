import numpy as np
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
    image = my_Sobel_edge_detection(image)
    

if __name__ == "__main__":
    image = cv2.imread('img/lena.bmp', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        sys.exit()
    image2 = cv2.imread('img/binary.png',cv2.IMREAD_COLOR)
    if image2 is None:
        print("Error: Image not found.")
        sys.exit()
    Sobel_result = my_Sobel_edge_detection(image=image)
    Prewitt_result = my_Prewitt_edge_detection(image=image)
    cv2.imwrite('hw2_result/sobel_result.png', Sobel_result)
    cv2.imwrite('hw2_result/prewitt_result.png', Prewitt_result)
    print("Complete.")