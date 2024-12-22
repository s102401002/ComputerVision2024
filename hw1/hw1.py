import numpy as np
from PIL import Image
import sys
import cv2

def my_mean_thresholding(image, window_size, c):
    height, width = image.shape
    result = np.zeros_like(image)
    
    padding = window_size // 2
    padded_image = np.pad(image, padding, mode='edge') 
    
    for i in range(height):
        for j in range(width):
            window = padded_image[i:i+window_size, j:j+window_size]
            threshold = np.mean(window) - c
            if image[i, j] > threshold:
                result[i, j] = 255 
            else:
                result[i, j] = 0
    return result

def my_Niblack_method(image, window_size, k):
    height, width = image.shape
    result = np.zeros_like(image)
    
    padding = window_size // 2
    padded_image = np.pad(image, padding, mode='edge') 
    
    for i in range(height):
        for j in range(width):
            window = padded_image[i:i+window_size, j:j+window_size]
            threshold = np.mean(window) + k * np.std(window)
            if image[i, j] > threshold:
                result[i, j] = 255 
            else:
                result[i, j] = 0
    return result

def my_Variance_based_thresholding(image):
    height, width = image.shape
    image_flat = image.flatten()
    hist, bin_edges = np.histogram(image_flat, bins=256, range=(0, 256))
    total_pixels = image.size
    max_variance = -np.inf
    threshold = 0

    for t in range(0, 256):
        w_background = np.sum(hist[:t]) / total_pixels # P1
        w_foreground = 1 - w_background # P2
        if w_background == 0 or w_foreground == 0 or np.sum(hist[:t]) == 0 or np.sum(hist[t:]) == 0:
            continue
        mean_background = np.sum(np.arange(t) * hist[:t]) / np.sum(hist[:t]) # m1
        mean_foreground = np.sum(np.arange(t, 256) * hist[t:]) / np.sum(hist[t:]) # m2
        variance = w_background * w_foreground * (mean_background - mean_foreground) ** 2 # 𝑃1𝑃2(𝑚1 −𝑚2)^2
        
        if variance > max_variance:
            max_variance = variance
            threshold = t
    
    result = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            if image[i, j] > threshold:
                result[i, j] = 255 
            else:
                result[i, j] = 0
    return result

def my_Entropy_based_thresholding(image):
    height, width = image.shape
    image_flat = image.flatten()
    hist, bin_edges = np.histogram(image_flat, bins=256, range=(0, 256))
    total_pixels = float(height * width)
    hist = hist.astype(float)
    log_hist = np.zeros_like(hist)
    log_hist = log_hist.astype(float)
    for i in range(0, 256):
        if(hist[i] > 0):
            log_hist[i] = np.log(hist[i])
    max_entropy = -np.inf
    threshold = 0
    for t in range(0, 256):
        sum1 = np.sum(hist[:t])
        sum2 = np.sum(hist[t:])
        if sum1 == 0 or sum2 == 0:
            continue
        hist1 = hist[:t] / sum1
        hist2 = hist[t:] / sum2
        h1 = -np.sum(hist1 * np.log2(hist1 + 1e-20))
        h2 = -np.sum(hist2 * np.log2(hist2 + 1e-20))
        h = h1 + h2
        if h > max_entropy:
            max_entropy = h
            threshold = t
    result = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            if image[i, j] > threshold:
                result[i, j] = 255 
            else:
                result[i, j] = 0
    return result

def my_Gaussianfilter_processing(image, kernel_size, sigma):
    padding = kernel_size // 2
    height, width, colors = image.shape
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
    for k in range(colors):
        for i in range(0,height):
            for j in range(0,width):
                window = np.empty((kernel_size, kernel_size))
                for i2 in range(i - padding, i + padding + 1):
                    for j2 in range(j - padding, j + padding + 1):
                        if i2 < 0 or j2 < 0 or i2 >= height or j2 >= width:
                            window[i2 - i + padding][j2 - j + padding] = image[i][j][k]
                        else:
                            window[i2 - i + padding][j2 - j + padding] = image[i2][j2][k]
                window *= kernel
                result[i][j][k] = np.sum(window)
    return result

def my_vector_median_filter(image, kernel_size):
    padding = kernel_size // 2
    height, width, colors = image.shape
    image = image.astype(np.float32)
    result = np.zeros_like(image)
    for k in range(colors):
        for i in range(0,height):
            for j in range(0,width):
                window = np.empty((kernel_size, kernel_size))
                for i2 in range(i - padding, i + padding + 1):
                    for j2 in range(j - padding, j + padding + 1):
                        if i2 < 0 or j2 < 0 or i2 >= height or j2 >= width:
                            window[i2 - i + padding][j2 - j + padding] = image[i][j][k]
                        else:
                            window[i2 - i + padding][j2 - j + padding] = image[i2][j2][k]
                window_flatten = window.flatten()
                SumArray = [0] * (kernel_size* kernel_size)
                for i2 in range(kernel_size* kernel_size):
                    for j2 in range(kernel_size* kernel_size):
                        distanceSum = 0.0
                        distanceSum += (window_flatten[i2] - window_flatten[j2]) ** 2
                        SumArray[i2] = (distanceSum, i2)
                
                SumArray = sorted(SumArray)
                result[i][j][k] = window_flatten[SumArray[(kernel_size* kernel_size)//2][1]]# find median
    return result

if __name__ == "__main__":
    image = cv2.imread('img/lena.bmp', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        sys.exit()
    image2 = cv2.imread('img/noise.bmp',cv2.IMREAD_COLOR)
    if image2 is None:
        print("Error: Image not found.")
        sys.exit()
    mean_result = my_mean_thresholding(image, window_size=10, c=5)
    Niblack_result = my_Niblack_method(image, window_size=10, k=-0.25)
    Variance_result = my_Variance_based_thresholding(image)
    Entropy_result =my_Entropy_based_thresholding(image)
    Vector_median_filter_result = my_vector_median_filter(image2, 5)
    Gaussianfilter_processing_filter_result = my_Gaussianfilter_processing(image2, kernel_size = 5, sigma = 2)
    Image.fromarray(mean_result).save('hw1_result/mean_thresholding_result.png')
    Image.fromarray(Niblack_result).save('hw1_result/Niblack_method_result.png')
    Image.fromarray(Variance_result).save('hw1_result/Variance_method_result.png')
    Image.fromarray(Entropy_result).save('hw1_result/Entropy_method_result.png')
    cv2.imwrite('hw1_result/Vector_median_filter_result.png', Vector_median_filter_result)
    cv2.imwrite('hw1_result/Gaussianfilter_processing_filter_result.png', Gaussianfilter_processing_filter_result)
    print("Complete.")