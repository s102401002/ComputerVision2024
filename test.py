import numpy as np
kernel_size = 3
padding = kernel_size // 2
kernel = np.empty((kernel_size, kernel_size))
image = [[14, 15, 16], [24, 25, 26], [34, 35, 36]]
sum = 0.0
sigma = 1.5
for i in range(kernel_size):
    for j in range(kernel_size):
        x = i - padding
        y = j - padding
        kernel[i][j] = np.exp(-(x ** 2 + y ** 2) / 2 / sigma ** 2) / 2 / np.pi / (sigma ** 2)
        sum += kernel[i][j]
kernel /= sum
image *= kernel
print(image)