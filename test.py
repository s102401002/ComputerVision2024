import numpy as np

a = np.array([[1,2,5], [3,4,6]])

padded_image = np.pad(a, 0, mode='edge') 
print(padded_image.flatten())
pixel_values = padded_image.flatten()

# Calculate the histogram
histogram, bin_edges = np.histogram(pixel_values, bins=256, range=(0, 256))
print(histogram)