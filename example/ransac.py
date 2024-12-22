from skimage import data
import cv2
import matplotlib.pyplot as plt
import numpy as np
 # 生成一個合成的測試圖像
def detect_line_ransac(points, max_iterations, distance_threshold):
    best_line = None
    max_inliers = 0
    for _ in range(max_iterations):
        sample_points = points[np.random.choice(points.shape[0], 2, 
        replace=False)]
        x1, y1 = sample_points[0]
        x2, y2 = sample_points[1]
        # 計算線段的參數(ax + by + c = 0)
        A = y2 -y1
        B = x1 -x2
        C = x2 * y1 -x1 * y2
        distances = np.abs(A * points[:, 0] + B * points[:, 1] + C) / np.sqrt(A**2 + B**2)
        inliers = points[distances < distance_threshold]
        if len(inliers) > max_inliers:
            best_line = (A, B, C)
            max_inliers = len(inliers)
    return best_line, inliers

image = data.camera()
edges = cv2.Canny(image, 50, 150, apertureSize=3)
max_iterations = 100
distance_threshold = 10
lines_detected = [] # 存放偵測到的線條
points = np.column_stack(np.where(edges > 0))
while len(points) > 50: # 迴圈直到剩餘點不足以擬合新線段
    best_line, inliers = detect_line_ransac(points, max_iterations, distance_threshold)
    if best_line is not None:
        lines_detected.append(best_line)
        points = np.array([pt for pt in points if pt.tolist() not in inliers.tolist()]) # 移除內點

output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for A, B, C in lines_detected:
    if B != 0: # 確保不會除以零
        x1, x2 = 0, image.shape[1]
        y1 = int(-C / B)
        y2 = int(-(A * x2 + C) / B)
    else:
        y1, y2 = 0, image.shape[0]
        x1 = int(-C / A)
        x2 = int(-C / A)
    cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
plt.imshow(output_image)
plt.title("Detected Lines")
plt.show()