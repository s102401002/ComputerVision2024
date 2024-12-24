import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
from functools import lru_cache
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
                angle = math.degrees(angle) 
                # converts an angle from radians to degrees.
                if angle not in r_table:
                    r_table[angle] = []
                r_table[angle].append([dx, dy])
    return r_table

@lru_cache(None)
def get_cos_sin_values(angle_step):
    angles_deg = np.linspace(0, 360, 360 // angle_step, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)
    cos_values = np.cos(angles_rad)
    sin_values = np.sin(angles_rad)
    return cos_values, sin_values

def generalized_hough_transform(reference, template, angle_step):
    # 1. 邊緣檢測
    template_edges = cv2.Canny(template, 50, 150)
    reference_edges = cv2.Canny(reference, 50, 150)
    # 2. 計算 R-table
    r_table = compute_r_table(template_edges)
    # 3. 創建accumulator, 所有角度的cos,sin值
    rows, cols = reference_edges.shape
    cos_values, sin_values = get_cos_sin_values(angle_step)
    accumulator = np.zeros((rows, cols, len(cos_values)), dtype=np.int32)

    # 4. 獲取reference所有邊緣點
    edge_points = np.argwhere(reference_edges == 255)
    # 5. 投票過程
    for y, x in edge_points:
        # 根據 R-table 投票
        for phi, vectors in r_table.items():
            for v in vectors:
                dx, dy = v
                # 計算投票的 xc 和 yc 值 
                # (直接乘以sin&cos陣列，產生length為360//angle_step的陣列)
                xc = x - dx * cos_values + dy * sin_values
                yc = y - dx * sin_values - dy * cos_values
                # 轉換為整數並確保在圖像範圍內
                xc = np.round(xc).astype(int)
                yc = np.round(yc).astype(int)
                # 篩選出有效的座標
                valid = (0 <= xc) & (xc < cols) & (0 <= yc) & (yc < rows)
                # 更新累加器
                valid_x_c = xc[valid]
                valid_y_c = yc[valid]
                valid_angle_idx = np.arange(len(cos_values))[valid]
                accumulator[valid_y_c, valid_x_c, valid_angle_idx] += 1
    x, y, angle_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    angle = angle_idx * angle_step
    return x, y, angle
def display_result(image, x, y, angle, box_size=(50, 50), ref='', tmp='', template_img=None):
    # 將圖片轉換為 0~255 的範圍
    result_image = np.clip(image, 0, 255).astype(np.uint8)

    # 使用 gridspec 將畫面分為上下兩部分
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])  # 上下部分的比例

    # 上半部顯示模板圖像及模板檔名
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(template_img, cmap="gray")
    ax0.set_title(f"Template: {tmp}", fontsize=16)
    ax0.axis("off")

    # 下半部顯示邊緣檢測的結果
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(result_image, cmap="gray")

    # 繪製中心點 x, y是相反的
    ax1.scatter(y, x, c='blue', s=20, label=f"Center (x={y}, y={x}, angle={angle}°)")

    # 繪製方框
    box_height, box_width = box_size
    top_left_x = y - box_width // 2
    top_left_y = x - box_height // 2
    rect = patches.Rectangle(
        (top_left_x, top_left_y),  
        box_width,                
        box_height,
        linewidth=1,
        edgecolor='yellow',
        facecolor='none'
    )
    ax1.add_patch(rect)

    ax1.legend(loc='upper right', fontsize=12)
    ax1.set_title(f"Reference: {ref}", fontsize=16)
    ax1.axis("off")

    plt.tight_layout()
    plt.savefig(f'hw3_result/result_{angle}.png')
    plt.show()
    
def rotate_template(angle):
    template = cv2.imread('img/template/Template.png', cv2.IMREAD_GRAYSCALE)
    template_path = 'img/template/Template.png'
    base_name = os.path.splitext(os.path.basename(template_path))[0]  #img/template/Template
    output_dir = os.path.dirname(template_path)  # img/template
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    h, w = template.shape
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 1.0是縮放比例
    rotated_image = cv2.warpAffine(template, rotation_matrix, (w, h))

    output_filename = f"{base_name}_{angle}.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, rotated_image)
def main():
    reference_path = 'img/Refernce.png'
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    angle_step = 1
    angle =  int( input('輸入想要辨識的角度：') ) 
    angle = angle % 360
    print(angle)
    template_path = 'img/template/Template_' + str(angle) + '.png'
    if not os.path.exists(template_path):
        rotate_template(angle)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    x, y, angle = generalized_hough_transform(reference, template, angle_step)
    display_result(reference, x, y, angle, box_size=template.shape, 
                   ref=reference_path, tmp=template_path, template_img=template)
if __name__ == '__main__':
    main()
