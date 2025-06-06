import cv2
import os
import numpy as np

# 输入文件夹路径
image_folder = './'  # 或者 valid/images/test/images
label_folder = './'        # 存放你的YOLOv11-OBB txt标签
output_folder = 'vis_output'   # 可视化输出

os.makedirs(output_folder, exist_ok=True)

# 获取所有图片名
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]

for img_file in image_files:
    # 读取图像
    image_path = os.path.join(image_folder, img_file)
    image = cv2.imread(image_path)

    # 构造对应的标签路径
    label_name = os.path.splitext(img_file)[0] + '.txt'
    label_path = os.path.join(label_folder, label_name)

    if not os.path.exists(label_path):
        continue

    # 读取并画出每个检测框
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            # 将归一化坐标还原为像素坐标
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * image.shape[1])
                y = int(coords[i+1] * image.shape[0])
                points.append((x, y))
            points = np.array(points, dtype=np.int32)

            # 画多边形
            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            # 可选：添加类别文字
            cv2.putText(image, f'{cls_id}', points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 保存可视化图像
    out_path = os.path.join(output_folder, img_file)
    cv2.imwrite(out_path, image)

print("✅ 可视化完成，请查看 vis_output 文件夹。")
