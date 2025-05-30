import cv2
import math
import os

image_path = r"D:\Git\opencv-shape-filtering\runs\obb\predict2\1.jpg"
label_path = r"D:\Git\opencv-shape-filtering\runs\obb\predict2\labels\1.txt"
save_path  = r"D:\Git\opencv-shape-filtering\runs\obb\predict2\angle_vis.jpg"
class_names = ["lander", "landing_point"] 

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"wrong path for img: {image_path}")
h, w = img.shape[:2]

if not os.path.exists(label_path):
    raise FileNotFoundError(f"label file not found: {label_path}")

with open(label_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    cls_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    pts = [(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])]

    for i in range(4):
        pt1 = pts[i]
        pt2 = pts[(i + 1) % 4]
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)

    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    angle = math.degrees(math.atan2(dy, dx))

    cx = sum(p[0] for p in pts) // 4
    cy = sum(p[1] for p in pts) // 4
    label = f"{class_names[cls_id]} {angle:.1f}°"
    cv2.putText(img, label, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

cv2.imwrite(save_path, img)
print(f"img saved at:{save_path}")
