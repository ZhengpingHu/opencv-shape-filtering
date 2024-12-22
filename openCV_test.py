import pygetwindow as gw
import numpy as np
import cv2
from mss import mss
from PIL import Image

sct = mss()

def set_grid_shape(x_lines=1, y_lines=1):
    return x_lines, y_lines

def set_circle_count(n_circles=1):
    return n_circles

mode = 'circles'  # 'grid' or 'circles'
line_thickness = 1
circle_number = 10
grid_x = 3
grid_y = 3

"Must have a specific title name for the processing"
window = None
for w in gw.getWindowsWithTitle("Chrome"):
    window = w
    break

if window:
    x_lines, y_lines = set_grid_shape(grid_x, grid_y)
    n_circles = set_circle_count(circle_number)

    while True:
        left, top, width, height = window.left, window.top, window.width, window.height
        monitor = {'top': top, 'left': left, 'width': width, 'height': height}

        img = Image.frombytes('RGB', (width, height), sct.grab(monitor).rgb)
        img_np = np.array(img)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        mask = np.zeros_like(img_bgr)

        if mode == 'grid':
            for i in range(x_lines):
                x_pos = (i + 1) * width // (x_lines + 1)
                mask[:, x_pos - line_thickness // 2:x_pos + line_thickness // 2] = img_bgr[
                    :, x_pos - line_thickness // 2:x_pos + line_thickness // 2
                ]

            for j in range(y_lines):
                y_pos = (j + 1) * height // (y_lines + 1)
                mask[y_pos - line_thickness // 2:y_pos + line_thickness // 2, :] = img_bgr[
                    y_pos - line_thickness // 2:y_pos + line_thickness // 2, :
                ]

        elif mode == 'circles':
            center_x, center_y = width // 2, height // 2
            max_radius = min(width, height) // 2
            step = max_radius // (n_circles + 1)

            for i in range(1, n_circles + 1):
                radius = i * step
                circular_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(circular_mask, (center_x, center_y), radius, 255, thickness=line_thickness)
                mask[circular_mask == 255] = img_bgr[circular_mask == 255]

        cv2.imshow('Filtered View', mask)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
else:
    print("No target window found.")
