import pygetwindow as gw
import numpy as np
import cv2
from mss import mss
from PIL import Image

# Demo from opencv tutorial.

sct = mss()

def capture_frame(keyword="pygame"):
    # Screen capture in windows, might be use in future
    window = None
    for w in gw.getWindows_Keyword(keyword):
        window = w
        break

    if window is None:
        print(f"No window titled '{keyword}' found.")
        return None, 0, 0

    left, top, width, height = window.left, window.top, window.width, window.height
    monitor = {'top': top, 'left': left, 'width': width, 'height': height}

    # screenshot by mss
    img = Image.frombytes('RGB', (width, height), sct.grab(monitor).rgb)
    img_np = np.array(img)# Original RGB img
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Turn to BGR
    return img_bgr, width, height


def convert_to_grayscale(img_bgr):
    # Turn BGR to gray channel
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def apply_mask(img, mode='none', line_thick=2, x_lines=3, y_lines=3, n_circles=5):
    # Adding mask to filt the img, only return the non-filted pixels.
    # Nothing mask added.
    if mode == 'none':
        return img

    # same channel with the img input.
    if len(img.shape) == 2:
        mask = np.zeros_like(img)
        height, width = img.shape
    else:
        # BGR
        mask = np.zeros_like(img)
        height, width, _ = img.shape

    if mode == 'grid':
        for i in range(x_lines):
            x_pos = (i + 1) * width // (x_lines + 1)
            if len(img.shape) == 2:
                # Gray
                mask[:, x_pos - line_thick//2 : x_pos + line_thick//2] = \
                    img[:, x_pos - line_thick//2 : x_pos + line_thick//2]
            else:
                # BGR
                mask[:, x_pos - line_thick//2 : x_pos + line_thick//2] = \
                    img[:, x_pos - line_thick//2 : x_pos + line_thick//2]

        for j in range(y_lines):
            y_pos = (j + 1) * height // (y_lines + 1)
            if len(img.shape) == 2:
                mask[y_pos - line_thick//2 : y_pos + line_thick//2, :] = \
                    img[y_pos - line_thick//2 : y_pos + line_thick//2, :]
            else:
                mask[y_pos - line_thick//2 : y_pos + line_thick//2, :] = \
                    img[y_pos - line_thick//2 : y_pos + line_thick//2, :]

    # Circle mask
    elif mode == 'circles':
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 2
        step = max_radius // (n_circles + 1)

        for i in range(1, n_circles + 1):
            radius = i * step
            circ_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(circ_mask, (center_x, center_y), radius, 255, thickness=line_thick)
            if len(img.shape) == 2:
                mask[circ_mask == 255] = img[circ_mask == 255]
            else:
                mask[circ_mask == 255] = img[circ_mask == 255]

    return mask


def display_frames(original, processed, title_original='Original', title_processed='Processed'):
    # show orig and processed imgs both.
    cv2.imshow(title_original, original)
    cv2.imshow(title_processed, processed)


def run_demo(keyword="Chrome", use_gray=False, mask_mode='none'):
    while True:
        img_bgr, w, h = capture_frame(keyword)
        if img_bgr is None:
            break

        if use_gray:
            img_gray = convert_to_grayscale(img_bgr)
            processed = apply_mask(img_gray, mode=mask_mode, line_thick=2,
                                   x_lines=3, y_lines=3, n_circles=5)
            display_frames(img_gray, processed, title_original='OriginalGray', title_processed='ProcessedGray')
        else:
            processed = apply_mask(img_bgr, mode=mask_mode, line_thick=2,
                                   x_lines=3, y_lines=3, n_circles=5)
            display_frames(img_bgr, processed, title_original='Original', title_processed='Processed')

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
