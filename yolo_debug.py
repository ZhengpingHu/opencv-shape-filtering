#!/usr/bin/env python3
# yolo_debug.py

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, help="YOLO .pt path")
    p.add_argument("--source",  required=True, help="test img path (png/jpg)")
    p.add_argument("--conf",    type=float, default=0.3, help="conf level")
    p.add_argument("--imgsz",   nargs=2, type=int,
                   default=[448,640],
                   help="same size (h w) with training")
    args = p.parse_args()

    model = YOLO(args.model, task="detect")

    img_bgr = cv2.imread(args.source)
    if img_bgr is None:
        raise FileNotFoundError(f"cannot find file {args.source}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = model(img_rgb,
                    conf=args.conf,
                    imgsz=(args.imgsz[1], args.imgsz[0]),
                    device="")

    r = results[0]

    print("===== ORIENTED DETECTION RESULT =====")
    if r.obb is not None:
        obb_data = r.obb.data.cpu().numpy()  # [cx, cy, w, h, angle]
        print("obb [cx, cy, w, h, angle]:\n", obb_data)
        
        from ultralytics.utils.ops import xywhr2xyxyxyxy
        corners = xywhr2xyxyxyxy(r.obb.data.cpu())  # shape (N, 4, 2)
        print("obb corners [x1,y1, x2,y2, x3,y3, x4,y4]:\n", corners.numpy())
    else:
        print("did not detect any frame")

    # 5) vis
    annotated = r.plot()
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLO DEBUG", annotated_bgr)
    print("\n press any button to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
