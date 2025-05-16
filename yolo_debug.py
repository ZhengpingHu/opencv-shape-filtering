#!/usr/bin/env python3
# yolo_debug.py

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, help="YOLO .pt 模型路径")
    p.add_argument("--source",  required=True, help="测试图像路径 (png/jpg)")
    p.add_argument("--conf",    type=float, default=0.3, help="置信度阈值")
    p.add_argument("--imgsz",   nargs=2, type=int,
                   default=[448,640],
                   help="推理时的输入尺寸 (h w)，与训练时保持一致")
    args = p.parse_args()

    # 1) 加载模型（task="detect" 对 OBB 会自动调用 obb 检测）
    model = YOLO(args.model, task="detect")

    # 2) 读取一张测试图
    img_bgr = cv2.imread(args.source)
    if img_bgr is None:
        raise FileNotFoundError(f"找不到文件 {args.source}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3) 推理：指定 imgsz 与训练匹配，保留原始纵横比再填充
    results = model(img_rgb,
                    conf=args.conf,
                    imgsz=(args.imgsz[1], args.imgsz[0]),  # ultralytics 要 (w,h)
                    device="")       # 留空表示用模型内置 device

    r = results[0]

    # 4) 打印 boxes 相关字段
    print("===== ORIENTED DETECTION RESULT =====")
    # r.obb 是一个 BaseTensor，data 中存储了 [cx, cy, w, h, angle]
    if r.obb is not None:
        # 把 OBB 数据移到 CPU 并转为 NumPy 数组，形状 (N, 5)
        obb_data = r.obb.data.cpu().numpy()  # [cx, cy, w, h, angle]
        print("obb [cx, cy, w, h, angle]:\n", obb_data)
        
        # 如果你需要四角顶点坐标，可以用内置的转换函数：
        from ultralytics.utils.ops import xywhr2xyxyxyxy
        corners = xywhr2xyxyxyxy(r.obb.data.cpu())  # 形状 (N, 4, 2)
        print("obb corners [x1,y1, x2,y2, x3,y3, x4,y4]:\n", corners.numpy())
    else:
        print("本帧未检测到任何定向边界框")

    # 5) 可视化
    annotated = r.plot()  # 返回 RGB
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLO DEBUG", annotated_bgr)
    print("\n按任意键关闭窗口并退出")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
