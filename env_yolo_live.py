#!/usr/bin/env python3
"""
env_yolo_live_with_launch.py

1) （可选）后台启动 env.py，渲染 LunarLander-v3 窗口
2) 实时抓取指定窗口内容，送入已训练好的 YOLOv11-obb 模型推理
3) 将检测结果叠加回窗口并展示
"""

import argparse, time, subprocess, sys, os
import numpy as np
import cv2
from PIL import Image
from mss import mss
import pygetwindow as gw
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',      required=True, help="YOLO .pt 模型路径")
    p.add_argument('--conf',       type=float, default=0.3, help="置信度阈值")
    p.add_argument('--fps',        type=float, default=5.0, help="最大帧率")
    p.add_argument('--title',      default='lunar', help="窗口标题子串（不区分大小写）")
    p.add_argument('--launch-env', action='store_true',
                   help="后台启动 env.py")
    p.add_argument('--env-script', default='env.py',
                   help="要启动的 env.py 脚本路径")
    args = p.parse_args()

    # 1) 可选：后台启动 env.py
    proc = None
    if args.launch_env:
        python_exec = sys.executable
        if not os.path.isfile(args.env_script):
            print(f"[ERROR] 找不到脚本 {args.env_script}")
            sys.exit(1)
        proc = subprocess.Popen([python_exec, args.env_script],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        print(f"[INFO] 已启动环境脚本 '{args.env_script}' (pid={proc.pid})，等待窗口渲染…")
        time.sleep(3)  # 等 env.py 稳定打开窗口

    # 2) 加载 YOLO 模型
    print(f"[INFO] 加载模型 {args.model}")
    model = YOLO(args.model, task="detect")

    sct      = mss()
    interval = 1.0 / args.fps

    print(f"[INFO] 开始查找窗口匹配 \"{args.title}\" …")
    try:
        while True:
            # 查找所有可见窗口，匹配标题子串
            wins = [w for w in gw.getAllWindows()
                    if args.title.lower() in w.title.lower() and getattr(w, 'visible', True)]
            if not wins:
                titles = [w.title for w in gw.getAllWindows() if getattr(w, 'visible', True)]
                print(f"[WARN] 未找到窗口 “{args.title}”。前 10 个可见窗口：{titles[:10]}")
                time.sleep(interval)
                continue

            w = wins[0]
            left, top, width, height = w.left, w.top, w.width, w.height

            # 截屏 → BGR ndarray
            shot  = sct.grab({'left': left, 'top': top, 'width': width, 'height': height})
            frame = np.array(Image.frombytes('RGB', (width, height), shot.rgb))

            # 推理 & 画框（返回 RGB）
            results   = model(frame, conf=args.conf)[0]
            annotated = results.plot()

            # 转 BGR 并展示
            bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imshow('Env + YOLO Live', bgr)
            key = cv2.waitKey(int(interval*1000)) & 0xFF
            if key in (27, ord('q')):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        # 3) 若启动了 env.py，就在退出时一起杀掉
        if proc:
            proc.terminate()
            print(f"[INFO] 已终止环境脚本 pid={proc.pid}")

if __name__ == '__main__':
    main()
