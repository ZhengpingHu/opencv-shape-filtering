#!/usr/bin/env python3
"""
env_yolo_live_with_launch.py
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
    p.add_argument('--model',      required=True, help="YOLO .pt path of model")
    p.add_argument('--conf',       type=float, default=0.3, help="conf level")
    p.add_argument('--fps',        type=float, default=5.0, help="max FPS")
    p.add_argument('--title',      default='lunar', help="keyword of window")
    p.add_argument('--launch-env', action='store_true',
                   help="backup env.py")
    p.add_argument('--env-script', default='env.py',
                   help="path to env.py script")
    args = p.parse_args()

    proc = None
    if args.launch_env:
        python_exec = sys.executable
        if not os.path.isfile(args.env_script):
            print(f"[ERROR] cannot find {args.env_script}")
            sys.exit(1)
        proc = subprocess.Popen([python_exec, args.env_script],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        print(f"[INFO] launch '{args.env_script}' (pid={proc.pid})，waiting for window…")
        time.sleep(3)  # env.py

    # 2) load YOLO model
    print(f"[INFO] load model {args.model}")
    model = YOLO(args.model, task="detect")

    sct      = mss()
    interval = 1.0 / args.fps

    print(f"[INFO] start searching \"{args.title}\" …")
    try:
        while True:
            # search window
            wins = [w for w in gw.getAllWindows()
                    if args.title.lower() in w.title.lower() and getattr(w, 'visible', True)]
            if not wins:
                titles = [w.title for w in gw.getAllWindows() if getattr(w, 'visible', True)]
                print(f"[WARN] cannot find “{args.title}”。top 10 window：{titles[:10]}")
                time.sleep(interval)
                continue

            w = wins[0]
            left, top, width, height = w.left, w.top, w.width, w.height

            # screenshot → BGR ndarray
            shot  = sct.grab({'left': left, 'top': top, 'width': width, 'height': height})
            frame = np.array(Image.frombytes('RGB', (width, height), shot.rgb))

            # prediction & frames（return RGB）
            results   = model(frame, conf=args.conf)[0]
            annotated = results.plot()

            # return BGR display
            bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imshow('Env + YOLO Live', bgr)
            key = cv2.waitKey(int(interval*1000)) & 0xFF
            if key in (27, ord('q')):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        # 3) if launch env.py，kill the process when exit
        if proc:
            proc.terminate()
            print(f"[INFO] end script pid={proc.pid}")

if __name__ == '__main__':
    main()
