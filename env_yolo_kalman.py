#!/usr/bin/env python3
# env_yolo_kalman.py

import argparse
import time
import numpy as np
import cv2
import gymnasium as gym
from ultralytics import YOLO

def build_kalman():
    """构建 OpenCV KalmanFilter，状态为 [x, y, vx, vy], 测量为 [x, y]"""
    kf = cv2.KalmanFilter(4, 2, 0)
    kf.transitionMatrix    = np.array([[1,0,1,0],
                                       [0,1,0,1],
                                       [0,0,1,0],
                                       [0,0,0,1]], dtype=np.float32)
    kf.measurementMatrix   = np.array([[1,0,0,0],
                                       [0,1,0,0]], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    kf.statePost           = np.zeros((4,1), dtype=np.float32)
    return kf

def main():
    p = argparse.ArgumentParser("Env + YOLO-OBB + Kalman")
    p.add_argument("--model",   required=True,  help="YOLO OBB .pt 权重")
    p.add_argument("--conf",    type=float, default=0.3, help="置信度阈值")
    p.add_argument("--fps",     type=float, default=5.0, help="最大帧率")
    p.add_argument("--gravity", type=float, default=-3.5, help="LunarLander 重力")
    args = p.parse_args()

    # 1) 创建环境并 reset
    env = gym.make("LunarLander-v3", render_mode="rgb_array", gravity=args.gravity)
    obs, info = env.reset()

    # 2) 加载模型
    print(f"[INFO] 加载 YOLO 模型: {args.model}")
    model = YOLO(args.model, task="detect")
    print("[DEBUG] model.names =", model.names)
    # 根据 yaml，你的 0: lander, 1: landing_point
    lander_id = 0
    point_id  = 1

    # 3) 构建卡尔曼
    kf = build_kalman()
    interval = 1.0 / args.fps
    print("[INFO] 启动主循环，按 Ctrl+C 退出")

    step = 0
    cum_reward = 0.0

    try:
        while True:
            t0 = time.time()

            # 渲染一帧
            frame = env.render()

            # YOLO 推理（显式指定 imgsz 保持与训练一致）
            r = model(frame,
                      conf=args.conf,
                      imgsz=(640, 448),  # (w,h)
                      device="")[0]

            # 可视化
            ann = r.plot()

            # 卡尔曼预测
            _ = kf.predict()

            # —— 1) 从 r.obb.data 拿原始数组 shape (N,7): [cx,cy,w,h,angle,conf,cls] ——
            if r.obb is not None:
                raw = r.obb.data.cpu().numpy()
            else:
                raw = np.zeros((0,7), dtype=np.float32)

            # —— DEBUG 打印一下看看 —— 
            print(f"[DEBUG] raw.shape={raw.shape}  first rows:\n{raw[:2]}")

            # —— 2) 拆分参数和类别 —— 
            obb_all  = raw[:, :5]            # cx, cy, w, h, angle
            cls_all  = raw[:, 6].astype(int) # class_id

            # 筛出 lander / landing_point
            idx_l = np.where(cls_all == lander_id)[0]
            obb_l  = obb_all[idx_l[:1]] if idx_l.size>0 else np.zeros((0,5), dtype=np.float32)

            idx_p = np.where(cls_all == point_id)[0]
            obb_p  = obb_all[idx_p[:1]] if idx_p.size>0 else np.zeros((0,5), dtype=np.float32)

            # 用 lander 的 (cx,cy) 做卡尔曼 correct
            if obb_l.shape[0] > 0:
                cx, cy = float(obb_l[0,0]), float(obb_l[0,1])
                meas = np.array([[cx],[cy]], dtype=np.float32)
                kf.correct(meas)

            # 读取滤波后状态
            st = kf.statePost.flatten()
            state6 = [float(st[0]), float(st[1]),
                      float(st[2]), float(st[3]),
                      0.0, 0.0]

            # 执行动作 & step
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            cum_reward += reward

            # 日志输出
            msg  = f"[Step {step}] act={action}"
            msg += f", lander_det={obb_l.shape[0]}"
            if obb_l.shape[0]:
                msg += f" center_l=({obb_l[0,0]:.1f},{obb_l[0,1]:.1f})"
            msg += f", point_det={obb_p.shape[0]}"
            if obb_p.shape[0]:
                msg += f" center_p=({obb_p[0,0]:.1f},{obb_p[0,1]:.1f})"
            msg += f"  State={state6}  rew={reward:.3f} cum={cum_reward:.3f}"
            print(msg)

            step += 1

            # 显示
            bgr = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
            cv2.imshow("YOLO-Kalman", bgr)
            if cv2.waitKey(1) in (27, ord('q')):
                break

            # episode 结束
            if terminated or truncated:
                obs, info = env.reset()
                kf = build_kalman()
                cum_reward = 0.0
                step = 0

            # 控制帧率
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("[INFO] 已退出")

if __name__ == "__main__":
    main()
