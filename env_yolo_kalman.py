#!/usr/bin/env python3
# env_yolo_kalman.py

import argparse
import numpy as np
import cv2
import gymnasium as gym
import torch
from ultralytics import YOLO
from collections import deque

class FeatureEnv:
    """
    环境包装：将 YOLO-OBB + 卡尔曼滤波 与 Gym LunarLander 结合，
    返回 8 维特征，并在 launch_env=True 时使用 OpenCV 窗口显示带框画面。
    """
    def __init__(self, model_path, title, fps, gravity, launch_env=False):
        # 1) 创建 rgb_array 环境
        self.env = gym.make(
            "LunarLander-v3",
            render_mode="rgb_array",
            gravity=gravity
        )
        self.frame_interval = 1.0 / fps

        # 2) 加载 YOLO（关闭自身日志）
        self.model = YOLO(model_path, task="detect", verbose=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        # 3) 卡尔曼滤波器
        self.kf = self._build_kalman()

        # 4) 如果需要可视化，就用 OpenCV 创建一个窗口
        self.launch_env = launch_env
        if self.launch_env:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            self.win_name = title

    def _build_kalman(self):
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

    def reset(self):
        obs, info = self.env.reset()
        self.kf = self._build_kalman()
        frame = self.env.render()
        return self._extract_features(frame)

    def step(self, action):
        # 1) 卡尔曼预测
        _ = self.kf.predict()
        # 2) 执行动作
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 3) 渲染一帧 RGB 数组
        frame = self.env.render()
        # 4) YOLO 推理
        r = self.model(frame, conf=0.3, imgsz=(640,448), verbose=False)[0]
        # 5) 如果可视化，用同一个 OpenCV 窗口画检测框
        if self.launch_env:
            ann = r.plot()  # 画好框的 RGB 图
            bgr = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.win_name, bgr)
            cv2.waitKey(1)
        # 6) 特征提取 & 卡尔曼校正
        state = self._extract_features(frame, r)
        done = terminated or truncated
        return state, reward, done

    def _extract_features(self, frame, result=None):
        # 如果没有传 result，就再推一次
        if result is None:
            result = self.model(frame, conf=0.3, imgsz=(640,448), verbose=False)[0]

        raw = result.obb.data.cpu().numpy() if getattr(result, 'obb', None) is not None else np.zeros((0,7))
        obb_all = raw[:, :5]
        cls_all = raw[:, 6].astype(int) if raw.size else np.zeros((0,), int)

        # 用 lander cls=0 更新 kalman
        idx_l = np.where(cls_all == 0)[0]
        if idx_l.size:
            cx, cy = obb_all[idx_l[0], :2]
            self.kf.correct(np.array([[cx],[cy]], dtype=np.float32))

        # landing_point cls=1
        idx_p = np.where(cls_all == 1)[0]
        if idx_p.size:
            x_p, y_p = obb_all[idx_p[0], :2]
        else:
            x_p, y_p = 0.0, 0.0

        st = self.kf.statePost.flatten()
        return np.array([st[0], st[1], st[2], st[3], 0.0, 0.0, x_p, y_p],
                        dtype=np.float32)

    def close(self):
        if self.launch_env:
            cv2.destroyAllWindows()
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="YOLO .pt 权重路径")
    parser.add_argument("--fps",     type=float, default=5.0)
    parser.add_argument("--gravity", type=float, default=-3.5)
    parser.add_argument("--render",  action="store_true",
                        help="是否打开可视化窗口")
    args = parser.parse_args()

    env = FeatureEnv(
        model_path=args.model,
        title="LunarLander-v3",
        fps=args.fps,
        gravity=args.gravity,
        launch_env=args.render
    )
    s = env.reset()
    print("Initial state:", s)
    for _ in range(5):
        s, r, done = env.step(0)
        print("Next state, reward:", s, r)
        if done: break
    env.close()
