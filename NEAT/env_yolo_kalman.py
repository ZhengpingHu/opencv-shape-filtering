#!/usr/bin/env python3
# env_yolo_kalman.py

import argparse
import numpy as np
import cv2
import gymnasium as gym
import torch
from ultralytics import YOLO

class FeatureEnv:
    """
    环境包装：将 YOLO-OBB + 卡尔曼滤波 与 Gym LunarLander 结合，
    返回 8 维特征：[x, y, vx, vy, angle, angular_velocity, x_p, y_p]
    并在 launch_env=True 时使用 OpenCV 窗口显示带框画面。
    """
    def __init__(self, model_path, title, fps, gravity, launch_env=False):
        self.env = gym.make(
            "LunarLander-v3",
            render_mode="rgb_array",
            gravity=gravity
        )
        self.frame_interval = 1.0 / fps

        self.model = YOLO(model_path, task="detect", verbose=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        self.kf = self._build_kalman()
        self.launch_env = launch_env
        if self.launch_env:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            self.win_name = title

    def _build_kalman(self):
        kf = cv2.KalmanFilter(4, 2, 0)
        kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost        = np.eye(4, dtype=np.float32)
        kf.statePost           = np.zeros((4,1), dtype=np.float32)
        return kf

    def reset(self):
        obs, info = self.env.reset()
        self.kf = self._build_kalman()
        frame = self.env.render()
        return self._extract_features(frame, None, obs)

    def step(self, action):
        _ = self.kf.predict()
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        result = self.model(frame, conf=0.3, imgsz=(640,448), verbose=False)[0]
        if self.launch_env:
            ann = result.plot()
            bgr = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.win_name, bgr)
            cv2.waitKey(1)
        state = self._extract_features(frame, result, obs)
        done = terminated or truncated
        return state, reward, done

    def _extract_features(self, frame, result=None, obs=None):
        if result is None:
            result = self.model(frame, conf=0.3, imgsz=(640,448), verbose=False)[0]
        raw = result.obb.data.cpu().numpy() if getattr(result, 'obb', None) is not None else np.zeros((0,7))
        obb_all = raw[:, :5]
        cls_all = raw[:, 6].astype(int) if raw.size else np.zeros((0,), int)

        # 更新 Kalman 位置
        idx_l = np.where(cls_all == 0)[0]
        if idx_l.size:
            cx, cy = obb_all[idx_l[0], :2]
            self.kf.correct(np.array([[cx],[cy]], dtype=np.float32))
        # 着陆点
        idx_p = np.where(cls_all == 1)[0]
        if idx_p.size:
            x_p, y_p = obb_all[idx_p[0], :2]
        else:
            x_p, y_p = 0.0, 0.0

        st = self.kf.statePost.flatten()
        angle   = obs[4] if obs is not None else 0.0
        ang_vel = obs[5] if obs is not None else 0.0
        return np.array([st[0], st[1], st[2], st[3], angle, ang_vel, x_p, y_p], dtype=np.float32)

    def close(self):
        if self.launch_env:
            cv2.destroyAllWindows()
        self.env.close()
