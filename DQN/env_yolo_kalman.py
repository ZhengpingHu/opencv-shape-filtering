#!/usr/bin/env python3
# env_yolo_kalman.py

import argparse
import numpy as np
import cv2
import gymnasium as gym
import torch
from ultralytics import YOLO

#import fixed_env

class FeatureEnv:
    """
    YOLO-OBB + Kalman + Touchdown Category Integration for LunarLander
    输出 8 维特征：[x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact]
    """
    def __init__(self, model_path, title, fps, gravity, launch_env=False):
        self.env = gym.make("LunarLander-v3", render_mode="rgb_array", gravity=-3.5)
        self.frame_interval = 1.0 / fps

        # 加载 YOLO 模型
        self.model = YOLO(model_path, task="detect", verbose=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        # Kalman 滤波器初始化
        self.kf = self._build_kalman()

        # 可视化窗口
        self.launch_env = launch_env
        if self.launch_env:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            self.win_name = title

        # 固定着陆点（环境设定）
        self.pad_x, self.pad_y = 0.0, 0.0

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
        result = self.model(frame, conf=0.7, imgsz=(640,448), verbose=False)[0]
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
            result = self.model(frame, conf=0.7, imgsz=(640,448), verbose=False)[0]

        raw = result.obb.data.cpu().numpy() if getattr(result, 'obb', None) is not None else np.zeros((0,7))
        obb_all = raw[:, :5]
        cls_all = raw[:, 6].astype(int) if raw.size else np.zeros((0,), int)

        # 提取 lander 主体 (class 0)
        idx_l = np.where(cls_all == 0)[0]
        if idx_l.size:
            cx, cy, w, h, raw_ang = obb_all[idx_l[0]]
            self.kf.correct(np.array([[cx],[cy]], dtype=np.float32))

            ang_base = raw_ang + (np.pi/2 if h > w else 0.0)
            cands = [ang_base, -ang_base] + [(c + np.pi) for c in [ang_base, -ang_base]]
            cands = [ (c + np.pi) % (2*np.pi) - np.pi for c in cands ]

            if obs is not None:
                true_ang = obs[4]
                errs = [ abs((c - true_ang + np.pi) % (2*np.pi) - np.pi) for c in cands ]
                ang = cands[int(np.argmin(errs))]
            else:
                ang = cands[0]
        else:
            ang = obs[4] if obs is not None else 0.0

        # 解析触地标签（class 1~3）
        leg1_contact, leg2_contact = 0, 0
        idx_touch = np.where((cls_all >= 1) & (cls_all <= 3))[0]
        if idx_touch.size:
            label = cls_all[idx_touch[0]]
            if label == 1:   # right
                leg1_contact, leg2_contact = 0, 1
            elif label == 2: # left
                leg1_contact, leg2_contact = 1, 0
            elif label == 3: # both
                leg1_contact, leg2_contact = 1, 1

        st = self.kf.statePost.flatten()
        ang_vel = obs[5] if obs is not None else 0.0
        return np.array([
            st[0], st[1], st[2], st[3],
            ang, ang_vel,
            leg1_contact, leg2_contact
        ], dtype=np.float32)

    def close(self):
        if self.launch_env:
            cv2.destroyAllWindows()
        self.env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model',      required=True, help="YOLO .pt path of model")
    p.add_argument('--conf',       type=float, default=0.33, help="conf level")
    p.add_argument('--fps',        type=float, default=5.0, help="max FPS")
    p.add_argument('--title',      default='lunar', help="keyword of window")
    p.add_argument('--launch-env', action='store_true', help="显示带框画面")
    p.add_argument('--gravity',    type=float, default=-3.5, help="环境重力")
    args = p.parse_args()

    env = FeatureEnv(args.model, args.title, args.fps, args.gravity, args.launch_env)
    state = env.reset()
    done = False
    while not done:
        action = env.env.action_space.sample()
        state, reward, done = env.step(action)
    env.close()
