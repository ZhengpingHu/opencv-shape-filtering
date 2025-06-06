from stable_baselines3 import DQN

# 加载模型
model = DQN.load("trained/dqn_feature_env.zip")

# 使用 FeatureEnv 作为环境
from env_yolo_kalman import FeatureEnv
env = FeatureEnv(model_path="best.pt", title="Replay", fps=5, gravity=-3.5, launch_env=True)

# 回放一条轨迹
for i in range(5):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done = env.step(action)
    print(reward)
    env.close()
