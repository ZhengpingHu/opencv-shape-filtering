#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from collections import deque

# ====================
# Hyperparameters
# ====================
HIDDEN_SIZE = 64
BUFFER_SIZE = 100_000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3                # learning rate (α)
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
NUM_EPISODES = 500
TARGET_UPDATE = 10       # how often to sync target network


# ====================
# Q‐Network
# ====================
class DQN(torch.nn.Module):
    def __init__(self, state_size=8, action_size=4, hidden_size=HIDDEN_SIZE):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_out = torch.nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        return self.layer_out(x)


# ====================
# Replay Buffer
# ====================
class ReplayBuffer:
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.stack(next_states),
                np.array(dones, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)


# ====================
# DQN Agent
# ====================
class DQNAgent:
    def __init__(self, state_size=8, action_size=4):
        self.env = gym.make('LunarLander-v3')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        # Networks & optimizer
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer()

        # Logging
        self.scores = []
        self.best_model_path = "best_model.pth"

    def select_action(self, state, testing=False):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if testing or random.random() > self.epsilon:
            with torch.no_grad():
                q_vals = self.q_network(state_tensor)
            action = int(q_vals.argmax(dim=1).cpu().item())
        else:
            action = random.randrange(self.q_network.layer_out.out_features)
        return action

    def step(self, state, action, reward, next_state, done):
        # store and learn
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) >= self.batch_size:
            self.update_model()

    def update_model(self):
        # sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # to tensors
        states      = torch.from_numpy(states).float().to(self.device)
        actions     = torch.from_numpy(actions).long().to(self.device)
        rewards     = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones       = torch.from_numpy(dones).float().to(self.device)

        # current Q
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # next Q from target
        next_q = self.target_network(next_states).max(1)[0].detach()
        # TD target
        target = rewards + self.gamma * next_q * (1 - dones)

        # loss
        loss = torch.nn.SmoothL1Loss()(q_values, target)

        # ——— 插桩输出 ———
        # print(f"[Batch]  Loss: {loss.item():.4f}  |  "
        #       f"Q_mean: {q_values.mean().item():.3f}  |  "
        #       f"Q_max: {q_values.max().item():.3f}  |  "
        #       f"Buffer: {len(self.memory)}")
        # ————————————

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_networks(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_agent(self, path):
        torch.save(self.q_network.state_dict(), path)
        print(f"[Info] Saved best model to {path}")

    def train(self):
        print("[Training] Start")
        for ep in range(1, NUM_EPISODES + 1):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.step(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            # end of episode
            self.scores.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if ep % TARGET_UPDATE == 0:
                self.sync_networks()

            # episode summary
            avg_last100 = np.mean(self.scores[-100:])
            print(f"Episode {ep:3d} | Reward: {total_reward:6.1f} | "
                  f"Avg100: {avg_last100:6.2f} | ε: {self.epsilon:.3f}")

        print("[Training] Finished")
        self.env.close()

    def plot_training_progress(self, window=100):
        plt.figure()
        plt.plot(self.scores, label="Episode Reward")
        # rolling mean
        rolling = [np.mean(self.scores[max(0, i-window+1):i+1]) for i in range(len(self.scores))]
        plt.plot(rolling, label=f"{window}-episode avg")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Progress")
        plt.legend()
        plt.show()

    def test(self, num_episodes=20):
        print("[Testing] Start")
        total_rewards = []
        for i in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self.select_action(state, testing=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
            total_rewards.append(ep_reward)
            print(f"Test Ep {i:2d} | Reward: {ep_reward:6.1f}")
        avg = np.mean(total_rewards)
        print(f"[Testing] Average Reward over {num_episodes} episodes: {avg:.2f}")
        self.env.close()


if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()
    agent.save_agent(agent.best_model_path)
    agent.plot_training_progress()
    agent.test()
