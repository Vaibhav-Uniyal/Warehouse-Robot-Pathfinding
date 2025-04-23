import os
import pickle
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
import v0_warehouse_robot_env
from sarsatrain import train_sarsa
from v0_warehouse_robot_train import run_q

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# === 1. Q-Learning ===
print("\n🎯 Training Q-Learning...")
run_q(episodes=10000, is_training=True, render=False)
os.rename("v0_warehouse_solution.pkl", f"{MODELS_DIR}/q_learning.pkl")
print("✅ Saved Q-Learning model")

# === 2. SARSA ===
print("\n🎯 Training SARSA...")
train_sarsa(episodes=10000)
os.rename("sarsa.pkl", f"{MODELS_DIR}/sarsa.pkl")
print("✅ Saved SARSA model")

# === 3. MDP ===
print("\n🎯 Training MDP...")
env = gym.make("warehouse-robot-v0")
state_space = np.prod(env.observation_space.high + 1)
action_space = env.action_space.n

def state_to_index(state):
    r, c, tr, tc = state
    return (r * 5 + c) * 20 + (tr * 5 + tc)

def index_to_state(index):
    tr, tc = divmod(index % 20, 5)
    r, c = divmod(index // 20, 5)
    return [r, c, tr, tc]

V = np.zeros(state_space)
gamma = 0.9
theta = 1e-4
P = {}
for s in range(state_space):
    state = index_to_state(s)
    env.unwrapped.warehouse_robot.robot_pos = state[:2]
    env.unwrapped.warehouse_robot.target_pos = state[2:]
    P[s] = {}
    for a in range(action_space):
        env.unwrapped.warehouse_robot.robot_pos = state[:2]
        done = env.unwrapped.warehouse_robot.perform_action(v0_warehouse_robot_env.wr.RobotAction(a))
        new_state = env.unwrapped.warehouse_robot.robot_pos + env.unwrapped.warehouse_robot.target_pos
        new_index = state_to_index(new_state)
        reward = 1 if done else 0
        P[s][a] = [(1.0, new_index, reward, done)]

policy = np.zeros(state_space, dtype=int)
while True:
    delta = 0
    for s in range(state_space):
        v = V[s]
        V[s] = max(sum(prob * (r + gamma * V[s_]) for prob, s_, r, _ in P[s][a]) for a in range(action_space))
        delta = max(delta, abs(v - V[s]))
    if delta < theta:
        break

for s in range(state_space):
    q_vals = [sum(prob * (r + gamma * V[s_]) for prob, s_, r, _ in P[s][a]) for a in range(action_space)]
    policy[s] = np.argmax(q_vals)

with open(f"{MODELS_DIR}/mdp_policy.pkl", "wb") as f:
    pickle.dump(policy, f)
print("✅ Saved MDP policy")

# === 4–7. Deep RL (SB3) ===
def train_sb3_model(model_name, algo):
    print(f"\n🎯 Training {model_name}...")
    env = gym.make("warehouse-robot-v0")
    model = algo("MlpPolicy", env, verbose=0, device="cuda")
    model.learn(total_timesteps=20000)
    model.save(f"{MODELS_DIR}/{model_name.lower().replace(' ', '_')}_model")
    print(f"✅ Saved {model_name} model")

train_sb3_model("A2C", A2C)
train_sb3_model("PPO", PPO)
train_sb3_model("DQN", DQN)
train_sb3_model("Policy Gradient", PPO)  # Same as PPO, just separate tag

print("\n🏁 All models trained and saved!")
