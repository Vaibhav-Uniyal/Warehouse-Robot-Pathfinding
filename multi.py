import gymnasium as gym
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO, DQN
from matplotlib.patches import Patch
from v0_warehouse_robot_env import WarehouseRobotEnv

MODEL_PATHS = {
    "Q-Learning": "D:/rl/models/q_learning.pkl",
    "SARSA": "D:/rl/models/sarsa.pkl",
    "A2C": "D:/rl/models/a2c_model.zip",
    "PPO": "D:/rl/models/ppo_model.zip",
    "DQN": "D:/rl/models/dqn_model.zip",
    "Policy Gradient": "D:/rl/models/policy_gradient_model.zip",
    "MDP": "D:/rl/models/mdp_policy.pkl"
}

MAX_STEPS = 30
RUNS = 100

def run_model(name, path):
    times = []
    success_count = 0
    for _ in range(RUNS):
        env = gym.make("warehouse-robot-v0", render_mode=None)
        obs, _ = env.reset()
        done = False
        steps = 0
        start = time.time()

        try:
            if name in ["Q-Learning", "SARSA", "MDP"]:
                with open(path, "rb") as f:
                    q = pickle.load(f)

                while not done and steps < MAX_STEPS:
                    state = tuple(obs.astype(int))
                    if name == "MDP":
                        idx = (state[0] * 5 + state[1]) * 20 + (state[2] * 5 + state[3])
                        action = int(q[idx])
                    else:
                        action = int(q[state].argmax())

                    obs, _, done, _, _ = env.step(action)
                    steps += 1

            else:
                algo = {
                    "A2C": A2C,
                    "PPO": PPO,
                    "DQN": DQN,
                    "Policy Gradient": PPO,
                }[name]
                model = algo.load(path, env=env)

                while not done and steps < MAX_STEPS:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _, _ = env.step(action)
                    steps += 1

        except Exception as e:
            print(f"❌ {name} failed on one run: {e}")
            done = False

        elapsed = time.time() - start
        times.append(elapsed)
        if done:
            success_count += 1

        env.close()

    return times, success_count

# === Run All Models ===
all_times = {}
success_counts = {}
for name, path in MODEL_PATHS.items():
    print(f"▶ Running {name}...")
    times, success = run_model(name, path)
    all_times[name] = times
    success_counts[name] = success

# === Plot 1: Avg Log Time + Std Dev ===
avg_times = {k: np.mean(v) for k, v in all_times.items()}
std_devs = {k: np.std(v) for k, v in all_times.items()}

models = list(avg_times.keys())
avg_vals = [avg_times[m] for m in models]
std_vals = [std_devs[m] for m in models]

colors = []
fastest = min(avg_vals)
worst_model = min(success_counts, key=lambda k: success_counts[k])
for m in models:
    if avg_times[m] == fastest:
        colors.append("green")
    elif m == worst_model:
        colors.append("red")
    else:
        colors.append("skyblue")

plt.figure(figsize=(12, 6))
bars = plt.bar(models, avg_vals, yerr=std_vals, log=True, capsize=5, color=colors)
for bar, val in zip(bars, avg_vals):
    plt.text(bar.get_x() + bar.get_width()/2, val * 1.05, f"{val:.4f}s", ha='center', va='bottom')

plt.title("⏱ Log-Scaled Avg Time per Model (100 runs, ±Std Dev)")
plt.ylabel("Log Time (seconds)")
plt.xlabel("Model")
plt.grid(True, which="both", axis='y')
plt.legend(handles=[
    Patch(color="green", label="Fastest (Best Avg Time)"),
    Patch(color="red", label="Most Failures"),
    Patch(color="skyblue", label="Others")
], title="Legend")
plt.tight_layout()
plt.savefig("avg_time_logscale.png")

# === Plot 2: Success Count ===
plt.figure(figsize=(12, 5))
bars = plt.bar(success_counts.keys(), success_counts.values(), color=[
    "green" if v == max(success_counts.values()) else "red" if v == min(success_counts.values()) else "skyblue"
    for v in success_counts.values()
])
for bar, val in zip(bars, success_counts.values()):
    plt.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val}/100", ha='center', va='bottom')

plt.title("✅ Convergence Accuracy per Model (100 runs)")
plt.ylabel("Success Count")
plt.xlabel("Model")
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig("model_success_count_readable.png")

print("✅ Benchmark done. Graphs saved as:")
print("- avg_time_logscale.png")
print("- model_success_count_readable.png")
