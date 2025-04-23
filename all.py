import time
import random
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN, A2C
from v0_warehouse_robot_env import WarehouseRobotEnv
from matplotlib.patches import Patch

# Model paths
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
MAX_TIME = 10  # seconds to mark non-converged

# Evaluate a model
def evaluate_model(model_name, model_path, seed):
    env = gym.make("warehouse-robot-v0", render_mode="human")
    obs, _ = env.reset(seed=seed)
    done = False
    steps = 0
    start = time.time()

    # Show label in GUI
    env.unwrapped.warehouse_robot.last_action = f"Running: {model_name}"

    if model_name in ["Q-Learning", "SARSA", "MDP"]:
        with open(model_path, "rb") as f:
            q_table = pickle.load(f)

        while not done and steps < MAX_STEPS:
            state = tuple(obs.astype(int))
            if model_name == "MDP":
                state_index = (state[0] * 5 + state[1]) * 20 + (state[2] * 5 + state[3])
                action = int(q_table[state_index])
            else:
                action = int(q_table[state].argmax())

            obs, _, done, _, _ = env.step(action)
            env.render()
            steps += 1

    else:
        algo_class = {
            "A2C": A2C,
            "PPO": PPO,
            "DQN": DQN,
            "Policy Gradient": PPO
        }[model_name]

        model = algo_class.load(model_path, env=env)

        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            env.render()
            steps += 1

    total_time = time.time() - start

    if not done:
        print(f"⚠️ {model_name} stopped after {MAX_STEPS} steps (timeout).")
    print(f"✅ {model_name} took {total_time:.2f} seconds\n")
    env.close()
    return total_time

# Shared random target seed
shared_seed = random.randint(0, 9999)

# Run models
results = {}
for name, path in MODEL_PATHS.items():
    print(f"\n🚀 Running {name}...")
    try:
        t = evaluate_model(name, path, shared_seed)
        results[name] = t
    except Exception as e:
        print(f"❌ Failed to run {name}: {e}")
        results[name] = None

# Filter valid
results = {k: v for k, v in results.items() if v is not None}
if not results:
    print("❌ All models failed.")
    exit()

# Sort by time
sorted_items = sorted(results.items(), key=lambda x: x[1])
labels = [k for k, _ in sorted_items]
values = [v for _, v in sorted_items]

# Assign colors
colors = []
for v in values:
    if v >= MAX_TIME:
        colors.append("gray")
    elif v == min([val for val in values if val < MAX_TIME]):
        colors.append("green")
    else:
        colors.append("red")

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, values, color=colors)

# Labels on bars
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{val:.2f}s", ha='center', va='bottom')

# Legend
legend_items = [
    Patch(color="green", label="Fastest"),
    Patch(color="red", label="Slower Models"),
    Patch(color="gray", label="Did Not Converge (Timeout)")
]
plt.legend(handles=legend_items, title="Legend", loc="upper right")

plt.title("⏱ Time Taken by Each RL Model to Reach the Target")
plt.ylabel("Time (seconds)")
plt.xlabel("Model")
plt.ylim(0, max(values) + 2)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("model_performance_comparison.png")
plt.show()
