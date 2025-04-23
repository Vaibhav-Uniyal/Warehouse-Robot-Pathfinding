'''
Script to evaluate multiple models in parallel and visualize their performance
This script runs each model multiple times and measures:
- Success rate
- Average time to reach target
- Performance comparison across models
'''
import time  # For measuring execution time
import random  # For setting random seeds
import gymnasium as gym  # For environment
import pickle  # For loading saved models
import matplotlib.pyplot as plt  # For visualization
import numpy as np  # For numerical operations
from stable_baselines3 import DQN  # For loading DQN model
from v0_warehouse_robot_env import WarehouseRobotEnv  # Custom environment
from matplotlib.patches import Patch  # For legend

# Paths to saved models
MODEL_PATHS = {
    "Q-Learning": "D:/rl/models/q_learning.pkl",  # Q-learning model
    "SARSA": "D:/rl/models/sarsa.pkl",            # SARSA model
    "DQN": "D:/rl/models/dqn_model.zip",          # DQN model
    "MDP": "D:/rl/models/mdp_policy.pkl"          # MDP policy
}

# Evaluation parameters
MAX_STEPS = 30  # Maximum steps per episode
MAX_TIME = 10   # Maximum time in seconds before timeout
NUM_TRIALS = 100  # Number of trials per model

def evaluate_model(model_name, model_path, seed):
    """
    Evaluate a single model's performance
    Args:
        model_name: Name of the model
        model_path: Path to saved model
        seed: Random seed for environment
    Returns:
        tuple: (success, time_taken)
            success: Boolean indicating if target was reached
            time_taken: Time taken to reach target or MAX_TIME if failed
    """
    # Create environment without rendering for speed
    env = gym.make("warehouse-robot-v0", render_mode=None)
    
    # Reset environment with given seed
    obs, _ = env.reset(seed=seed)
    done = False
    steps = 0
    start = time.time()  # Start timer

    # Load and run appropriate model
    if model_name in ["Q-Learning", "SARSA", "MDP"]:
        # Load tabular models (Q-learning, SARSA, MDP)
        with open(model_path, "rb") as f:
            q_table = pickle.load(f)

        while not done and steps < MAX_STEPS:
            # Convert state to tuple for indexing
            state = tuple(obs.astype(int))
            
            if model_name == "MDP":
                # Convert state to index for MDP policy
                state_index = (state[0] * 5 + state[1]) * 20 + (state[2] * 5 + state[3])
                action = int(q_table[state_index])
            else:
                # Get best action from Q-table
                action = int(q_table[state].argmax())

            # Execute action
            obs, _, done, _, _ = env.step(action)
            steps += 1

    else:
        # Load and run DQN model
        algo_class = {
            "DQN": DQN
        }[model_name]

        model = algo_class.load(model_path, env=env)

        while not done and steps < MAX_STEPS:
            # Get action from DQN model
            action, _ = model.predict(obs, deterministic=True)
            # Execute action
            obs, _, done, _, _ = env.step(action)
            steps += 1

    # Calculate total time
    total_time = time.time() - start
    success = done and steps < MAX_STEPS

    # Print progress indicator
    if trial % 10 == 0:  # Print every 10 trials
        print(f"Trial {trial}/{NUM_TRIALS} - {model_name}: {'✅' if success else '❌'}")
    
    env.close()
    return success, total_time

def main():
    """
    Main function to evaluate all models multiple times
    """
    # Set random seed for reproducibility
    seed = random.randint(1, 1000000)  # Random seed for variety
    random.seed(seed)
    print(f"Using seed: {seed}")
    
    # Initialize results storage
    results = {
        name: {"successes": 0, "times": []} 
        for name in MODEL_PATHS.keys()
    }
    
    # Run multiple trials for each model
    global trial  # Make trial count available for progress printing
    for trial in range(NUM_TRIALS):
        for model_name, model_path in MODEL_PATHS.items():
            try:
                # Evaluate model with unique seed for each trial
                success, time_taken = evaluate_model(
                    model_name, 
                    model_path, 
                    seed + trial
                )
                
                # Update results
                if success:
                    results[model_name]["successes"] += 1
                results[model_name]["times"].append(time_taken)
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
    
    # Calculate statistics
    stats = {}
    for name, data in results.items():
        if data["times"]:
            stats[name] = {
                "success_rate": data["successes"] / NUM_TRIALS,
                "avg_time": np.mean(data["times"]),
                "std_time": np.std(data["times"])
            }
    
    # Print summary
    print("\n📊 Results Summary:")
    for name, stat in stats.items():
        print(f"{name}:")
        print(f"  Success Rate: {stat['success_rate']:.2%}")
        print(f"  Average Time: {stat['avg_time']:.2f} ± {stat['std_time']:.2f} seconds")
    
    # === Plot 1: Log-Scaled Average Time ===
    plt.figure(figsize=(12, 5))
    avg_times = [stat["avg_time"] for stat in stats.values()]
    std_times = [stat["std_time"] for stat in stats.values()]
    
    # Determine colors based on performance
    colors = []
    fastest_time = min(avg_times)
    worst_model = min(stats.items(), key=lambda x: x[1]["success_rate"])[0]
    for name, stat in stats.items():
        if stat["avg_time"] == fastest_time:
            colors.append("green")  # Fastest model
        elif name == worst_model:
            colors.append("red")    # Most failures
        else:
            colors.append("skyblue")  # Others
    
    bars = plt.bar(stats.keys(), avg_times, yerr=std_times, capsize=5, color=colors)
    
    # Add time values on top of bars
    for bar, val in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, val * 1.05,
                f"{val:.4f}s", ha='center', va='bottom')
    
    plt.title("⏱ Log-Scaled Avg Time per Model (100 runs, ±Std Dev)")
    plt.ylabel("Log Time (seconds)")
    plt.yscale('log')  # Set log scale for y-axis
    plt.grid(True, which="both", axis='y')
    
    # Add legend
    plt.legend(handles=[
        Patch(color="green", label="Fastest (Best Avg Time)"),
        Patch(color="red", label="Most Failures"),
        Patch(color="skyblue", label="Others")
    ], title="Legend")
    
    plt.tight_layout()
    plt.savefig("avg_time_logscale.png")
    
    # === Plot 2: Success Count ===
    plt.figure(figsize=(12, 5))
    success_counts = {name: data["successes"] for name, data in results.items()}
    
    # Color coding: green for best, red for worst
    colors = ['green' if v == max(success_counts.values()) else 
             'red' if v == min(success_counts.values()) else 'skyblue'
             for v in success_counts.values()]
    
    bars = plt.bar(success_counts.keys(), success_counts.values(), color=colors)
    
    # Add success count values on top of bars
    for bar, val in zip(bars, success_counts.values()):
        plt.text(bar.get_x() + bar.get_width()/2, val + 1,
                f"{val}/{NUM_TRIALS}", ha='center', va='bottom')
    
    plt.title("✅ Convergence Accuracy per Model (100 runs)")
    plt.ylabel("Success Count")
    plt.xlabel("Model")
    plt.ylim(0, 105)  # Set y-axis limit to show values above bars
    
    plt.tight_layout()
    plt.savefig("model_success_count_readable.png")
    
    plt.show()

if __name__ == "__main__":
    main()
