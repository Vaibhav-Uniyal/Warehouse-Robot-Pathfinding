'''
Script to evaluate and visualize the performance of all trained models
This script runs each model in the environment and measures:
- Time taken to reach target
- Success rate
- Visual performance
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
NUM_TRIALS = 1  # Number of trials per model

def get_random_target_position(seed=None):
    """
    Generate a random target position
    Args:
        seed: Optional seed for reproducibility
    Returns:
        list: [row, column] position
    """
    if seed is not None:
        random.seed(seed)
    return [
        random.randint(1, 3),  # Random row (1-3)
        random.randint(1, 4)   # Random column (1-4)
    ]

def evaluate_model(model_name, model_path, seed, target_pos=None):
    """
    Evaluate a single model's performance
    Args:
        model_name: Name of the model
        model_path: Path to saved model
        seed: Random seed for environment
        target_pos: Optional target position to use (for consistency across models)
    Returns:
        tuple: (success, time_taken, steps_taken)
            success: Boolean indicating if target was reached
            time_taken: Time taken to reach target or MAX_TIME if failed
            steps_taken: Number of steps taken
    """
    # Create environment with human rendering
    env = gym.make("warehouse-robot-v0", render_mode="human")
    
    # Reset environment with given seed and target position
    if target_pos is not None:
        env.unwrapped.warehouse_robot.target_pos = target_pos
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset(seed=seed)
        target_pos = env.unwrapped.warehouse_robot.target_pos
    
    done = False
    steps = 0
    start = time.time()  # Start timer

    # Display model name and target position in GUI
    env.unwrapped.warehouse_robot.last_action = f"Running: {model_name}\nTarget: {target_pos}"

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
            env.render()  # Update visualization
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
            env.render()  # Update visualization
            steps += 1

    # Calculate total time
    total_time = time.time() - start
    success = done and steps < MAX_STEPS

    # Print results
    if not success:
        print(f"⚠️ {model_name} stopped after {MAX_STEPS} steps (timeout).")
    print(f"✅ {model_name} took {total_time:.2f} seconds in {steps} steps\n")
    
    env.close()
    return success, total_time, steps

def main():
    """
    Main function to evaluate all models
    """
    # Generate a random seed between 1 and 1000000
    seed = random.randint(1, 1000000)
    random.seed(seed)
    
    print(f"Using random seed: {seed}")
    
    # Initialize results storage
    results = {
        name: {"successes": 0, "times": [], "steps": []} 
        for name in MODEL_PATHS.keys()
    }
    
    # Run multiple trials for each model
    for trial in range(NUM_TRIALS):
        print(f"\n🚀 Trial {trial + 1}/{NUM_TRIALS}")
        
        # Generate a new random target position for this trial
        target_pos = get_random_target_position(seed + trial)
        print(f"Target position for this trial: {target_pos}")
        
        for model_name, model_path in MODEL_PATHS.items():
            try:
                # Evaluate model with same target position
                success, time_taken, steps = evaluate_model(
                    model_name, 
                    model_path, 
                    seed + trial,
                    target_pos
                )
                
                # Update results
                if success:
                    results[model_name]["successes"] += 1
                results[model_name]["times"].append(time_taken)
                results[model_name]["steps"].append(steps)
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
    
    # Calculate statistics
    stats = {}
    for name, data in results.items():
        if data["times"]:
            stats[name] = {
                "success_rate": data["successes"] / NUM_TRIALS,
                "avg_time": np.mean(data["times"]),
                "std_time": np.std(data["times"]),
                "avg_steps": np.mean(data["steps"]),
                "std_steps": np.std(data["steps"])
            }
    
    # Print summary
    print("\n📊 Results Summary:")
    for name, stat in stats.items():
        print(f"{name}:")
        print(f"  Success Rate: {stat['success_rate']:.2%}")
        print(f"  Average Time: {stat['avg_time']:.2f} ± {stat['std_time']:.2f} seconds")
        print(f"  Average Steps: {stat['avg_steps']:.1f} ± {stat['std_steps']:.1f} steps")
    
    # Plot results
    plt.figure(figsize=(15, 6))
    
    # Time plot with color coding
    plt.subplot(1, 2, 1)
    avg_times = [stat["avg_time"] for stat in stats.values()]
    std_times = [stat["std_time"] for stat in stats.values()]
    
    # Determine colors based on performance
    colors = []
    fastest_time = min(avg_times)
    slowest_time = max(avg_times)
    for time, success_rate in zip(avg_times, [stat["success_rate"] for stat in stats.values()]):
        if success_rate < 1.0 or time > MAX_TIME:
            colors.append('gray')  # Non-converged or too slow
        elif time == fastest_time:
            colors.append('green')  # Fastest
        elif time == slowest_time:
            colors.append('red')  # Slowest
        else:
            colors.append('skyblue')  # Middle performance
    
    bars = plt.bar(stats.keys(), avg_times, yerr=std_times, color=colors)
    plt.title("Average Time to Target")
    plt.ylabel("Time (seconds)")
    
    # Add time values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom')
    
    # Steps plot
    plt.subplot(1, 2, 2)
    avg_steps = [stat["avg_steps"] for stat in stats.values()]
    std_steps = [stat["std_steps"] for stat in stats.values()]
    
    bars = plt.bar(stats.keys(), avg_steps, yerr=std_steps, color=colors)
    plt.title("Average Steps to Target")
    plt.ylabel("Number of Steps")
    
    # Add step values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='green', label='Fastest'),
        Patch(facecolor='red', label='Slowest'),
        Patch(facecolor='gray', label='Non-converged/Timeout'),
        Patch(facecolor='skyblue', label='Other')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("model_performance_comparison.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
