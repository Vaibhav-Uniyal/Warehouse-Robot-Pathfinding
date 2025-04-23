import os
import pickle
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
import v0_warehouse_robot_env
import matplotlib.pyplot as plt

# Create models directory
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_q_learning():
    """
    Train Q-Learning model
    Q(s,a) = Q(s,a) + α[r + γ max(Q(s',a')) - Q(s,a)]
    where:
    - s: current state
    - a: current action
    - r: reward
    - γ: discount factor
    - α: learning rate
    - s': next state
    - a': next action
    """
    print("\n🎯 Training Q-Learning...")
    
    # Initialize the warehouse robot environment
    env = gym.make("warehouse-robot-v0")  # Create environment instance
    
    # Q-learning hyperparameters
    alpha = 0.1  # Learning rate: controls how much we update Q-values
    gamma = 0.99  # Discount factor: importance of future rewards (0.99 means future rewards are highly valued)
    epsilon = 0.1  # Exploration rate: probability of taking random action
    episodes = 10000  # Number of training episodes
    
    # Initialize Q-table: a matrix of state-action pairs
    # state_space: total number of possible states
    # action_space: total number of possible actions
    state_space = np.prod(env.observation_space.high + 1)  # Calculate total number of states
    action_space = env.action_space.n  # Get number of possible actions
    Q = np.zeros((state_space, action_space))  # Initialize Q-table with zeros
    
    # Training loop: run for specified number of episodes
    for episode in range(episodes):
        # Reset environment and get initial state
        obs, _ = env.reset()  # Reset environment and get initial observation
        state = tuple(obs.astype(int))  # Convert observation to tuple for indexing
        done = False  # Flag to track if episode is complete
        
        # Episode loop: continue until goal is reached or episode is done
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:  # With probability epsilon
                action = env.action_space.sample()  # Take random action (exploration)
            else:
                action = np.argmax(Q[state])  # Take best action (exploitation)
            
            # Execute action and observe results
            next_obs, reward, terminated, truncated, _ = env.step(action)  # Take action
            next_state = tuple(next_obs.astype(int))  # Convert next observation to tuple
            done = terminated or truncated  # Check if episode is done
            
            # Q-learning update rule
            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state][action]
            )
            # Update current state
            state = next_state
    
    # Save trained Q-table to file
    with open(f"{MODELS_DIR}/q_learning.pkl", "wb") as f:
        pickle.dump(Q, f)  # Serialize and save Q-table
    print("✅ Saved Q-Learning model")

def train_sarsa():
    """
    Train SARSA model
    Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
    where:
    - s: current state
    - a: current action
    - r: reward
    - γ: discount factor
    - α: learning rate
    - s': next state
    - a': next action
    """
    print("\n🎯 Training SARSA...")
    
    # Initialize environment
    env = gym.make("warehouse-robot-v0")
    
    # SARSA hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 0.1  # Exploration rate
    episodes = 10000  # Number of training episodes
    
    # Initialize Q-table
    state_space = np.prod(env.observation_space.high + 1)
    action_space = env.action_space.n
    Q = np.zeros((state_space, action_space))
    
    # Training loop
    for episode in range(episodes):
        # Reset environment and get initial state
        obs, _ = env.reset()
        state = tuple(obs.astype(int))
        
        # Choose initial action using epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Random action
        else:
            action = np.argmax(Q[state])  # Best action
        
        done = False
        while not done:
            # Take action and observe results
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_obs.astype(int))
            done = terminated or truncated
            
            # Choose next action using epsilon-greedy
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            # SARSA update rule
            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * Q[next_state][next_action] - Q[state][action]
            )
            
            # Update current state and action
            state = next_state
            action = next_action
    
    # Save trained Q-table
    with open(f"{MODELS_DIR}/sarsa.pkl", "wb") as f:
        pickle.dump(Q, f)
    print("✅ Saved SARSA model")

def train_mdp():
    """
    Train MDP using Value Iteration
    V(s) = maxₐ Σₛ' P(s'|s,a)[R(s,a,s') + γV(s')]
    where:
    - V(s): value of state s
    - P(s'|s,a): transition probability from s to s' given action a
    - R(s,a,s'): reward for transition from s to s' given action a
    - γ: discount factor
    """
    print("\n🎯 Training MDP...")
    env = gym.make("warehouse-robot-v0")
    
    # Define state and action spaces
    state_space = np.prod(env.observation_space.high + 1)  # Total number of states
    action_space = env.action_space.n  # Number of possible actions

    # Helper functions for state conversion
    def state_to_index(state):
        """
        Convert state tuple to index
        state: [robot_row, robot_col, target_row, target_col]
        returns: unique index for the state
        """
        r, c, tr, tc = state
        return (r * 5 + c) * 20 + (tr * 5 + tc)

    def index_to_state(index):
        """
        Convert index back to state tuple
        index: unique state index
        returns: [robot_row, robot_col, target_row, target_col]
        """
        tr, tc = divmod(index % 20, 5)
        r, c = divmod(index // 20, 5)
        return [r, c, tr, tc]

    # Initialize value function
    V = np.zeros(state_space)  # Value function for each state
    gamma = 0.99  # Discount factor
    theta = 1e-4  # Convergence threshold

    # Build transition model
    P = {}  # Transition probabilities
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

    # Value Iteration
    print("Running Value Iteration...")
    while True:
        delta = 0
        for s in range(state_space):
            v = V[s]
            # Update value function
            V[s] = max(sum(prob * (r + gamma * V[s_]) for prob, s_, r, _ in P[s][a]) 
                      for a in range(action_space))
            delta = max(delta, abs(v - V[s]))
        if delta < theta:  # Check for convergence
            break

    # Extract policy from value function
    policy = np.zeros(state_space, dtype=int)
    for s in range(state_space):
        # Calculate Q-values for each action
        q_vals = [sum(prob * (r + gamma * V[s_]) for prob, s_, r, _ in P[s][a]) 
                 for a in range(action_space)]
        # Choose best action
        policy[s] = np.argmax(q_vals)

    # Save policy
    with open(f"{MODELS_DIR}/mdp_policy.pkl", "wb") as f:
        pickle.dump(policy, f)
    print("✅ Saved MDP policy")

def train_deep_rl_models():
    """
    Train deep reinforcement learning models using Stable Baselines3
    """
    def train_sb3_model(model_name, algo):
        """
        Generic function to train any Stable Baselines3 model
        model_name: name of the model
        algo: algorithm class (A2C, PPO, or DQN)
        """
        print(f"\n🎯 Training {model_name}...")
        env = gym.make("warehouse-robot-v0")
        
        # Initialize model with MLP policy
        model = algo("MlpPolicy", env, verbose=0, device="cuda")
        
        # Train model for specified timesteps
        model.learn(total_timesteps=20000)
        
        # Save trained model
        model.save(f"{MODELS_DIR}/{model_name.lower().replace(' ', '_')}_model")
        print(f"✅ Saved {model_name} model")

    # Train A2C (Advantage Actor-Critic)
    # ∇J(θ) = E[∇logπ(a|s) * A(s,a)]
    train_sb3_model("A2C", A2C)

    # Train PPO (Proximal Policy Optimization)
    # L(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
    train_sb3_model("PPO", PPO)

    # Train DQN (Deep Q-Network)
    # L(θ) = E[(r + γ maxₐ' Q(s',a';θ⁻) - Q(s,a;θ))²]
    train_sb3_model("DQN", DQN)

    # Train Policy Gradient (using PPO)
    train_sb3_model("Policy Gradient", PPO)

def main():
    """
    Main function to train all models
    """
    # 1. Train Q-Learning
    train_q_learning()

    # 2. Train SARSA
    train_sarsa()

    # 3. Train MDP
    train_mdp()

    # 4-7. Train Deep RL models
    train_deep_rl_models()

    print("\n🏁 All models trained and saved!")

if __name__ == "__main__":
    main()
