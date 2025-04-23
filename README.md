# Warehouse-Robot-Pathfinding

This project simulates a warehouse robot that must navigate a 2D grid to reach a randomly placed target using various AI/ML strategies. It compares classical RL (Q-Learning, SARSA), modern deep RL (PPO, A2C, DQN), and an MDP-based value iteration approach.

---

## 📦 Features

✅ Custom OpenAI Gym environment using Pygame  
✅ Trains and evaluates 7 models:  
- Q-Learning  
- SARSA  
- A2C (SB3)  
- PPO (SB3)  
- DQN (SB3)  
- Policy Gradient (PPO)  
- MDP (Value Iteration)

✅ Performance Comparison Charts:
- Average Time to Goal (log scale)
- Success Count over 100 runs

---

## 🧠 Environment Structure

- 4x5 grid
- One robot (start at [0,0])
- One target (randomly placed)
- 4 actions: UP, DOWN, LEFT, RIGHT


# 1. Install dependencies
pip install gymnasium stable-baselines3 pygame matplotlib

# 2. (Optional) Enable GPU
# Requires CUDA-compatible setup for A2C, PPO, DQN speedup

# 3. Run training
python model.py                            # to train all models

# 4. Run evaluation
python all.py                              # Evaluate all models w/ GUI
python multi.py                            # Benchmark all models (100x) and generate graphs
