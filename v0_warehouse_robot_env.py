'''
Custom Gym environment for the warehouse robot problem
This module wraps the WarehouseRobot class to make it compatible with Gymnasium's interface
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

Key Components:
- Environment: Gymnasium-compatible wrapper for WarehouseRobot
- Action Space: Discrete actions (LEFT, DOWN, RIGHT, UP)
- Observation Space: Box space representing robot and target positions
- Reward: 1 for reaching target, 0 otherwise
- Termination: When robot reaches target
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import v0_warehouse_robot as wr  # Import the base warehouse robot implementation
import numpy as np

# Register this environment with Gymnasium
# This allows using gym.make('warehouse-robot-v0') to create instances
register(
    id='warehouse-robot-v0',                                # Unique identifier for the environment
    entry_point='v0_warehouse_robot_env:WarehouseRobotEnv', # Module and class to use
)

class WarehouseRobotEnv(gym.Env):
    """
    Gymnasium environment wrapper for the warehouse robot problem
    Implements the required Gymnasium interface methods
    """
    
    # Required metadata for Gymnasium environments
    metadata = {
        "render_modes": ["human"],  # Supported rendering modes
        'render_fps': 4             # Frames per second for rendering
    }

    def __init__(self, grid_rows=4, grid_cols=5, render_mode=None):
        """
        Initialize the environment
        Args:
            grid_rows: Number of rows in the grid
            grid_cols: Number of columns in the grid
            render_mode: Rendering mode ('human' or None)
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode

        # Initialize the underlying warehouse robot problem
        self.warehouse_robot = wr.WarehouseRobot(
            grid_rows=grid_rows, 
            grid_cols=grid_cols, 
            fps=self.metadata['render_fps']
        )

        # Define action space: discrete actions (LEFT, DOWN, RIGHT, UP)
        self.action_space = spaces.Discrete(len(wr.RobotAction))

        # Define observation space: [robot_row, robot_col, target_row, target_col]
        self.observation_space = spaces.Box(
            low=0,  # Minimum position (top-left corner)
            high=np.array([self.grid_rows-1, self.grid_cols-1, self.grid_rows-1, self.grid_cols-1]),  # Maximum position
            shape=(4,),  # 4-dimensional observation
            dtype=np.int32  # Integer positions
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        Args:
            seed: Random seed for reproducible initial states
            options: Additional options (not used)
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)  # Required by Gymnasium for random seed control

        # Reset the warehouse robot
        self.warehouse_robot.reset(seed=seed)

        # Construct observation: [robot_row, robot_col, target_row, target_col]
        obs = np.concatenate((
            self.warehouse_robot.robot_pos,    # Robot position
            self.warehouse_robot.target_pos    # Target position
        ))
        
        # Additional information (empty in this case)
        info = {}

        # Render if in human mode
        if self.render_mode == 'human':
            self.render()

        return obs, info

    def step(self, action):
        """
        Execute one time step in the environment
        Args:
            action: Action to take (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP)
        Returns:
            observation: New state after action
            reward: Reward for the action
            terminated: Whether episode is done
            truncated: Whether episode was truncated (not used)
            info: Additional information
        """
        # Execute action and check if target was reached
        target_reached = self.warehouse_robot.perform_action(wr.RobotAction(action))

        # Determine reward and termination
        reward = 0
        terminated = False
        if target_reached:
            reward = 1
            terminated = True

        # Construct new observation
        obs = np.concatenate((
            self.warehouse_robot.robot_pos,    # Updated robot position
            self.warehouse_robot.target_pos    # Target position (unchanged)
        ))

        # Additional information (empty in this case)
        info = {}

        # Render if in human mode
        if self.render_mode == 'human':
            print(wr.RobotAction(action))  # Print action for debugging
            self.render()

        return obs, reward, terminated, False, info

    def render(self):
        """
        Render the current state of the environment
        """
        self.warehouse_robot.render()

# Test the environment if run directly
if __name__ == "__main__":
    # Create environment with human rendering
    env = gym.make('warehouse-robot-v0', render_mode='human')

    # Optional: Check if environment follows Gymnasium interface
    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    # Reset environment and get initial state
    obs = env.reset()[0]

    # Take random actions
    while True:
        # Sample random action
        rand_action = env.action_space.sample()
        
        # Execute action
        obs, reward, terminated, _, _ = env.step(rand_action)

        # Reset if episode is done
        if terminated:
            obs = env.reset()[0]
