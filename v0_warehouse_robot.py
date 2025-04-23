'''
This module models the problem to be solved. In this very simple example, the problem is to optimize a Robot that works in a Warehouse.
The Warehouse is divided into a rectangular grid. A Target is randomly placed on the grid and the Robot's goal is to reach the Target.

Key Components:
- Grid: 2D space where the robot and target are placed
- Robot: Agent that can move in 4 directions (LEFT, DOWN, RIGHT, UP)
- Target: Goal position that the robot needs to reach
- Actions: Discrete movements the robot can take
- States: Positions of robot and target on the grid
'''
import random  # For random target placement
from enum import Enum  # For creating enumerated constants
import pygame  # For visualization
import sys  # For system operations
from os import path  # For file path operations

# Define possible actions the Robot can take
class RobotAction(Enum):
    LEFT=0   # Move left (decrease column)
    DOWN=1   # Move down (increase row)
    RIGHT=2  # Move right (increase column)
    UP=3     # Move up (decrease row)

# Define types of tiles that can appear on the grid
class GridTile(Enum):
    _FLOOR=0  # Empty space
    ROBOT=1   # Robot position
    TARGET=2  # Target position

    # Return the first letter of tile name for console display
    def __str__(self):
        return self.name[:1]

class WarehouseRobot:
    """
    Main class representing the warehouse robot environment
    Manages the grid, robot position, target position, and visualization
    """

    def __init__(self, grid_rows=4, grid_cols=5, fps=1):
        """
        Initialize the warehouse environment
        Args:
            grid_rows: Number of rows in the grid
            grid_cols: Number of columns in the grid
            fps: Frames per second for visualization
        """
        self.grid_rows = grid_rows  # Number of rows in grid
        self.grid_cols = grid_cols  # Number of columns in grid
        self.reset()  # Initialize robot and target positions

        self.fps = fps  # Visualization speed
        self.last_action = ''  # Track last action for display
        self._init_pygame()  # Initialize visualization

    def _init_pygame(self):
        """
        Initialize Pygame for visualization
        Sets up display window, fonts, and loads sprites
        """
        pygame.init()  # Initialize pygame
        pygame.display.init()  # Initialize display module

        # Set up game clock for controlling frame rate
        self.clock = pygame.time.Clock()

        # Set up fonts for displaying action information
        self.action_font = pygame.font.SysFont("Calibre", 30)
        self.action_info_height = self.action_font.get_height()

        # Define cell dimensions for visualization
        self.cell_height = 64  # Height of each grid cell
        self.cell_width = 64   # Width of each grid cell
        self.cell_size = (self.cell_width, self.cell_height)        

        # Calculate window size based on grid dimensions
        self.window_size = (
            self.cell_width * self.grid_cols,  # Total width
            self.cell_height * self.grid_rows + self.action_info_height  # Total height
        )

        # Create game window
        self.window_surface = pygame.display.set_mode(self.window_size) 

        # Load and resize sprites for visualization
        # Robot sprite
        file_name = path.join(path.dirname(__file__), "sprites/bot_blue.png")
        img = pygame.image.load(file_name)
        self.robot_img = pygame.transform.scale(img, self.cell_size)

        # Floor sprite
        file_name = path.join(path.dirname(__file__), "sprites/floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = pygame.transform.scale(img, self.cell_size)

        # Target sprite
        file_name = path.join(path.dirname(__file__), "sprites/package.png")
        img = pygame.image.load(file_name)
        self.goal_img = pygame.transform.scale(img, self.cell_size)

    def reset(self, seed=None):
        """
        Reset the environment to initial state
        Args:
            seed: Random seed for reproducible target placement
        """
        # Set robot to starting position (top-left corner)
        self.robot_pos = [0, 0]

        # Place target at random position
        random.seed(seed)  # Set random seed if provided
        self.target_pos = [
            random.randint(1, self.grid_rows-1),  # Random row
            random.randint(1, self.grid_cols-1)   # Random column
        ]

    def perform_action(self, robot_action: RobotAction) -> bool:
        """
        Execute a robot action and update its position
        Args:
            robot_action: Action to perform (LEFT, DOWN, RIGHT, UP)
        Returns:
            bool: True if robot reached target, False otherwise
        """
        self.last_action = robot_action  # Store action for display

        # Update robot position based on action
        if robot_action == RobotAction.LEFT:
            if self.robot_pos[1] > 0:  # Check left boundary
                self.robot_pos[1] -= 1
        elif robot_action == RobotAction.RIGHT:
            if self.robot_pos[1] < self.grid_cols-1:  # Check right boundary
                self.robot_pos[1] += 1
        elif robot_action == RobotAction.UP:
            if self.robot_pos[0] > 0:  # Check top boundary
                self.robot_pos[0] -= 1
        elif robot_action == RobotAction.DOWN:
            if self.robot_pos[0] < self.grid_rows-1:  # Check bottom boundary
                self.robot_pos[0] += 1

        # Check if robot reached target
        return self.robot_pos == self.target_pos

    def render(self):
        """
        Render the current state of the environment
        Displays grid, robot, target, and current action
        """
        # Print current state to console
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if [r,c] == self.robot_pos:
                    print(GridTile.ROBOT, end=' ')
                elif [r,c] == self.target_pos:
                    print(GridTile.TARGET, end=' ')
                else:
                    print(GridTile._FLOOR, end=' ')
            print()  # New line after each row
        print()  # Extra new line

        # Process user input events
        self._process_events()

        # Clear screen to white
        self.window_surface.fill((255,255,255))

        # Draw grid
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                # Calculate position for this cell
                pos = (c * self.cell_width, r * self.cell_height)
                
                # Draw floor
                self.window_surface.blit(self.floor_img, pos)

                # Draw target if at this position
                if [r,c] == self.target_pos:
                    self.window_surface.blit(self.goal_img, pos)

                # Draw robot if at this position
                if [r,c] == self.robot_pos:
                    self.window_surface.blit(self.robot_img, pos)
                
        # Display current action
        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0,0,0), (255,255,255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)       

        # Update display
        pygame.display.update()
                
        # Control frame rate
        self.clock.tick(self.fps)

    def _process_events(self):
        """
        Handle user input events
        Processes window close and escape key events
        """
        for event in pygame.event.get():
            # Handle window close
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle key presses
            if event.type == pygame.KEYDOWN:
                # Handle escape key
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

# Test the environment if run directly
if __name__ == "__main__":
    warehouseRobot = WarehouseRobot()
    warehouseRobot.render()

    while True:
        # Take random actions for testing
        rand_action = random.choice(list(RobotAction))
        print(rand_action)

        warehouseRobot.perform_action(rand_action)
        warehouseRobot.render()