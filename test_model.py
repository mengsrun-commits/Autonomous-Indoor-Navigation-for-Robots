import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math
from stable_baselines3 import PPO

# ==========================================
# NEW MAZE ENVIRONMENT
# ==========================================
class MazeRobotEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(MazeRobotEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # MAP SETTINGS
        self.width = 600
        self.height = 600
        self.robot_radius = 15
        self.speed = 5
        self.turn_speed = 0.15 
        
        # --- NEW OBSTACLES (THE MAZE) ---
        # Format: (x, y, width, height)
        self.obstacles = [
            # 1. Outer Walls
            (0, 0, 600, 10), (0, 590, 600, 10), (0, 0, 10, 600), (590, 0, 10, 600),
            
            # 2. The "S" Shape Maze Walls
            (0, 150, 450, 20),   # Long wall from left
            (150, 300, 450, 20), # Long wall from right
            (0, 450, 450, 20),   # Long wall from left again
            
            # 3. Some tricky pillars to dodge
            (500, 50, 50, 50),
            (50, 520, 50, 50)
        ]
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at top left (safe zone)
        self.x = 50
        self.y = 50
        self.angle = 1.57 # Face down
        self.steps = 0
        self.sensor_readings = self._get_sensors()
        return np.array(self.sensor_readings, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        
        if action == 0: # Forward
            self.x += math.cos(self.angle) * self.speed
            self.y += math.sin(self.angle) * self.speed
            reward = 1.0 
        elif action == 1: # Left
            self.angle -= self.turn_speed
        elif action == 2: # Right
            self.angle += self.turn_speed

        # Collision & Safety
        if self._check_collision(self.x, self.y):
            reward = -100
            terminated = True
        
        self.sensor_readings = self._get_sensors()
        if min(self.sensor_readings) < 0.2: reward -= 5 

        truncated = self.steps >= 1000 # Longer time limit for maze
        if self.render_mode == "human": self.render()

        return np.array(self.sensor_readings, dtype=np.float32), reward, terminated, truncated, {}

    # --- REUSED PHYSICS HELPERS ---
    def _check_collision(self, x, y):
        for ox, oy, w, h in self.obstacles:
            if ox <= x <= ox+w and oy <= y <= oy+h: return True
        return False

    def _cast_ray(self, angle_offset):
        ray_angle = self.angle + angle_offset
        for dist in range(0, 200, 5):
            tx = self.x + math.cos(ray_angle) * dist
            ty = self.y + math.sin(ray_angle) * dist
            if self._check_collision(tx, ty): return dist / 200
        return 1.0

    def _get_sensors(self):
        return [self._cast_ray(-0.78), self._cast_ray(0), self._cast_ray(0.78)]

    def render(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Draw Maze Walls
        for ox, oy, w, h in self.obstacles:
            cv2.rectangle(img, (int(ox), int(oy)), (int(ox+w), int(oy+h)), (255, 255, 255), -1)
        # Draw Robot
        rx, ry = int(self.x), int(self.y)
        cv2.circle(img, (rx, ry), self.robot_radius, (0, 255, 255), -1) # Yellow Robot
        # Draw Whiskers
        angles = [-0.78, 0, 0.78]
        for i, ang in enumerate(angles):
            dist = self.sensor_readings[i] * 200
            ex = rx + math.cos(self.angle + ang) * dist
            ey = ry + math.sin(self.angle + ang) * dist
            color = (0, 255, 0) if self.sensor_readings[i] > 0.3 else (0, 0, 255)
            cv2.line(img, (rx, ry), (int(ex), int(ey)), color, 2)
            
        cv2.imshow("Maze Test", img)
        cv2.waitKey(20)

# ==========================================
# RUN THE TEST
# ==========================================
if __name__ == '__main__':
    # 1. Create the Hard Maze Env
    env = MazeRobotEnv(render_mode="human")
    
    # 2. Load your OLD brain
    try:
        # Ensure 'ppo_2d_robot.zip' is in the same folder!
        model = PPO.load("ppo_2d_robot")
        print("Brain Loaded! Attempting Maze Run...")
    except:
        print("Error: Could not find 'ppo_2d_robot.zip'. Run the training script first!")
        exit()

    # 3. Watch it run
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            print("Resetting...")
            obs, _ = env.reset()