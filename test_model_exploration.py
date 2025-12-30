import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math
from stable_baselines3 import PPO

# ==========================================
# COMPLEX OFFICE ENVIRONMENT (Fixed Physics & Layout)
# ==========================================
class ComplexOfficeEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(ComplexOfficeEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # MAP SETTINGS
        self.width = 800
        self.height = 600
        self.robot_radius = 15 # The physical size of the robot
        self.speed = 5
        self.turn_speed = 0.15 
        
        # --- DEFINE THE OFFICE LAYOUT WITH DOORS ---
        self.obstacles = [
            # 1. OUTER BOUNDARY
            (0, 0, 800, 10), (0, 590, 800, 10), (0, 0, 10, 600), (790, 0, 10, 600),
            
            # 2. VERTICAL DIVIDER (Separates Left/Right Rooms)
            (400, 0, 10, 250),   # Top Vertical
            (400, 350, 10, 250), # Bottom Vertical
            
            # 3. HALLWAY WALLS (With Explicit Door Gaps)
            # Top Wall (y=250) - Gap at x=160-220 and x=580-640
            (10, 250, 150, 10),   # Wall 1
            (220, 250, 180, 10),  # Wall 2 (Door is between 160 and 220)
            (410, 250, 170, 10),  # Wall 3
            (640, 250, 150, 10),  # Wall 4 (Door is between 580 and 640)
            
            # Bottom Wall (y=350) - Symmetric gaps
            (10, 350, 150, 10),
            (220, 350, 180, 10),
            (410, 350, 170, 10),
            (640, 350, 150, 10),
            
            # 4. FURNITURE (Obstacles inside rooms)
            (100, 100, 50, 50),  # Top-Left Desk
            (600, 100, 50, 50),  # Top-Right Desk
            (100, 500, 50, 50),  # Bottom-Left Desk
            (600, 500, 50, 50),  # Bottom-Right Desk
        ]
        
        # Memory Map for Visualization (Blue Trails)
        self.scale = 10 
        self.grid_w = self.width // self.scale
        self.grid_h = self.height // self.scale
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start in the Central Hallway
        self.x = 50
        self.y = 300 
        self.angle = 0.0 # Face Right
        self.steps = 0
        self.memory_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.sensor_readings = self._get_sensors()
        return np.array(self.sensor_readings, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        
        # Movement
        if action == 0:
            self.x += math.cos(self.angle) * self.speed
            self.y += math.sin(self.angle) * self.speed
        elif action == 1: self.angle -= self.turn_speed
        elif action == 2: self.angle += self.turn_speed

        # --- UPDATED COLLISION CHECK ---
        # We use the Circle Collision for the robot body
        if self._check_circle_collision(self.x, self.y):
            terminated = True
            print("CRASHED! (Edge Hit)")
        
        self.sensor_readings = self._get_sensors()
        
        # Visualization Painting
        self._update_map_for_vis()

        truncated = self.steps >= 2000
        if self.render_mode == "human": self.render()

        return np.array(self.sensor_readings, dtype=np.float32), reward, terminated, truncated, {}

    # --- BODY COLLISION (Checks Robot Radius vs Walls) ---
    def _check_circle_collision(self, x, y):
        radius = self.robot_radius
        for ox, oy, w, h in self.obstacles:
            # Find closest point on rectangle to circle center
            closest_x = max(ox, min(x, ox + w))
            closest_y = max(oy, min(y, oy + h))
            
            # Calculate distance from center to that point
            dist_x = x - closest_x
            dist_y = y - closest_y
            
            # If distance < radius, we hit
            if (dist_x**2 + dist_y**2) < (radius**2):
                return True
        return False

    # --- LASER COLLISION (Checks Exact Point vs Walls) ---
    def _check_point_collision(self, x, y):
        for ox, oy, w, h in self.obstacles:
            if ox <= x <= ox+w and oy <= y <= oy+h: return True
        return False

    def _cast_ray(self, angle_offset):
        ray_angle = self.angle + angle_offset
        # Max distance 300
        for dist in range(0, 300, 10):
            tx = self.x + math.cos(ray_angle) * dist
            ty = self.y + math.sin(ray_angle) * dist
            # Lasers use Point Collision
            if self._check_point_collision(tx, ty): return dist / 300.0
        return 1.0

    def _get_sensors(self):
        return [self._cast_ray(-0.78), self._cast_ray(0), self._cast_ray(0.78)]

    # Just for drawing blue trails
    def _update_map_for_vis(self):
        start_grid = (int(self.x/self.scale), int(self.y/self.scale))
        cv2.circle(self.memory_map, start_grid, 1, 1, -1)

    def render(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 1. Draw Blue Trails
        visible_mask = cv2.resize(self.memory_map, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        img[:, :, 0] = visible_mask * 200 
        
        # 2. Draw Obstacles
        for ox, oy, w, h in self.obstacles:
            cv2.rectangle(img, (int(ox), int(oy)), (int(ox+w), int(oy+h)), (255, 255, 255), -1)
            
        # 3. Draw Robot
        rx, ry = int(self.x), int(self.y)
        cv2.circle(img, (rx, ry), self.robot_radius, (0, 255, 255), -1)
        
        # 4. Draw Rays
        for i, ang in enumerate([-0.78, 0, 0.78]):
            dist = self.sensor_readings[i] * 300
            ex = rx + math.cos(self.angle+ang)*dist
            ey = ry + math.sin(self.angle+ang)*dist
            color = (0, 255, 0) if self.sensor_readings[i] > 0.2 else (0, 0, 255)
            cv2.line(img, (rx, ry), (int(ex), int(ey)), color, 2)
            
        cv2.imshow("Complex Office Test", img)
        cv2.waitKey(20)

if __name__ == '__main__':
    # 1. Create Env
    env = ComplexOfficeEnv(render_mode="human")
    
    # 2. Load Model (Using the permissive one which handles doorways best)
    try:
        model = PPO.load("ppo_2d_robot") 
        print("Loaded 'ppo_permissive'! Running Office Simulation...")
    except:
        print("Error: Model not found. Did you run the 'train_permissive_robot.py' script?")
        exit()

    # 3. Run Loop
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        
        if done or truncated:
            print("Resetting...")
            obs, _ = env.reset()