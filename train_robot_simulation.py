import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math
from stable_baselines3 import PPO

# ==========================================
# CUSTOM 2D PHYSICS SIMULATOR
# ==========================================
class Robot2DEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(Robot2DEnv, self).__init__()
        self.render_mode = render_mode
        
        # ACTIONS: 0=Forward, 1=Left, 2=Right
        # (Removed backward to force it to learn turning)
        self.action_space = spaces.Discrete(3)
        
        # OBSERVATION: [Left_Dist, Center_Dist, Right_Dist]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # MAP SETTINGS
        self.width = 600
        self.height = 600
        self.robot_radius = 15
        self.speed = 5
        self.turn_speed = 0.15 # Radians (approx 8 degrees)
        
        # Define Obstacles (Rectangles: x, y, w, h)
        self.obstacles = [
            (200, 200, 100, 100), # Center Box
            (100, 400, 50, 150),  # Bottom Left
            (400, 100, 150, 50),  # Top Right
            (0, 0, 600, 10),      # Top Wall
            (0, 590, 600, 10),    # Bottom Wall
            (0, 0, 10, 600),      # Left Wall
            (590, 0, 10, 600)     # Right Wall
        ]
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start in a safe open spot
        self.x = 100
        self.y = 100
        self.angle = np.random.uniform(0, 2*math.pi)
        self.steps = 0
        
        # Get initial sensor readings
        self.sensor_readings = self._get_sensors()
        return np.array(self.sensor_readings, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        
        # --- 1. MOVE ROBOT ---
        if action == 0:   # Forward
            self.x += math.cos(self.angle) * self.speed
            self.y += math.sin(self.angle) * self.speed
            reward = 1.0 # Encouragement to move
            
        elif action == 1: # Left
            self.angle -= self.turn_speed
            reward = -0.1 # Slight penalty
            
        elif action == 2: # Right
            self.angle += self.turn_speed
            reward = -0.1

        # --- 2. CHECK COLLISIONS (Body hitting wall) ---
        if self._check_collision(self.x, self.y):
            reward = -100
            terminated = True
            
        # --- 3. GET SENSORS ---
        self.sensor_readings = self._get_sensors()
        
        # Safety Reward: If too close to wall, penalize
        if min(self.sensor_readings) < 0.2:
            reward -= 5 

        # Truncate if stuck in a loop
        truncated = self.steps >= 500
        
        if self.render_mode == "human":
            self.render()

        return np.array(self.sensor_readings, dtype=np.float32), reward, terminated, truncated, {}

    # --- PHYSICS HELPERS ---
    def _check_collision(self, x, y):
        # Check if circle overlaps with any box
        for ox, oy, w, h in self.obstacles:
            # simple AABB collision for center point
            if ox <= x <= ox+w and oy <= y <= oy+h:
                return True
        return False

    def _cast_ray(self, angle_offset):
        # Raycast math to find distance to nearest wall
        ray_angle = self.angle + angle_offset
        max_dist = 200 # Sensors see up to 200 pixels
        step_size = 5
        
        for dist in range(0, max_dist, step_size):
            tx = self.x + math.cos(ray_angle) * dist
            ty = self.y + math.sin(ray_angle) * dist
            
            if self._check_collision(tx, ty):
                return dist / max_dist # Normalize 0.0 to 1.0
                
        return 1.0 # Clear path

    def _get_sensors(self):
        # Left (-45 deg), Center (0 deg), Right (+45 deg)
        left = self._cast_ray(-0.78) 
        center = self._cast_ray(0)
        right = self._cast_ray(0.78)
        return [left, center, right]

    # --- DRAWING ---
    def render(self):
        # 1. Background
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 2. Draw Obstacles (White)
        for ox, oy, w, h in self.obstacles:
            cv2.rectangle(img, (int(ox), int(oy)), (int(ox+w), int(oy+h)), (255, 255, 255), -1)
            
        # 3. Draw Robot (Blue Circle)
        rx, ry = int(self.x), int(self.y)
        cv2.circle(img, (rx, ry), self.robot_radius, (255, 0, 0), -1)
        
        # 4. Draw Rays (Visual Whiskers)
        # We re-calculate end points just for drawing
        angles = [-0.78, 0, 0.78]
        readings = self.sensor_readings
        
        for i, ang in enumerate(angles):
            dist = readings[i] * 200
            end_x = rx + math.cos(self.angle + ang) * dist
            end_y = ry + math.sin(self.angle + ang) * dist
            
            # Color: Green if safe, Red if close
            color = (0, 255, 0)
            if readings[i] < 0.3: color = (0, 0, 255)
            
            cv2.line(img, (rx, ry), (int(end_x), int(end_y)), color, 2)

        # Show Window
        cv2.imshow("Robot Simulator", img)
        cv2.waitKey(10)

# ==========================================
# MAIN TRAINING LOOP
# ==========================================
if __name__ == '__main__':
    mode = input("Select Mode (1=Train, 2=Watch): ")

    if mode == "1":
        # --- TRAIN FAST (No Graphics) ---
        # 1. Create environment WITHOUT human render mode
        env = Robot2DEnv(render_mode=None) 
        
        # 2. Setup Model
        model = PPO("MlpPolicy", env, verbose=1)
        
        print("Training fast... (Should take ~2 minutes)")
        model.learn(total_timesteps=100_000)
        
        model.save("ppo_2d_robot")
        print("Training Complete! Saved as 'ppo_2d_robot.zip'")

    elif mode == "2":
        # --- WATCH RESULT (Graphics On) ---
        # 1. Create environment WITH human render mode
        env = Robot2DEnv(render_mode="human")
        
        # 2. Load the trained brain
        try:
            model = PPO.load("ppo_2d_robot")
            print("Loaded Brain! Watching performance...")
        except:
            print("Error: No model found. Train first!")
            exit()

        # 3. Run a loop to watch it
        obs, _ = env.reset()
        while True:
            # Predict action using the trained brain
            action, _ = model.predict(obs)
            
            # Move the robot
            obs, reward, done, truncated, _ = env.step(action)
            
            if done or truncated:
                obs, _ = env.reset()