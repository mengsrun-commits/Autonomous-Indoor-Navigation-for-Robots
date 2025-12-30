import torch
import torch.nn as nn
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import cv2
import math
from torch.distributions import Categorical

# ==========================================
# 1. DEFINE THE MODEL ARCHITECTURE
# (Must match the training code EXACTLY)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # ACTOR
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # CRITIC (Not strictly needed for testing, but required to load state_dict)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self.actor(state)
        # We pick the action with the highest probability (Deterministic)
        # instead of sampling (Stochastic) for best testing performance
        action = torch.argmax(action_probs) 
        return action.item()

# ==========================================
# 2. DEFINE THE ENVIRONMENT 
# (Copy-Paste of your Robot2DEnv)
# ==========================================
class Robot2DEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(Robot2DEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        self.width = 600
        self.height = 600
        self.robot_radius = 15
        self.speed = 5
        self.turn_speed = 0.15 
        
        self.obstacles = [
            (200, 200, 100, 100), (100, 400, 50, 150), (400, 100, 150, 50),
            (-10, -10, 620, 20), (-10, 590, 620, 20), (-10, -10, 20, 620), (590, -10, 20, 620)
        ]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 100
        self.y = 100
        self.angle = np.random.uniform(0, 2*math.pi)
        self.steps = 0
        self.sensor_readings = self._get_sensors()
        return np.array(self.sensor_readings, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        
        if action == 0:   
            self.x += math.cos(self.angle) * self.speed
            self.y += math.sin(self.angle) * self.speed
        elif action == 1: self.angle -= self.turn_speed
        elif action == 2: self.angle += self.turn_speed

        if self._check_collision(self.x, self.y, self.robot_radius):
            terminated = True
            print("CRASHED!")
            
        self.sensor_readings = self._get_sensors()
        truncated = self.steps >= 1000 # Longer run for testing
        if self.render_mode == "human": self.render()
        return np.array(self.sensor_readings, dtype=np.float32), reward, terminated, truncated, {}

    def _check_collision(self, x, y, radius=0):
        for ox, oy, w, h in self.obstacles:
            nx = max(ox, min(x, ox + w))
            ny = max(oy, min(y, oy + h))
            if ((x-nx)**2 + (y-ny)**2) <= radius**2: return True
        return False

    def _get_sensors(self):
        readings = []
        for ang in [-0.78, 0, 0.78]:
            ray_angle = self.angle + ang
            dist_val = 1.0
            for d in range(0, 200, 5):
                tx = self.x + math.cos(ray_angle) * d
                ty = self.y + math.sin(ray_angle) * d
                if self._check_collision(tx, ty, radius=1):
                    dist_val = d / 200.0
                    break
            readings.append(dist_val)
        return readings

    def render(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for ox, oy, w, h in self.obstacles:
            cv2.rectangle(img, (int(ox), int(oy)), (int(ox+w), int(oy+h)), (255, 255, 255), -1)
        rx, ry = int(self.x), int(self.y)
        cv2.circle(img, (rx, ry), self.robot_radius, (0, 255, 255), -1) # Yellow Robot for Test Mode
        for i, ang in enumerate([-0.78, 0, 0.78]):
            end_x = rx + math.cos(self.angle+ang) * (self.sensor_readings[i]*200)
            end_y = ry + math.sin(self.angle+ang) * (self.sensor_readings[i]*200)
            color = (0, 255, 0) if self.sensor_readings[i] > 0.2 else (0, 0, 255)
            cv2.line(img, (rx, ry), (int(end_x), int(end_y)), color, 2)
        cv2.imshow("Robot Simulator - TEST MODE", img)
        cv2.waitKey(20) # Slower for better viewing

# ==========================================
# 3. MAIN TEST LOOP
# ==========================================
if __name__ == '__main__':
    # 1. Setup Environment
    env = Robot2DEnv(render_mode="human")
    
    # 2. Load Model
    state_dim = 3
    action_dim = 3
    policy = ActorCritic(state_dim, action_dim)
    
    filename = 'ppo_gae_weights.pth'
    
    try:
        policy.load_state_dict(torch.load(filename))
        policy.eval() # Set to evaluation mode (freezes layers)
        print(f"Loaded {filename} successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Train the model first!")
        exit()

    # 3. Run Loop
    episodes = 5
    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        print(f"Starting Test Episode {i+1}...")
        
        while not done:
            # Get Action from Neural Network
            action = policy.act(obs)
            
            # Step Environment
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                done = True
                obs, _ = env.reset()
                
    print("Testing Finished.")
    cv2.destroyAllWindows()