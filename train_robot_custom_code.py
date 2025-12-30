import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math
import time

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Optimized Training on: {device}")

# ==========================================
# 1. OPTIMIZED ROBOT ENVIRONMENT
# ==========================================
class Robot2DEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(Robot2DEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # MAP SETTINGS
        self.width = 600
        self.height = 600
        self.robot_radius = 15
        self.speed = 4
        self.turn_speed = 0.3
        
        self.obstacles = [
            (200, 200, 100, 100), (100, 400, 50, 150), (400, 100, 150, 50),
            (-10, -10, 620, 20), (-10, 590, 620, 20), (-10, -10, 20, 620), (590, -10, 20, 620)
        ]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Loop until we find a safe spot
        safe = False
        while not safe:
            # 1. Pick a random spot anywhere on the map
            # (We stay away from the absolute edges by robot_radius)
            self.x = np.random.uniform(self.robot_radius, self.width - self.robot_radius)
            self.y = np.random.uniform(self.robot_radius, self.height - self.robot_radius)
            
            # 2. Check if this spot is inside (or too close to) any obstacle
            # We add a 'margin' (e.g., 20px) so it doesn't spawn touching a wall
            if not self._check_collision(self.x, self.y, radius=self.robot_radius + 20):
                safe = True
                
        # 3. Randomize Angle
        self.angle = np.random.uniform(0, 2*math.pi)
        
        self.steps = 0
        self.sensor_readings = self._get_sensors()
        return np.array(self.sensor_readings, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        
        # 1. MOVEMENT
        if action == 0:   
            self.x += math.cos(self.angle) * self.speed
            self.y += math.sin(self.angle) * self.speed
            reward = 1.0 
        elif action == 1: 
            self.angle -= self.turn_speed
            # REMOVED PENALTY: Turning is now free. 
            # We want it to feel free to adjust course.
            reward = 0 
        elif action == 2: 
            self.angle += self.turn_speed
            reward = 0

        # 2. COLLISION
        if self._check_collision(self.x, self.y, self.robot_radius):
            reward = -100
            terminated = True
            
        self.sensor_readings = self._get_sensors()
        
        # --- 3. THE "FEAR GRADIENT" (Crucial Fix) ---
        # Instead of a binary punishment, we punish PROXIMITY.
        # If the robot gets closer than 30% of ray length, it starts losing points.
        # This acts like a "force field" pushing it away.
        
        min_dist = min(self.sensor_readings)
        if min_dist < 0.25:
            # The closer it gets, the higher the penalty.
            # At 0.25 distance: Penalty is 0
            # At 0.1 distance: Penalty is High
            penalty = (0.25 - min_dist) * 2
            reward -= penalty

        truncated = self.steps >= 500
        if self.render_mode == "human": self.render()
        
        return np.array(self.sensor_readings, dtype=np.float32), reward, terminated, truncated, {}

    def _check_collision(self, x, y, radius=0):
        # Optimized: Pre-calculate squares to avoid sqrt() calls
        r2 = radius**2
        for ox, oy, w, h in self.obstacles:
            nx = max(ox, min(x, ox + w))
            ny = max(oy, min(y, oy + h))
            if ((x-nx)**2 + (y-ny)**2) <= r2: return True
        return False

    def _get_sensors(self):
        readings = []
        for ang in [-0.78, 0, 0.78]:
            ray_angle = self.angle + ang
            dist_val = 1.0
            
            # OPTIMIZATION: Step size increased to 15 (3x speedup vs 5)
            # This reduces CPU cycles significantly with minimal accuracy loss
            for d in range(0, 200, 15):
                tx = self.x + math.cos(ray_angle) * d
                ty = self.y + math.sin(ray_angle) * d
                # Ray uses point collision (radius=1)
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
        cv2.circle(img, (rx, ry), self.robot_radius, (255, 0, 0), -1)
        for i, ang in enumerate([-0.78, 0, 0.78]):
            end_x = rx + math.cos(self.angle+ang) * (self.sensor_readings[i]*200)
            end_y = ry + math.sin(self.angle+ang) * (self.sensor_readings[i]*200)
            cv2.line(img, (rx, ry), (int(end_x), int(end_y)), (0, 255, 0), 2)
        cv2.imshow("Robot Simulator", img)
        cv2.waitKey(1)

# ==========================================
# 2. OPTIMIZED PPO BRAIN (GPU + Fast GAE)
# ==========================================0

LEARNING_RATE = 0.0003 # Lower LR is often more stable for PPO
GAMMA = 0.99        
GAE_LAMBDA = 0.95    
EPS_CLIP = 0.2      
K_EPOCHS = 10         # Increased epochs per update to squeeze more juice out of data
UPDATE_TIMESTEP = 2000 # Larger batch size for GPU efficiency

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        # Move state to GPU
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # 1. Bulk Convert Lists to Tensors (Much faster than converting one by one)
        old_states = torch.tensor(np.array(memory.states), dtype=torch.float32).to(device)
        old_actions = torch.tensor(memory.actions, dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(device)
        rewards = memory.rewards
        is_terminals = memory.is_terminals

        # 2. Get Value Predictions
        values = self.policy.critic(old_states).squeeze().detach()
        
        # 3. Optimized GAE Loop (O(N) Complexity)
        advantages = []
        gae = 0
        values_cpu = values.cpu().numpy() # Move to CPU once for the loop
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - is_terminals[t]
                next_value = 0 
            else:
                next_non_terminal = 1.0 - is_terminals[t]
                next_value = values_cpu[t+1]
            
            delta = rewards[t] + GAMMA * next_value * next_non_terminal - values_cpu[t]
            gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
            advantages.append(gae) # Append is O(1), Insert(0) is O(N)

        advantages.reverse() # O(N)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        
        # Returns
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 4. Optimize Policy
        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================
if __name__ == '__main__':
    # Toggle this to watch (but train slower) or not watch (train fast)
    render = False 
    
    env = Robot2DEnv(render_mode="human" if render else None)
    
    state_dim = 3  
    action_dim = 3 
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim)
    
    print("PPO Training Started...")
    start_time = time.time()
    time_step = 0
    
    try:
        for i_episode in range(1, 5001):
            state, _ = env.reset()
            ep_reward = 0
            
            while True:
                time_step +=1
                
                # Get Action
                action, action_logprob = ppo.policy_old.act(state)
                state_new, reward, done, truncated, _ = env.step(action)
                
                # Save to memory
                memory.states.append(state) # Save numpy array directly
                memory.actions.append(action)
                memory.logprobs.append(action_logprob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                state = state_new
                ep_reward += reward
                
                # Update PPO
                if time_step % UPDATE_TIMESTEP == 0:
                    print(f"Update @ Step {time_step} | Last Ep Reward: {ep_reward:.2f}")
                    ppo.update(memory)
                    memory.clear()
                
                if done or truncated:
                    break
            
            if i_episode % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Episode {i_episode} | Avg Time/Ep: {elapsed/50:.3f}s")
                start_time = time.time()

    except KeyboardInterrupt:
        print("\nTraining Interrupted.")
        
    torch.save(ppo.policy.state_dict(), 'ppo_optimized.pth')
    print("Model saved as 'ppo_optimized.pth'")