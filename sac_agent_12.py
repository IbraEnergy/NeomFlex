import os
import logging
import shutil
import random
from collections import deque
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim):
#         super(Actor, self).__init__()
#         self.log_std_min = -20
#         self.log_std_max = 2
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.mean = nn.Linear(hidden_dim, action_dim)
#         self.log_std = nn.Linear(hidden_dim, action_dim)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        #print(f"Actor init - state_dim: {state_dim}, action_dim: {action_dim}")  # Debug
        self.log_std_min = -20
        self.log_std_max = 2
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # Should be exact state dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    # def forward(self, state):
    #     x = torch.relu(self.fc1(state))
    #     x = torch.relu(self.fc2(x))
    #     mean = self.mean(x)
    #     log_std = self.log_std(x)
    #     log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
    #     return mean, log_std
    def forward(self, state):
        # print(f"Actor forward - Input state shape: {state.shape}")
        # print(f"FC1 weight shape: {self.fc1.weight.shape}")
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        q = self.q(x)
        return q

class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[idx] for idx in batch])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=1_000_000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor network
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic networks
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Target critics
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Parameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        
        
        # self.agents = {}
        # for agent_id in env.territories:
        #     # Calculate state dim from observation space
        #     state_dim = env.observation_spaces[agent_id].shape[0]  # This will include time
        #     action_dim = env.action_spaces[agent_id].shape[0]
            
        #     print(f"Agent {agent_id} - State dim: {state_dim}, Action dim: {action_dim}")  # Debug
            
        #     self.agents[agent_id] = SACAgent(
        #         state_dim=state_dim,
        #         action_dim=action_dim,
        #         hidden_dim=256
        #     )
        
        

    # def select_action(self, state, evaluate=False):
    #     state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
    #     if evaluate:
    #         action, _ = self.actor.sample(state)
    #     else:
    #         action, _ = self.actor.sample(state)
    #     return action.detach().cpu().numpy()[0]
    def select_action(self, state, evaluate=False):
        # Ensure state has correct shape before sending to network
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        state = state.to(self.device)
        
        # Ensure state has correct batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        # Debug print
        #print(f"State shape before actor: {state.shape}")  # Should be (1, state_dim)
        
        if evaluate:
            action, _ = self.actor.sample(state)
        else:
            action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        # Update critics
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target_value)
        critic2_loss = nn.MSELoss()(current_q2, target_value)

        # Update critic networks
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }, filename)

    # def load(self, filename):
    #     checkpoint = torch.load(filename)
    #     self.actor.load_state_dict(checkpoint['actor'])
    #     self.critic1.load_state_dict(checkpoint['critic1'])
    #     self.critic2.load_state_dict(checkpoint['critic2'])
    #     self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    #     self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
    #     self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
    
    def load(self, filename):
        checkpoint = torch.load(filename, weights_only=True)  # Add weights_only=True
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])

class DataRecorder:
    def __init__(self, env, save_dir='results/data'):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.current_episode = 0
        self.episode_data = []
        self.pv_buses = env.pv_buses
        self.bat_buses = env.bat_buses
        self.territories = env.territories

    def record_step(self, env, actions, rewards, info, step):
        record = {
            'step': step,
            'load': env.data.iloc[env.time_step]['Demand'],
            'solar': env.data.iloc[env.time_step]['Solar'],
            'total_losses': env.net.res_line.pl_mw.sum()
        }
        
        # Record voltages
        for i, v in enumerate(env.net.res_bus.vm_pu.values):
            record[f'voltage_bus_{i}'] = v
            
        # Record battery states
        for bus in self.bat_buses:
            sgen_idx = env.bat_sgen_indices[bus]
            record[f'bat_{bus}_power'] = env.net.sgen.at[sgen_idx, 'p_mw']
            record[f'bat_{bus}_soc'] = env.bat_soc[bus]
            
        # Record PV states
        for bus in self.pv_buses:
            sgen_idx = env.pv_sgen_indices[bus]
            record[f'pv_{bus}_p'] = env.net.sgen.at[sgen_idx, 'p_mw']
            record[f'pv_{bus}_q'] = env.net.sgen.at[sgen_idx, 'q_mvar']
            
        # Record actions and rewards
        for agent_id in actions:
            record[f'action_{agent_id}'] = str(actions[agent_id].tolist())
            record[f'reward_{agent_id}'] = rewards[agent_id]
            
        self.episode_data.append(record)

    def save_episode(self):
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            df.to_csv(os.path.join(self.save_dir, f'episode_{self.current_episode}.csv'), index=False)
            self.episode_data = []
            self.current_episode += 1

    def get_action_summary(self, actions, env):
            """Get concise action summary for logging."""
            if not actions:  # Handle case where no actions available
                return "No actions available"
                
            summary = []
            for bus in self.bat_buses:
                agent = self.bat_buses[bus]
                if agent in actions:  # Check if agent has actions
                    action_idx = env.agent_assets[agent]['action_map'][f'bat_{bus}']
                    power = actions[agent][action_idx]
                    summary.append(f"B{bus}:{power:.2f}")
                    
            for bus in self.pv_buses:
                agent = self.pv_buses[bus]
                if agent in actions:  # Check if agent has actions
                    action_idx = env.agent_assets[agent]['action_map'][f'pv_{bus}']
                    var = actions[agent][action_idx]
                    summary.append(f"PV{bus}:{var:.2f}")
                    
            return " ".join(summary) if summary else "No actions available"

# class Trainer:
#     def __init__(self, env, episodes=1000, batch_size=256, max_steps=48, save_dir='results'):
class Trainer:
    def __init__(self, env, episodes=1000, batch_size=256, max_steps=48, save_dir='results', curriculum_start=300):
        self.env = env
        self.episodes = episodes
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.save_dir = save_dir
        self.curriculum_start = curriculum_start #NEW
        self.current_episode = 0  #NEW
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        self.model_dir = os.path.join(save_dir, 'models')
        self.log_dir = os.path.join(save_dir, 'logs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize agents with correct dimensions
        self.agents = {}
        for agent_id in env.territories:
            # Get test observation to determine actual dimensions
            test_obs = env._get_obs(agent_id, np.ones(len(env.net.bus)))
            state_dim = len(test_obs)  # This will get actual observation size
            action_dim = env.action_spaces[agent_id].shape[0]
            
            #print(f"Initializing {agent_id} network with state_dim={state_dim}, action_dim={action_dim}")
            
            self.agents[agent_id] = SACAgent(
                state_dim=state_dim,  # Use actual observation size
                action_dim=action_dim,
                hidden_dim=256
            )
        
        
        
        # # Initialize agents
        # self.agents = {
        #     agent_id: SACAgent(
        #         state_dim=env.observation_spaces[agent_id].shape[0],
        #         action_dim=env.action_spaces[agent_id].shape[0]
        #     )
        #     for agent_id in env.territories
        # }
        
        

        
        
        # Initialize metrics
        self.total_rewards = {agent_id: [] for agent_id in self.agents}
        self.average_rewards = {agent_id: [] for agent_id in self.agents}
        self.voltage_violations = []
        
        # Initialize recorder
        self.recorder = DataRecorder(env, os.path.join(save_dir, 'data'))
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        self.log_file = os.path.join(self.log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

    def train(self):
        logging.info("Starting training...")
        start_time = datetime.now()
        
        try:
            for episode in range(self.episodes):
                self.run_episode(episode)
                
                if (episode + 1) % 10 == 0:
                    self.log_progress(episode)
                
                if (episode + 1) % 100 == 0:
                    self.save_checkpoint(episode)
        except KeyboardInterrupt:
            logging.info("\nTraining interrupted by user.")
            self.save_results('interrupted')
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise
        finally:
            self.save_results('final')
            training_time = datetime.now() - start_time
            logging.info(f"\nTraining completed in {training_time}")
    
    # def run_episode(self, episode):
    #     obs = self.env.reset()
    #     episode_rewards = {agent_id: 0 for agent_id in self.agents}
    #     done = {agent_id: False for agent_id in self.agents}
    #     step = 0

    #     while not all(done.values()) and step < self.max_steps:
    #         # Get actions
    #         actions = {}
    #         for agent_id, agent in self.agents.items():
    #             if not done[agent_id]:
    #                 actions[agent_id] = agent.select_action(obs[agent_id])
            
    #         # Environment step
    #         next_obs, rewards, done_flags, info = self.env.step(actions)
            
    #         # Record data
    #         self.recorder.record_step(self.env, actions, rewards, info, step)
            
            # Update agents
    def run_episode(self, episode):
            obs = self.env.reset()
            episode_rewards = {agent_id: 0 for agent_id in self.agents}
            done = {agent_id: False for agent_id in self.agents}
            step = 0
            self.current_episode = episode  #NEW
            self.env.use_solar_penalty = episode >= self.curriculum_start  #NEW
    
            while not all(done.values()) and step < self.max_steps:
                # Get actions
                actions = {}
                for agent_id, agent in self.agents.items():
                    if not done[agent_id]:
                        actions[agent_id] = agent.select_action(obs[agent_id])
                
                # Store current actions for logging
                self.current_actions = actions
                
                # Environment step
                next_obs, rewards, done_flags, info = self.env.step(actions)
                
                # Record data
                self.recorder.record_step(self.env, actions, rewards, info, step)
            
                
                # Update agents
                for agent_id, agent in self.agents.items():
                    if not done[agent_id]:
                        # Store transition
                        agent.replay_buffer.push(
                            obs[agent_id],
                            actions[agent_id],
                            rewards[agent_id],
                            next_obs[agent_id],
                            float(done_flags[agent_id])
                        )
                        
                        # Update if enough samples
                        if len(agent.replay_buffer) > self.batch_size:
                            agent.update(self.batch_size)
                        
                        episode_rewards[agent_id] += rewards[agent_id]
                
                # Update state
                obs = next_obs
                done = done_flags
                step += 1
            
            # Save episode data
            self.recorder.save_episode()
            
            # Update metrics
            for agent_id in self.agents:
                self.total_rewards[agent_id].append(episode_rewards[agent_id])
                recent_rewards = self.total_rewards[agent_id][-100:]
                self.average_rewards[agent_id].append(np.mean(recent_rewards))
            
            # Check voltage violations
            self.voltage_violations.append(
                any(v < 0.95 or v > 1.05 for v in self.env.net.res_bus.vm_pu.values)
            )
    def log_progress(self, episode):
            logging.info(f"\nEpisode {episode + 1}/{self.episodes}")
            for agent_id in self.agents:
                logging.info(
                    f"  {agent_id} - "
                    f"Reward: {self.total_rewards[agent_id][-1]:.2f}, "
                    f"Avg(100): {self.average_rewards[agent_id][-1]:.2f}"
                )
            logging.info(f"  Voltage Violation: {self.voltage_violations[-1]}")
            # Only show action summary if we have actions stored
            if hasattr(self, 'current_actions'):
                logging.info(f"  Actions Summary: {self.recorder.get_action_summary(self.current_actions, self.env)}")
            logging.info("-" * 50)

    def save_checkpoint(self, episode):
        checkpoint_dir = os.path.join(self.model_dir, f'checkpoint_{episode + 1}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save models
        for agent_id, agent in self.agents.items():
            agent.save(os.path.join(checkpoint_dir, f'{agent_id}.pt'))
        
        # Save metrics
        np.savez(
            os.path.join(checkpoint_dir, 'metrics.npz'),
            total_rewards=self.total_rewards,
            average_rewards=self.average_rewards,
            voltage_violations=self.voltage_violations
        )

    def save_results(self, tag='final'):
        results_dir = os.path.join(self.save_dir, f'{tag}_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save models
        for agent_id, agent in self.agents.items():
            agent.save(os.path.join(results_dir, f'{agent_id}.pt'))
        
        # Save all metrics
        np.savez(
            os.path.join(results_dir, 'training_metrics.npz'),
            total_rewards=self.total_rewards,
            average_rewards=self.average_rewards,
            voltage_violations=self.voltage_violations
        )
        
        # Copy log file
        if hasattr(self, 'log_file'):
            shutil.copy2(self.log_file, os.path.join(results_dir, 'training.log'))

if __name__ == "__main__":
    from grid_env_14 import GridEnv
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create environment
    env = GridEnv(data_path='Training.csv')
    
    # Create and run trainer
    trainer = Trainer(
        env=env,
        episodes=5000,
        batch_size=256,
        max_steps=48,
        save_dir='results'
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving results...")
    finally:
        print("\nTraining completed. Data saved to results directory.")