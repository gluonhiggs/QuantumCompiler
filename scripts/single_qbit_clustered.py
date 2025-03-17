import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.her import HerReplayBuffer
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def rotation_gate(axis, angle):
    if axis == 'x':
        return np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)],
                         [-1j * np.sin(angle / 2), np.cos(angle / 2)]], dtype=complex)
    elif axis == 'y':
        return np.array([[np.cos(angle / 2), -np.sin(angle / 2)],
                         [np.sin(angle / 2), np.cos(angle / 2)]], dtype=complex)
    elif axis == 'z':
        return np.array([[np.exp(-1j * angle / 2), 0],
                         [0, np.exp(1j * angle / 2)]], dtype=complex)

gate_descriptions = ["rxp", "rxn", "ryp", "ryn", "rzp", "rzn"]
gate_matrices = [
    rotation_gate('x', np.pi / 128),
    rotation_gate('x', -np.pi / 128),
    rotation_gate('y', np.pi / 128),
    rotation_gate('y', -np.pi / 128),
    rotation_gate('z', np.pi / 128),
    rotation_gate('z', -np.pi / 128)
]

def random_su2_matrix():
    alpha = np.random.uniform(0, 2 * np.pi)
    beta = np.random.uniform(0, np.pi)
    gamma = np.random.uniform(0, 2 * np.pi)
    z1 = np.array([[np.exp(-1j * alpha / 2), 0], [0, np.exp(1j * alpha / 2)]], dtype=complex)
    y = np.array([[np.cos(beta / 2), -np.sin(beta / 2)], [np.sin(beta / 2), np.cos(beta / 2)]], dtype=complex)
    z2 = np.array([[np.exp(-1j * gamma / 2), 0], [0, np.exp(1j * gamma / 2)]], dtype=complex)
    return np.dot(z1, np.dot(y, z2))

class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, tolerance):
        super().__init__()
        self.gate_set = gate_set
        self.tolerance = tolerance
        self.action_space = spaces.Discrete(len(self.gate_set))
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        })
        self.max_steps = 130
        self.axis_map = {0: 'x', 1: 'x', 2: 'y', 3: 'y', 4: 'z', 5: 'z'}
        self.last_axis = None
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        self.target_U = random_su2_matrix()
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        self.last_axis = None
        return self._get_observation(), {}
    
    def step(self, action):
        gate = self.gate_set[action]
        self.U_n = np.dot(self.U_n, gate)
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        
        reward = 0.0 if fidelity >= self.tolerance else -1.0
        current_axis = self.axis_map[int(action)]
        if self.last_axis is not None and self.last_axis != current_axis:
            reward -= 2.0
        else:
            reward += 0.2
        
        self.last_axis = current_axis
        self.current_step += 1
        done = (fidelity >= self.tolerance) or (self.current_step >= self.max_steps)
        if done and fidelity >= self.tolerance:
            reward += 5 * (self.max_steps - self.current_step)
        
        obs = self._get_observation()
        info = {}
        truncated = False
        return obs, reward, done, truncated, info
    
    def _get_observation(self):
        obs = np.concatenate([self.O_n.real.flatten(), self.O_n.imag.flatten()]).astype(np.float32)
        achieved_goal = np.concatenate([self.U_n.real.flatten(), self.U_n.imag.flatten()]).astype(np.float32)
        desired_goal = np.concatenate([self.target_U.real.flatten(), self.target_U.imag.flatten()]).astype(np.float32)
        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }
    
    def average_gate_fidelity(self, U, V):
        diff = U - V
        singular_values = np.linalg.svd(diff, compute_uv=False)
        return 1 - np.max(singular_values)
    
    def compute_reward(self, achieved_goals, desired_goals, info):
        n_samples = achieved_goals.shape[0]
        rewards = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            # Reconstruct 2x2 complex matrix from 8D real array
            U_n = np.array([
                [achieved_goals[i][0] + 1j * achieved_goals[i][1], achieved_goals[i][2] + 1j * achieved_goals[i][3]],
                [achieved_goals[i][4] + 1j * achieved_goals[i][5], achieved_goals[i][6] + 1j * achieved_goals[i][7]]
            ], dtype=complex)
            
            U_target = np.array([
                [desired_goals[i][0] + 1j * desired_goals[i][1], desired_goals[i][2] + 1j * desired_goals[i][3]],
                [desired_goals[i][4] + 1j * desired_goals[i][5], desired_goals[i][6] + 1j * desired_goals[i][7]]
            ], dtype=complex)
            
            fidelity = self.average_gate_fidelity(U_n, U_target)
            rewards[i] = 0.0 if fidelity >= self.tolerance else -1.0
        
        return rewards

policy_kwargs = dict(
    net_arch=[128, 128],
    activation_fn=nn.SELU,
)

class PlottingCallback(BaseCallback):
    def __init__(self, verbose=0, save_path=None):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.save_path = save_path

    def _on_step(self):
        if self.locals.get('dones')[0]:
            episode_info = self.locals.get('infos')[0].get('episode')
            if episode_info:
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
        return True

    def _on_training_end(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, label="Episode Reward")
        plt.title("Episode Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths, label="Episode Length", color='orange')
        plt.title("Episode Length Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.legend()
        plt.tight_layout()
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, "dqn_her_su2.png"))
        plt.close()

def evaluate_agent(model, env, num_episodes=10):
    success_count = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        gate_sequence = []
        target_U = env.target_U
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            gate_sequence.append(action)
            obs, reward, done, _, _ = env.step(action)
        fidelity = env.average_gate_fidelity(env.U_n, target_U)
        sequence_length = len(gate_sequence)
        transitions = sum(1 for i in range(1, len(gate_sequence)) if env.axis_map[int(gate_sequence[i])] != env.axis_map[int(gate_sequence[i-1])])
        print(f"Target U:\n{target_U}")
        print(f"Final Fidelity: {fidelity:.4f}")
        print(f"Sequence Length: {sequence_length}")
        print(f"Axis Transitions: {transitions}")
        if fidelity >= env.tolerance:
            success_count += 1
            print("Success! Gate Sequence:", [gate_descriptions[action] for action in gate_sequence])
        else:
            print("Failed to approximate the target unitary.")
    success_rate = success_count / num_episodes
    return success_rate

if __name__ == "__main__":
    env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=0.98)
    env = Monitor(env)
    model = DQN(
        'MultiInputPolicy',
        env,
        learning_rate=0.0005,
        batch_size=1000,
        train_freq=(1, 'episode'),
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.99976,
        learning_starts=500000,
        target_update_interval=20000,
        buffer_size=100000,
        verbose=0,
        device='cuda',  # Change to 'cpu' if no GPU
        policy_kwargs=policy_kwargs,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            goal_selection_strategy='future',
            n_sampled_goal=4,
        )  # Removed 'online_sampling'
    )
    plotting_callback = PlottingCallback(save_path='./data')
    model.learn(total_timesteps=500000000, log_interval=100, callback=plotting_callback)
    
    success_rate = evaluate_agent(model, env)
    print(f'Success rate: {success_rate * 100:.2f}%')