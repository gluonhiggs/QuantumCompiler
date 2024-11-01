import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN, HER
import torch.nn as nn
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import matplotlib.pyplot as plt

# Define rotation gates (same as your original code)
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

# Define gate descriptions and matrices
gate_descriptions = ["rxp", "rxn", "ryp", "ryn", "rzp", "rzn"]
gate_matrices = [
    rotation_gate('x', np.pi / 128),    # rxp
    rotation_gate('x', -np.pi / 128),   # rxn
    rotation_gate('y', np.pi / 128),    # ryp
    rotation_gate('y', -np.pi / 128),   # ryn
    rotation_gate('z', np.pi / 128),    # rzp
    rotation_gate('z', -np.pi / 128)    # rzn
]

def get_fixed_target_unitary():
    return np.array([
        [0.76749896 - 0.43959894j, -0.09607122 + 0.45658344j],
        [0.09607122 + 0.45658344j,  0.76749896 + 0.43959894j]
    ], dtype=complex)

# Define the environment
class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, tolerance):
        super().__init__()
        self.gate_set = gate_set
        self.tolerance = tolerance
        self.action_space = spaces.Discrete(len(self.gate_set))
        
        # Adjust observation space for HER (state + goal)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        })
        
        self.max_steps = 130
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        self.target_U = get_fixed_target_unitary()
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        return self._get_obs_dict()
    
    def step(self, action):
        gate = self.gate_set[action]
        self.U_n = np.dot(self.U_n, gate)
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        reward = self._compute_reward()
        done = self._check_done()
        obs = self._get_obs_dict()
        self.current_step += 1
        truncated = False
        return obs, reward, done, truncated, {}

    def _get_obs_dict(self):
        observation = np.concatenate([self.O_n.real.flatten(), self.O_n.imag.flatten()])
        return {
            "observation": observation.astype(np.float32),
            "achieved_goal": observation.astype(np.float32),
            "desired_goal": np.concatenate([self.target_U.real.flatten(), self.target_U.imag.flatten()]).astype(np.float32)
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        return 0 if fidelity >= self.tolerance else -1
    
    def _compute_reward(self):
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        return 0 if fidelity >= self.tolerance else -1

    def _check_done(self):
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        return fidelity >= self.tolerance or self.current_step >= self.max_steps
    
    def average_gate_fidelity(self, U, V):
        diff = U - V
        singular_values = np.linalg.svd(diff, compute_uv=False)
        pseudo_fidelity = 1 - np.max(singular_values)
        return pseudo_fidelity

def evaluate_agent(model, env, num_episodes=10):
    success_count = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        gate_sequence = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            gate_sequence.append(action)
            obs, reward, done, _, _ = env.step(action)
        fidelity = env.average_gate_fidelity(env.U_n, env.target_U)
        sequence_length = len(gate_sequence)
        print(f"Final Fidelity: {fidelity}")
        print(f"Sequence Length: {sequence_length}")
        if fidelity >= env.tolerance:
            success_count += 1
            print("Successfully approximated the target unitary.")
            # Map action indices to gate descriptions
            gate_descriptions_list = [gate_descriptions[action] for action in gate_sequence]
            print("Gate Sequence:")
            print(gate_descriptions_list)
            # Compute the resultant matrix after applying the gate sequence
            U_S = np.eye(2, dtype=complex)
            for gate in gate_sequence:
                U_S = np.dot(U_S, env.gate_set[gate])
            print("Resultant matrix after applying the gate sequence:")
            print(U_S)
        else:
            print("Failed to approximate the target unitary.")
    success_rate = success_count / num_episodes
    return success_rate

# HER requires using a model that can accept goals, DQN will be wrapped by HER
env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=0.98)
env = Monitor(env)

goal_selection_strategy = "future"  # Strategy for HER (e.g., 'future', 'final', 'episode', 'random')
model = HER(
    policy="MlpPolicy",
    env=env,
    model_class=DQN,
    n_sampled_goal=4,  # Number of HER goals to sample per experience
    goal_selection_strategy=goal_selection_strategy,
    online_sampling=True,
    max_episode_length=env.max_steps,
    policy_kwargs=dict(net_arch=[128, 128], activation_fn=nn.SELU),
    buffer_size=int(1e5),
    batch_size=256,
    verbose=1
)

# Train the model
model.learn(total_timesteps=1_000_000)

# Evaluate the model
success_rate = evaluate_agent(model, env)
print(f'Success rate: {success_rate * 100:.2f}%')
