import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import torch.nn as nn
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Rotation gate generator
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

# Define gate set and descriptions
gate_descriptions = ["rxp", "rxn", "ryp", "ryn", "rzp", "rzn"]
gate_matrices = [
    rotation_gate('x', np.pi / 128),    # rxp
    rotation_gate('x', -np.pi / 128),   # rxn
    rotation_gate('y', np.pi / 128),    # ryp
    rotation_gate('y', -np.pi / 128),   # ryn
    rotation_gate('z', np.pi / 128),    # rzp
    rotation_gate('z', -np.pi / 128)    # rzn
]

# Generate Haar random unitary
def generate_haar_unitary():
    z = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = np.multiply(q, ph, q)
    return q

# Define the environment for quantum compilation
class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, tolerance):
        super().__init__()
        self.gate_set = gate_set
        self.tolerance = tolerance
        self.action_space = spaces.Discrete(len(self.gate_set))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.max_steps = 300  # Increased step limit per episode
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        self.target_U = generate_haar_unitary()  # Randomly generate Haar unitary
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        return self._get_observation(), {}
    
    def step(self, action):
        gate = self.gate_set[action]
        self.U_n = np.dot(self.U_n, gate)
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        reward = self._compute_reward()
        done = self._check_done()
        obs = self._get_observation()
        self.current_step += 1
        info = {}
        truncated = False
        return obs, reward, done, truncated, info
    
    def _get_observation(self):
        obs = np.concatenate([self.O_n.real.flatten(), self.O_n.imag.flatten()])
        return obs.astype(np.float32)
    
    def _compute_reward(self):
        L = self.max_steps
        n = self.current_step
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        distance = 1 - fidelity
        
        if distance < (1 - self.tolerance):
            reward = (L - n) + 1
        else:
            reward = -distance / L
        return reward

    def _check_done(self):
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        if fidelity >= self.tolerance or self.current_step >= self.max_steps:
            return True
        else:
            return False
    
    def average_gate_fidelity(self, U, V):
        """IBM fidelity"""
        d = 2
        U_dagger = np.conjugate(U.T)
        trace_U_dagger_V = np.trace(np.dot(U_dagger, V))
        fidelity = (1 / d**2) * np.abs(trace_U_dagger_V)**2
        return fidelity

# Training using PPO
def train_agent():
    env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=0.99)
    env = Monitor(env)
    policy_kwargs = dict(
        net_arch=[128, 128],
        activation_fn=nn.SELU,
    )
    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0001,  
        batch_size=128,         
        # n_steps=2048,  # PPO-specific: Number of steps before update
        verbose=1,
        device='cuda',  # Use 'cpu' if no GPU available
    )

    model.learn(total_timesteps=1000000)
    model.save("ppo_quantum_compiler")
    
# Evaluation function
def evaluate_agent(model, env, num_episodes=100):
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
        print(f"Final Fidelity: {fidelity}")
        if fidelity >= env.tolerance:
            success_count += 1
    success_rate = success_count / num_episodes
    return success_rate

if __name__ == '__main__':
    train_agent()
    env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=0.99)
    model = PPO.load("ppo_quantum_compiler")
    success_rate = evaluate_agent(model, env)
    print(f'Success rate: {success_rate * 100:.2f}%')
