import os
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN, PPO
import torch.nn as nn
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing

# Suppress TensorFlow warnings (optional)
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

# Gate set B
my_gate_set = [
    rotation_gate('x', np.pi / 128),
    rotation_gate('x', -np.pi / 128),
    rotation_gate('y', np.pi / 128),
    rotation_gate('y', -np.pi / 128),
    rotation_gate('z', np.pi / 128),
    rotation_gate('z', -np.pi / 128)
]

def build_target_unitary(gate_set, num_gates=87):
    U = np.eye(2, dtype=complex)
    for _ in range(num_gates):
        index = np.random.choice(len(gate_set))
        gate = gate_set[index]
        U = np.dot(gate, U)
    return U

class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, tolerance):
        super(QuantumCompilerEnv, self).__init__()
        self.gate_set = gate_set
        self.tolerance = tolerance
        self.action_space = spaces.Discrete(len(self.gate_set))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.max_steps = 300  # Updated to match the paper
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        self.target_U = build_target_unitary(self.gate_set, num_gates=87)
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        return self._get_observation()
    
    def step(self, action):
        gate = self.gate_set[action]
        self.U_n = np.dot(gate, self.U_n)
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        reward = self._compute_reward()
        done = self._check_done()
        obs = self._get_observation()
        info = {}
        self.current_step += 1
        return obs, reward, done, info
    
    def _get_observation(self):
        obs = np.concatenate([self.O_n.real.flatten(), self.O_n.imag.flatten()])
        return obs.astype(np.float32)
    
    def _compute_reward(self):
        L = self.max_steps
        n = self.current_step
        fidelity = self._average_gate_fidelity(self.U_n, self.target_U)
        distance = 1 - fidelity
        
        if distance < (1 - self.tolerance):
            reward = (L - n) + 1
        else:
            reward = -distance / L
        return reward

    def _check_done(self):
        fidelity = self._average_gate_fidelity(self.U_n, self.target_U)
        if fidelity >= self.tolerance or self.current_step >= self.max_steps:
            return True
        else:
            return False
    
    def _average_gate_fidelity(self, U, V):
        return (np.abs(np.trace(np.dot(U.conj().T, V)))**2 + 2) / 6

policy_kwargs = dict(
    net_arch=[128, 128],
    activation_fn=nn.SELU,
)

def describe_gate(gate):
    # Helper function to map gate matrices back to descriptions
    gate_list = [
        rotation_gate('x', np.pi / 128),
        rotation_gate('x', -np.pi / 128),
        rotation_gate('y', np.pi / 128),
        rotation_gate('y', -np.pi / 128),
        rotation_gate('z', np.pi / 128),
        rotation_gate('z', -np.pi / 128)
    ]
    gate_descriptions = [
        "R_x(π/128)",
        "R_x(-π/128)",
        "R_y(π/128)",
        "R_y(-π/128)",
        "R_z(π/128)",
        "R_z(-π/128)"
    ]
    for idx, key_gate in enumerate(gate_list):
        if np.allclose(gate, key_gate):
            return gate_descriptions[idx]
    return "Unknown Gate"

def evaluate_agent(model, env, num_episodes=1):
    success_count = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        gate_sequence = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            gate_sequence.append(action)
            obs, reward, done, info = env.step(action)
        fidelity = env._average_gate_fidelity(env.U_n, env.target_U)
        sequence_length = len(gate_sequence)
        print(f"Final Fidelity: {fidelity}")
        print(f"Sequence Length: {sequence_length}")
        if fidelity >= env.tolerance:
            success_count += 1
            print("Successfully approximated the target unitary.")
            # Map action indices to gate descriptions
            gate_descriptions = [describe_gate(env.gate_set[action]) for action in gate_sequence]
            print("Gate Sequence:")
            for idx, desc in enumerate(gate_descriptions, 1):
                print(f"{idx}: {desc}")
        else:
            print("Failed to approximate the target unitary.")
    success_rate = success_count / num_episodes
    return success_rate

def make_env():
    def _init():
        env = QuantumCompilerEnv(gate_set=my_gate_set, tolerance=0.99)
        return env
    return _init
if __name__ == '__main__':
    # Set the multiprocessing start method to 'fork' (Unix-based systems)
    multiprocessing.set_start_method('fork')

    # Create the vectorized environment with 40 agents
    num_agents = 40  # As per your requirement

    env = SubprocVecEnv([make_env() for _ in range(num_agents)])

    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0001,
        batch_size=128,
        n_steps=512,  # Adjust based on your needs
        verbose=1,
        device='cuda',  # Change to 'cpu' if not using GPU
    )

    model.learn(total_timesteps=500000)

    # For evaluation, we need a single environment
    eval_env = QuantumCompilerEnv(gate_set=my_gate_set, tolerance=0.99)

    success_rate = evaluate_agent(model, eval_env)
    print(f'Success rate: {success_rate * 100:.2f}%')
