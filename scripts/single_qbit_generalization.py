import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import torch

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

class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, tolerance):
        super().__init__()
        self.gate_set = gate_set
        self.tolerance = tolerance
        self.action_space = spaces.Discrete(len(self.gate_set))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.max_steps = 130
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        self.target_U = get_fixed_target_unitary()
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
        return np.concatenate([self.O_n.real.flatten(), self.O_n.imag.flatten()]).astype(np.float32)
    
    def _compute_reward(self):
        L = self.max_steps
        n = self.current_step
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        distance = 1 - fidelity
        if distance < (1 - self.tolerance):
            reward = 2 * (L - n) + 1
        else:
            reward = - distance / L
        return reward

    def _check_done(self):
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        return fidelity >= self.tolerance or self.current_step >= self.max_steps
    
    def average_gate_fidelity(self, U, V):
        diff = U - V
        singular_values = np.linalg.svd(diff, compute_uv=False)
        return 1 - np.max(singular_values)

def aq_star_search(env, gate_set, gate_descriptions, start_U=None, max_depth=30, alpha=1.0, beta=2.0):
    start_U = start_U if start_U is not None else np.eye(2, dtype=complex)
    queue = [(0, 0, [], start_U, None)]
    visited = set()
    best_seq = []
    best_fidelity = env.average_gate_fidelity(start_U, env.target_U)
    
    while queue:
        cost, steps, seq, U, last_axis = heappop(queue)
        if steps > max_depth:
            continue
        
        fidelity = env.average_gate_fidelity(U, env.target_U)
        print(f"AQ* Step {steps}, Fidelity: {fidelity:.4f}, Queue Size: {len(queue)}")
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_seq = seq[:]
        if fidelity >= env.tolerance:
            return seq
        if fidelity < best_fidelity - 0.05:
            continue
        
        for action in range(len(gate_set)):
            gate = gate_set[action]
            new_U = np.dot(U, gate)
            new_key = tuple(new_U.flatten())
            if new_key not in visited:
                if len(visited) < 10000:
                    visited.add(new_key)
                gate_axis = gate_descriptions[action][1]
                transitions = 1 if last_axis and last_axis != gate_axis else 0
                new_cost = -fidelity + alpha * steps + beta * transitions
                heappush(queue, (new_cost, steps + 1, seq + [action], new_U, gate_axis))
    
    return best_seq

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

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            episode_info = self.locals.get('infos')[0].get('episode')
            if episode_info is not None:
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
        return True

    def _on_training_end(self) -> None:
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
            plt.savefig(os.path.join(self.save_path, "dqn_strong_v1.png"))
        plt.close()

def train_agent():
    env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=0.98)
    env = Monitor(env)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0005,
        buffer_size=100000,
        learning_starts=50000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.5,
        exploration_final_eps=0.02,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    callback = PlottingCallback(save_path="./data")
    model.learn(total_timesteps=2000000, callback=callback)
    model.save("dqn_quantum_compiler")
    return model, env

def evaluate_agent(model, env, num_episodes=10):
    success_count = 0
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        gate_sequence = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            gate_sequence.append(action)
            obs, reward, done, _, _ = env.step(action)
        
        print(f"\nEpisode {i+1}:")
        print(f"Original Sequence Length: {len(gate_sequence)}")
        dqn_fidelity = env.unwrapped.average_gate_fidelity(env.unwrapped.U_n, env.unwrapped.target_U)
        print(f"DQN Fidelity: {dqn_fidelity:.4f}")
        
        refined_sequence = aq_star_search(env.unwrapped, gate_matrices, gate_descriptions, start_U=env.unwrapped.U_n, max_depth=30, beta=2.0)
        U_refined = np.eye(2, dtype=complex)
        for action in refined_sequence:
            U_refined = np.dot(U_refined, gate_matrices[action])
        
        fidelity = env.unwrapped.average_gate_fidelity(U_refined, env.unwrapped.target_U)
        print(f"Refined Sequence Length: {len(refined_sequence)}")
        print(f"Final Fidelity: {fidelity:.4f}")
        if fidelity >= env.unwrapped.tolerance:
            success_count += 1
            print("Success! Refined Gate Sequence:", [gate_descriptions[action] for action in refined_sequence])
        else:
            print("Failed to approximate.")
    
    success_rate = success_count / num_episodes
    print(f"Success Rate: {success_rate:.2%}")
    return success_rate

if __name__ == "__main__":
    model, env = train_agent()
    success_rate = evaluate_agent(model, env, num_episodes=10)