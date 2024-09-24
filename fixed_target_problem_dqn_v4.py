import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
import torch.nn as nn
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import matplotlib.pyplot as plt
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

gate_descriptions = ["rxp", "rxn", "ryp", "ryn", "rzp", "rzn"]
gate_matrices = [
    rotation_gate('x', np.pi / 256),    # rxp
    rotation_gate('x', -np.pi / 256),   # rxn
    rotation_gate('y', np.pi / 256),    # ryp
    rotation_gate('y', -np.pi / 256),   # ryn
    rotation_gate('z', np.pi / 256),    # rzp
    rotation_gate('z', -np.pi / 256)    # rzn
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
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8+1,), dtype=np.float32)
        self.max_steps = 260  # Adjust corresponding to the base
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        self.target_U = get_fixed_target_unitary()
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        self.previous_axis = None
        self.axis_changes = 0
        self.axis_change_penalty_rate = 0.05
        return self._get_observation(), {}
    
    def step(self, action):
        gate = self.gate_set[action]
        # Extract the axis from the gate description
        axis = gate_descriptions[action][1]
         # Check if the axis has changed
        if self.previous_axis is not None and axis != self.previous_axis:
            self.axis_changes += 1
        # Update previous axis
        self.previous_axis = axis
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
        obs = np.concatenate([self.O_n.real.flatten(), self.O_n.imag.flatten(), [self.axis_changes / self.max_steps]])
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
        reward -= self.axis_changes * self.axis_change_penalty_rate
        return reward

    def _check_done(self):
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        if fidelity >= self.tolerance or self.current_step >= self.max_steps:
            return True
        else:
            return False
    
    def average_gate_fidelity(self, U, V):
        """
            ref fidelity
        """
        # return (np.abs(np.trace(np.dot(U.conj().T, V)))**2 + 4) / 12

        """
          IBM fidelity
        """
        d = 2
        U_dagger = np.conjugate(U.T)
        trace_U_dagger_V = np.trace(np.dot(U_dagger, V))
        fidelity = (1 / d**2) * np.abs(trace_U_dagger_V)**2

        """
            operator norm fidelity
        """
        # difference = U - V
        # singular_values = np.linalg.svd(difference, compute_uv=False)
        # fidelity = np.max(singular_values)
        return fidelity 


policy_kwargs = dict(
    net_arch=[128, 128],
    activation_fn=nn.SELU,
)
class PlottingCallback(BaseCallback):
    def __init__(self, verbose=0, save_path=None):
        super(PlottingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.save_path = save_path  # Path to save the plots

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:  # 'dones' is a list
            # Get the reward and episode length
            episode_info = self.locals.get('infos')[0].get('episode')
            if episode_info is not None:
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
        return True

    def _on_training_end(self) -> None:
        plt.figure(figsize=(12, 6))

        # Plotting episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, label="Episode Reward")
        plt.title("Episode Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()

        # Plotting episode lengths
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths, label="Episode Length", color='orange')
        plt.title("Episode Length Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.legend()

        plt.tight_layout()

        # Save plots if save_path is provided
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, "training_plots_v4.png"))
        plt.close()

def evaluate_agent(model, env, num_episodes=1):
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



if __name__ == '__main__':
    env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=0.98)
    env = Monitor(env)
    model = DQN(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0005,  
        batch_size=10000,      
        train_freq=(1, 'episode'),
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        # gamma= 0.99976,
        exploration_fraction=0.99976, 
        learning_starts=50000,
        target_update_interval=20000,
        buffer_size=100000,
        verbose=1,
        device='cuda',  # Change to 'cpu' if not using GPU
    )
    # Define the custom plotting callback
    plotting_callback = PlottingCallback(save_path='./data')
    model.learn(total_timesteps=10000000, log_interval=100, callback=plotting_callback)
    
    success_rate = evaluate_agent(model, env)
    print(f'Success rate: {success_rate * 100:.2f}%')
