import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
import torch
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
# Define the rotation gates
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

# Generate Haar random target unitary matrix
def get_haar_random_unitary():
    z = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q @ np.diag(ph)

# Define the RL environment
class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, tolerance=0.99, max_steps=300):
        super(QuantumCompilerEnv, self).__init__()
        self.gate_set = gate_set
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(gate_set))  # Select one of the rotation gates
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)  # Real + Imaginary parts of the unitary
        
        self.reset()

    def reset(self):
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)  # Start with the identity matrix
        self.target_U = get_haar_random_unitary()  # Haar-random target unitary
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)  # Difference between current and target
        return self._get_observation()

    def step(self, action):
        gate = self.gate_set[action]
        self.U_n = np.dot(self.U_n, gate)  # Apply the selected gate
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        reward = self._compute_reward()
        done = self._check_done()
        self.current_step += 1
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        obs = np.concatenate([self.O_n.real.flatten(), self.O_n.imag.flatten()])
        return obs.astype(np.float32)

    def _compute_reward(self):
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        if fidelity >= self.tolerance:
            reward = 100.0  # High reward for achieving the target
        else:
            reward = -1.0  # Penalize for each step taken
        return reward

    def _check_done(self):
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        return fidelity >= self.tolerance or self.current_step >= self.max_steps

    def average_gate_fidelity(self, U, V):
        d = 2
        trace_U_dagger_V = np.trace(np.dot(np.conjugate(U.T), V))
        fidelity = (1 / d**2) * np.abs(trace_U_dagger_V)**2
        return fidelity

class PlottingCallback(BaseCallback):
    def __init__(self, verbose=0, save_path=None):
        super(PlottingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.save_path = save_path  # Path to save the plots

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:  # 'dones' is a list
            # Get the reward and episode length from the current episode
            episode_info = self.locals.get('infos')[0].get('episode')
            if episode_info is not None:
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
        return True

    def _on_training_end(self) -> None:
        plt.figure(figsize=(12, 6))

        # Plotting episode rewards over time
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, label="Episode Reward")
        plt.title("Episode Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()

        # Plotting episode lengths over time
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths, label="Episode Length", color='orange')
        plt.title("Episode Length Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.legend()

        plt.tight_layout()

        # Save plots if a save path is provided
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)  # Ensure directory exists
            plt.savefig(os.path.join(self.save_path, "training_plots.png"))

        plt.close()

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

if __name__ == '__main__':
    env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=0.99)
    env = Monitor(env)
    # Use the PPO algorithm for training
    model = PPO('MlpPolicy', env, verbose=1)
    # Define the custom plotting callback
    plotting_callback = PlottingCallback(save_path='./data')

    # Train the model with the PlottingCallback
    model.learn(total_timesteps=8000000, log_interval=100, callback=plotting_callback)

    # Evaluate the agent (optional, not related to plotting)
    success_rate = evaluate_agent(model, env)
    print(f'Success rate: {success_rate * 100:.2f}%')

