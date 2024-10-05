import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

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

def make_env():
    def _init():
        env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=0.99)
        env = Monitor(env)  # Wrap each individual environment with Monitor
        return env
    return _init

# Define the RL environment
class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, tolerance=0.99, target_U=None):
        super(QuantumCompilerEnv, self).__init__()
        self.gate_set = gate_set
        self.tolerance = tolerance
        self.target_U = target_U
        self.action_space = spaces.Discrete(len(gate_set))  # Select one of the rotation gates
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(8,), dtype=np.float32)  # Adjusted bounds
        self.max_steps = 300
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        # If a specific target_U is provided, use it; otherwise, generate a random one
        if self.target_U is None:
            self.target_U = get_haar_random_unitary()
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

        if fidelity >= self.tolerance:
            reward = (L - n) + 1  # Encourage shorter sequences
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
        d = U.shape[0]
        U_dagger = np.conjugate(U.T)
        trace_U_dagger_V = np.trace(np.dot(U_dagger, V))
        fidelity = (np.abs(trace_U_dagger_V)**2 + d) / (d * (d + 1))
        return fidelity

class PlottingCallback(BaseCallback):
    def __init__(self, verbose=1, save_path=None):
        super(PlottingCallback, self).__init__(verbose)
        self.save_path = save_path  # Path to save the plots
        self.num_envs = None
        self.current_rewards = None
        self.current_lengths = None
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_start(self):
        self.num_envs = self.training_env.num_envs
        self.current_rewards = np.zeros(self.num_envs)
        self.current_lengths = np.zeros(self.num_envs)

    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        dones = self.locals['dones']

        self.current_rewards += rewards
        self.current_lengths += 1

        for i in range(self.num_envs):
            if dones[i]:
                # Episode finished for environment i
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(self.current_lengths[i])

                # Reset the counters
                self.current_rewards[i] = 0
                self.current_lengths[i] = 0
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
            plt.savefig(os.path.join(self.save_path, "rotation_operators_v2.png"))

        plt.close()

def evaluate_agent(model, target_unitaries, env_class, num_episodes=10):
    success_count = 0
    total_episodes = len(target_unitaries) * num_episodes
    for target_U in target_unitaries:
        env = env_class(gate_set=gate_matrices, tolerance=0.99, target_U=target_U)
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            gate_sequence = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                gate_sequence.append(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
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
    success_rate = success_count / total_episodes
    return success_rate


policy_kwargs = dict(
    net_arch=[128, 128],
    activation_fn=nn.SELU,
)

if __name__ == '__main__':
    # Create 40 environments
    num_envs = 40
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
    # Use the PPO algorithm for training
    model = PPO('MlpPolicy',
                envs,
                policy_kwargs=policy_kwargs,
                learning_rate=0.0001,
                batch_size=128,
                tensorboard_log="./data/ppo_tensorboard/",
                verbose=1)
    # Define the custom plotting callback
    plotting_callback = PlottingCallback(save_path='./data')

    # Train the model with the PlottingCallback
    model.learn(total_timesteps=8000000, log_interval=100, callback=plotting_callback)

    # Evaluate the agent
    target_unitaries = [get_haar_random_unitary() for _ in range(10)]
    eval_env_class = QuantumCompilerEnv
    success_rate = evaluate_agent(model,target_unitaries,eval_env_class)
    print(f'Success rate: {success_rate * 100:.2f}%')
