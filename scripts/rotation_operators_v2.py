import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import json
from tqdm import tqdm

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
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)  # Adjusted bounds
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
        # consider reshape the reward by adding a new penalty term (e.g. -0.01 to prevent taking more steps)
        # reward -= 0.01
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
        self.episode_count = 0  # To track episode count
        self.total_timesteps = 0 

    def _on_training_start(self):
        self.num_envs = self.training_env.num_envs
        self.current_rewards = np.zeros(self.num_envs)
        self.current_lengths = np.zeros(self.num_envs)
        self.total_timesteps = 0

    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        self.total_timesteps += self.locals['n_steps']

        self.current_rewards += rewards
        self.current_lengths += 1

        for i in range(self.num_envs):
            if dones[i]:
                # Episode finished for environment i
                self.episode_count += 1
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(self.current_lengths[i])

                if self.verbose > 0:
                    print(f"Env {i}: Episode {self.episode_count}, Reward: {self.current_rewards[i]}, Length: {self.current_lengths[i]}, Timesteps per Total Timesteps: {self.total_timesteps}")


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

def matrix_to_readable_string(matrix, precision=9):
    """Convert a complex matrix to a human-readable string."""
    lines = []
    for row in matrix:
        row_str = '['
        for elem in row:
            real_part = f"{elem.real:.{precision}f}"
            imag_part = f"{elem.imag:.{precision}f}"
            if elem.imag >= 0:
                elem_str = f"{real_part}+{imag_part}j"
            else:
                elem_str = f"{real_part}{imag_part}j"
            row_str += f"{elem_str}, "
        row_str = row_str.rstrip(', ') + ']'
        lines.append(row_str)
    return '\n'.join(lines)

def evaluate_agent(model, target_unitaries, env_class, output_file='evaluation_results.jsonl'):
    success_count = 0
    total_episodes = len(target_unitaries)
    # Create a single environment instance
    env = env_class(gate_set=gate_matrices, tolerance=0.99)
    with open(output_file, 'w') as f:
        for idx, target_U in enumerate(tqdm(target_unitaries, desc="Evaluating")):
            env.target_U = target_U 
            obs, _ = env.reset()
            done = False
            gate_sequence = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                gate_sequence.append(gate_descriptions[action])
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            fidelity = env.average_gate_fidelity(env.U_n, env.target_U)
            sequence_length = len(gate_sequence)
            success = fidelity >= env.tolerance
            if success:
                success_count += 1
            # Prepare the result entry
            result = {
                'index': idx,
                'fidelity': fidelity,
                'sequence_length': sequence_length,
                'success': bool(success),
                'gate_sequence': gate_sequence,
                # Include matrices in human-readable format
                'target_U': matrix_to_readable_string(env.target_U),
                'approximate_U': matrix_to_readable_string(env.U_n)
            }
            # Write the result entry as a JSON line
            f.write(json.dumps(result) + '\n')
    success_rate = success_count / total_episodes
    print(f'Success rate: {success_rate * 100:.2f}%')
    return success_rate


policy_kwargs = dict(
    net_arch=[128, 128],
    activation_fn=nn.SELU,
)

if __name__ == '__main__':
    # Set random seeds for reproducibility
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create environments and normalize observations
    num_envs = 40  # Adjusted number of environments
    agent_steps = 1500000
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
    # envs = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # Define the PPO model
    model = PPO('MlpPolicy',
                envs,
                policy_kwargs=policy_kwargs,
                learning_rate=1e-4,
                batch_size=128,
                # ent_coef=0.001,
                device='cuda',
                verbose=1)
    # Define the custom plotting callback
    plotting_callback = PlottingCallback(save_path='./data')
    
    # Train the model with the PlottingCallback
    model.learn(total_timesteps=agent_steps, log_interval=100, callback=plotting_callback)
    
    # Evaluate the agent
    num_test_targets = 1000  # Testing with 1 million targets
    target_unitaries = [get_haar_random_unitary() for _ in range(num_test_targets)]
    eval_env_class = QuantumCompilerEnv
    success_rate = evaluate_agent(model, target_unitaries, eval_env_class, output_file='evaluation_results.jsonl')
    print(f'Success rate: {success_rate * 100:.2f}%')
