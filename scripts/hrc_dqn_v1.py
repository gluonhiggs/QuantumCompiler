import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN, HER
import torch.nn as nn
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import matplotlib.pyplot as plt
# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
filename = os.path.splitext(os.path.basename(__file__))[0]


# HRC universal gate set
# \begin{equation}
#         V_1 = \frac{1}{\sqrt{5}}
#     \begin{pmatrix}
#         1 & 2\rm{i} \\
#         2\rm{i} & 1
#     \end{pmatrix} \qquad
#     V_2 = \frac{1}{\sqrt{5}}
#     \begin{pmatrix}
#         1 & 2 \\
#         -2 & 1
#     \end{pmatrix} \qquad
#     V_3 = \frac{1}{\sqrt{5}}
#     \begin{pmatrix}
#         1+2\rm{i} & 0 \\
#         0 & 1-2\rm{i}
#     \end{pmatrix}
# \end{equation}
gate_descriptions = ["V1", "V2", "V3"]
gate_matrices = [
    np.array([[1/np.sqrt(5), 2j/np.sqrt(5)], [2j/np.sqrt(5), 1/np.sqrt(5)]], dtype=complex),  # V1
    np.array([[1/np.sqrt(5), 2/np.sqrt(5)], [-2/np.sqrt(5), 1/np.sqrt(5)]], dtype=complex),  # V2
    np.array([[(1+2j)/np.sqrt(5), 0], [0, (1-2j)/np.sqrt(5)]], dtype=complex)  # V3
]


def get_fixed_target_unitary():
    return np.array([
        [0.76749896 - 0.43959894j, -0.09607122 + 0.45658344j],
        [0.09607122 + 0.45658344j,  0.76749896 + 0.43959894j]
    ], dtype=complex)

class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, fidelity):
        super().__init__()
        self.gate_set = gate_set
        self.fidelity = fidelity
        self.action_space = spaces.Discrete(len(self.gate_set))
        # Adjust observation space for HER (state + goal)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        })
        self.max_steps = 130  # Updated to match the paper
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
        obs = np.concatenate([self.O_n.real.flatten(), self.O_n.imag.flatten()])
        return obs.astype(np.float32)
    
    def _compute_reward(self):
        L = self.max_steps
        distance = 1 - self.average_gate_fidelity(self.U_n, self.target_U)
        if distance < (1 - self.fidelity):
            reward = 0
        else:
            reward = - 1 / L
        return reward

    def _check_done(self):
        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        if fidelity >= self.fidelity or self.current_step >= self.max_steps:
            return True
        else:
            return False
    
    def average_gate_fidelity(self, U, V):
        diff = U - V
        operator_norm = np.linalg.norm(diff, 'fro')
        pseudo_fidelity = 1 - operator_norm
        return pseudo_fidelity

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
            plt.savefig(os.path.join(self.save_path, f"{filename}.png"))
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
        if fidelity >= env.fidelity:
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
    env = QuantumCompilerEnv(gate_set=gate_matrices, fidelity=0.9)
    env = Monitor(env)
    goal_selection_strategy = "future"
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
        verbose=1,
        device='gpu',
    )

    # Define the custom plotting callback
    plotting_callback = PlottingCallback(save_path='./data')
    model.learn(total_timesteps=10000000, log_interval=100, callback=plotting_callback)
    
    success_rate = evaluate_agent(model, env)
    print(f'Success rate: {success_rate * 100:.2f}%')
