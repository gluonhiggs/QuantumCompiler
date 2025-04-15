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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from optuna.importance import get_param_importances
import gc
import torch
import optuna


def get_haar_random_unitary():
    z = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q @ np.diag(ph)

def rotation_gate(axis, angle):
    if axis == 'x':
        return np.array([
            [np.cos(angle / 2), -1j * np.sin(angle / 2)],
            [-1j * np.sin(angle / 2), np.cos(angle / 2)]
        ], dtype=complex)
    elif axis == 'y':
        return np.array([
            [np.cos(angle / 2), -np.sin(angle / 2)],
            [np.sin(angle / 2),  np.cos(angle / 2)]
        ], dtype=complex)
    elif axis == 'z':
        return np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ], dtype=complex)

# gate_descriptions = ["rxp", "rxn", "ryp", "ryn", "rzp", "rzn"]
# gate_matrices = [
#     rotation_gate('x',  np.pi / 128),
#     rotation_gate('x', -np.pi / 128),
#     rotation_gate('y',  np.pi / 128),
#     rotation_gate('y', -np.pi / 128),
#     rotation_gate('z',  np.pi / 128),
#     rotation_gate('z', -np.pi / 128)
# ]

gate_descriptions = ["V1", "V2", "V3"]
gate_matrices = [
    (1/np.sqrt(5)) * np.array([[1, 2j], [2j, 1]], dtype=complex),  # V1
    (1/np.sqrt(5)) * np.array([[1, 2], [-2, 1]], dtype=complex),   # V2
    (1/np.sqrt(5)) * np.array([[1+2j, 0], [0, 1-2j]], dtype=complex)  # V3
]
# Define the QuantumCompilerEnv class
class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, tolerance=0.02, max_steps=130):
        super().__init__()
        self.gate_set = gate_set
        self.tolerance = tolerance
        self.max_steps = max_steps

        # Use [-2,2] observation bounds (similar to Env B)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(len(self.gate_set))
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        # Now sample Haar-random target from Env B
        self.target_U = get_haar_random_unitary()
        self.last_axis = None
        return self._get_observation(), {}

    def step(self, action):
        gate = self.gate_set[action]
        self.U_n = np.dot(self.U_n, gate)
        obs = self._get_observation()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        diff = self.get_diff(self.U_n, self.target_U)
        done = (diff <= self.tolerance) or (self.current_step >= self.max_steps)
        self.current_step += 1
        info = {}
        truncated = False
        return obs, reward, done, truncated, info

    def _get_observation(self):
        O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        obs = np.concatenate([O_n.real.flatten(), O_n.imag.flatten()]).astype(np.float32)

        achieved_goal = np.concatenate([self.U_n.real.flatten(), self.U_n.imag.flatten()]).astype(np.float32)
        desired_goal  = np.concatenate([self.target_U.real.flatten(), self.target_U.imag.flatten()]).astype(np.float32)

        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }

    def get_diff(self, U, V):
        diff = np.linalg.norm(U - V, 2)
        return diff

    def compute_reward(self, achieved_goals, desired_goals, info):
        # Ensure inputs are 2D: (batch_size, 8)
        if achieved_goals.ndim == 1:
            achieved_goals = achieved_goals[None, :]  # Add batch dimension if single input
        if desired_goals.ndim == 1:
            desired_goals = desired_goals[None, :]
        batch_size = achieved_goals.shape[0]

        # Construct batched complex matrices
        U_n = np.zeros((batch_size, 2, 2), dtype=complex)
        U_n[:, 0, 0] = achieved_goals[:, 0] + 1j * achieved_goals[:, 4]
        U_n[:, 0, 1] = achieved_goals[:, 1] + 1j * achieved_goals[:, 5]
        U_n[:, 1, 0] = achieved_goals[:, 2] + 1j * achieved_goals[:, 6]
        U_n[:, 1, 1] = achieved_goals[:, 3] + 1j * achieved_goals[:, 7]

        U_target = np.zeros((batch_size, 2, 2), dtype=complex)
        U_target[:, 0, 0] = desired_goals[:, 0] + 1j * desired_goals[:, 4]
        U_target[:, 0, 1] = desired_goals[:, 1] + 1j * desired_goals[:, 5]
        U_target[:, 1, 0] = desired_goals[:, 2] + 1j * desired_goals[:, 6]
        U_target[:, 1, 1] = desired_goals[:, 3] + 1j * desired_goals[:, 7]

        # Compute differences for all pairs
        diffs = np.array([np.linalg.norm(U_n[i] - U_target[i], 2) for i in range(batch_size)])

        # Compute rewards
        rewards = np.where(diffs < self.tolerance, 0, -1/self.max_steps)


        # Return scalar if batch_size is 1 (for step), array otherwise (for HER)
        return rewards[0] if batch_size == 1 else rewards

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
        # Plot each metric in its own figure to adhere to best styling practices
        plt.figure()
        plt.plot(self.episode_rewards, label="Episode Reward")
        plt.title("Episode Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, "single_qbit_diff_sparse.png"))
        plt.close()

        plt.figure()
        plt.plot(self.episode_lengths, label="Episode Length")
        plt.title("Episode Length Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.legend()
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, "single_qbit_diff_sparse_length.png"))
        plt.close()

def evaluate_agent(model:DQN, vec_env, num_episodes=5):
    # Access the raw environment for direct attribute calls
    env = vec_env.envs[0]
    success_count = 0
    
    for _ in range(num_episodes):
        gate_sequence = []
        obs, info = env.reset()  # because your Env returns (obs, info)
        done = False
        target_U = env.target_U
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            gate_sequence.append(action)
            done = done or truncated
        diff = env.get_diff(env.U_n, target_U)
        if diff <= env.tolerance:
            success_count += 1
            gate_descriptions_list = [gate_descriptions[int(action)] for action in gate_sequence]
            print("Gate Sequence:")
            print(gate_descriptions_list)
            # Write the target unitary, the gate sequence, and the final unitary to a file
            with open("gate_sequence_sparse.txt", "a") as f:
                f.write(f"Target Unitary:\n{target_U}\n")
                f.write(f"Gate Sequence: {gate_descriptions_list}\n")
                f.write(f"Final Unitary:\n{env.U_n}\n\n")

    return success_count / num_episodes

def make_env(seed=None, idx=0):
    """
    Return a function that when called, creates a new QuantumCompilerEnv,
    wraps it with Monitor, etc.
    """
    def _init():
        env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=0.1)
        # You could do: env.seed(seed + idx) if you want distinct seeds
        env = Monitor(env)
        return env
    return _init

def make_vec_env(n_envs=1, use_subproc=True, seed=0):
    """
    Create a SubprocVecEnv (or DummyVecEnv) for parallel env execution.
    """
    env_fns = [make_env(seed=seed, idx=i) for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        # For 1 env, or if you don't want to use subproc
        return DummyVecEnv(env_fns)

def objective(trial):
    # 1) Suggest the number of parallel envs
    n_envs = trial.suggest_int("n_envs", 1, 16)

    # 2) Suggest other hyperparams
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.2, 1.0)
    learning_starts = trial.suggest_int("learning_starts", 10000, 200000)
    train_freq = trial.suggest_categorical("train_freq", [(1, 'step'), (2, 'step'), (4, 'step'), (8, 'step')])
    net_arch_depth = trial.suggest_categorical("net_arch_depth", [1, 2, 3, 4, 5, 6, 7, 8])
    net_arch_width = trial.suggest_categorical("net_arch_width", [64, 128, 256, 512, 1024])
    device = trial.suggest_categorical("device", ["cpu", "cuda"])

    # Build net_arch
    net_arch = [net_arch_width] * net_arch_depth
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=nn.SELU,
    )

    # 3) Create vectorized env
    vec_env = make_vec_env(n_envs=n_envs, use_subproc=True)
    class RewardCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.total_reward = 0
            self.episode_count = 0

        def _on_step(self):
            if self.locals.get('dones')[0]:
                ep_info = self.locals.get('infos')[0].get('episode')
                if ep_info:
                    self.total_reward += ep_info['r']
                    self.episode_count += 1
            return True
    callback = RewardCallback()
    # 4) Build model
    model = DQN(
        'MultiInputPolicy',
        vec_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_freq= train_freq,
        buffer_size=buffer_size,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        exploration_fraction=exploration_fraction,
        learning_starts=learning_starts,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            goal_selection_strategy='future',
            n_sampled_goal=4,
        )
    )
    # 5) Train for 200k timesteps
    model.learn(total_timesteps=1_000_000, callback =callback, log_interval=100)

    # 6) Evaluate
    mean_reward = callback.total_reward / max(1, callback.episode_count)  # Avoid division by 0

    # Cleanup
    del model
    del vec_env
    gc.collect()
    return mean_reward

if __name__ == "__main__":
    # Example usage of Optuna
    sampler = optuna.samplers.TPESampler(seed=123)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=20)  # e.g. 5 trials for demonstration

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Success Rate): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    # Write the best trial to a CSV file
    with open("best_trial.csv", "w") as f:
        f.write("param,value\n")
        for key, value in trial.params.items():
            f.write(f"{key},{value}\n")
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("param_importances.png")
    best_params = trial.params
    vec_env = make_vec_env(n_envs=best_params["n_envs"], use_subproc=(best_params["n_envs"] > 1))

    net_arch_depth = best_params["net_arch_depth"]
    net_arch_width = best_params["net_arch_width"]
    net_arch = [net_arch_width] * net_arch_depth
    
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=nn.SELU,
    )

    model = DQN(
        'MultiInputPolicy',
        vec_env,
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        train_freq=best_params["train_freq"],  # e.g. (1, 'step')
        buffer_size=best_params["buffer_size"],
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=best_params["exploration_fraction"],
        learning_starts=best_params["learning_starts"],
        gamma=0.99931,
        verbose=1,
        device=best_params["device"],
        policy_kwargs=policy_kwargs,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            goal_selection_strategy='future',
            n_sampled_goal=4,
        )
    )

    callback = PlottingCallback(save_path='./data')
    model.learn(total_timesteps=20_000_000, log_interval=1000, callback=callback)

    # Save the model
    model.save("single_qbit_diff_sparse")
    # # Load the model
    # model = DQN.load("single_qbit_clustered_v1_copy", env=vec_env)
    # Evaluate the model

    eval_env = make_vec_env(n_envs=1, use_subproc=False)
    final_success = evaluate_agent(model, eval_env, num_episodes=1000)
    print(f"Final success rate: {final_success}")