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
import optuna
from optuna.importance import get_param_importances
import gc
import torch


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

gate_descriptions = ["rxp", "rxn", "ryp", "ryn", "rzp", "rzn"]
gate_matrices = [
    rotation_gate('x',  np.pi / 128),
    rotation_gate('x', -np.pi / 128),
    rotation_gate('y',  np.pi / 128),
    rotation_gate('y', -np.pi / 128),
    rotation_gate('z',  np.pi / 128),
    rotation_gate('z', -np.pi / 128)
]

class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, tolerance=0.98, max_steps=130):
        super().__init__()
        self.gate_set = gate_set
        self.tolerance = tolerance
        self.max_steps = max_steps

        # Use [-2,2] observation bounds (similar to Env B)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-2, high=2, shape=(8,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-2, high=2, shape=(8,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-2, high=2, shape=(8,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(len(self.gate_set))

        # Axis transitions from Env A
        self.axis_map = {0: 'x', 1: 'x', 2: 'y', 3: 'y', 4: 'z', 5: 'z'}
        self.last_axis = None

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

        fidelity = self.average_gate_fidelity(self.U_n, self.target_U)
        if fidelity < self.tolerance:
            reward = fidelity - 1.0
        else:
            reward = 0.0

        current_axis = self.axis_map[int(action)]
        if self.last_axis is not None and self.last_axis != current_axis:
            reward -= 2.0
        else:
            reward += 0.2

        self.last_axis = current_axis

        self.current_step += 1
        done = (fidelity >= self.tolerance) or (self.current_step >= self.max_steps)

        if done and fidelity >= self.tolerance:
            reward += 5.0 * (self.max_steps - self.current_step)

        obs = self._get_observation()
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

    def average_gate_fidelity(self, U, V):
        diff = U - V
        singular_values = np.linalg.svd(diff, compute_uv=False)
        return 1 - np.max(singular_values)

    def compute_reward(self, achieved_goals, desired_goals, info):
        n = achieved_goals.shape[0]
        rewards = np.zeros(n, dtype=np.float32)
        for i in range(n):
            U_n = np.array([
                [achieved_goals[i][0] + 1j*achieved_goals[i][1], achieved_goals[i][2] + 1j*achieved_goals[i][3]],
                [achieved_goals[i][4] + 1j*achieved_goals[i][5], achieved_goals[i][6] + 1j*achieved_goals[i][7]]
            ], dtype=complex)
            U_target = np.array([
                [desired_goals[i][0] + 1j*desired_goals[i][1], desired_goals[i][2] + 1j*desired_goals[i][3]],
                [desired_goals[i][4] + 1j*desired_goals[i][5], desired_goals[i][6] + 1j*desired_goals[i][7]]
            ], dtype=complex)

            fidelity = self.average_gate_fidelity(U_n, U_target)
            if fidelity < self.tolerance:
                rewards[i] = fidelity - 1.0
            else:
                rewards[i] = 0.0
        return rewards


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
            plt.savefig(os.path.join(self.save_path, "single_qbit_clustered_v1.png"))
        plt.close()

        plt.figure()
        plt.plot(self.episode_lengths, label="Episode Length")
        plt.title("Episode Length Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.legend()
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, "single_qbit_clustered_v1_length.png"))
        plt.close()

def evaluate_agent(model, vec_env, num_episodes=5):
    # Access the raw environment for direct attribute calls
    env = vec_env.envs[0]
    success_count = 0
    for _ in range(num_episodes):
        obs, info = env.reset()  # because your Env returns (obs, info)
        done = False
        target_U = env.target_U
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
        fidelity = env.average_gate_fidelity(env.U_n, target_U)
        if fidelity >= env.tolerance:
            success_count += 1
    return success_count / num_episodes


def objective(trial):
    # 1) Suggest the number of parallel envs
    n_envs = trial.suggest_int("n_envs", 1, 16)

    # 2) Suggest other hyperparams
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [5000, 10000, 20000, 50000, 100000])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.2, 1.0)
    learning_starts = trial.suggest_int("learning_starts", 10000, 100000)
    train_freq = trial.suggest_categorical("train_freq", [(1, 'step'), (2, 'step'), (4, 'step'), (8, 'step')])
    net_arch_depth = trial.suggest_categorical("net_arch_depth", [1, 2, 3])
    net_arch_width = trial.suggest_categorical("net_arch_width", [64, 128, 256, 512])
    device = trial.suggest_categorical("device", ["cpu", "cuda"])

    # Build net_arch
    net_arch = [net_arch_width] * net_arch_depth
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=nn.SELU,
    )

    # 3) Create vectorized env
    vec_env = make_vec_env(n_envs=n_envs, use_subproc=True)

    # 4) Build model
    model = DQN(
        'MultiInputPolicy',
        vec_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_freq= train_freq,
        buffer_size=buffer_size,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=exploration_fraction,
        learning_starts=learning_starts,
        verbose=0,
        device=device,
        policy_kwargs=policy_kwargs,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            goal_selection_strategy='future',
            n_sampled_goal=4,
        )
    )

    # 5) Train for 200k timesteps
    model.learn(total_timesteps=200_000)

    # 6) Evaluate
    # We can either evaluate on the vectorized environment (using the first env)
    # or create a separate single env. Let's do a separate single env to avoid
    # confusion with multiple parallel instances
    eval_env = make_vec_env(n_envs=1, use_subproc=False)  # single env
    success_rate = evaluate_agent(model, eval_env, num_episodes=10)

    # Return success_rate to maximize
    return success_rate


def make_env(seed=None, idx=0):
    """
    Return a function that when called, creates a new QuantumCompilerEnv,
    wraps it with Monitor, etc.
    """
    def _init():
        env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=0.98)
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


if __name__ == "__main__":
    # Example usage of Optuna
    import optuna

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
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("param_importances.png") 



# if __name__ == "__main__":
#     env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=0.98)
#     env = Monitor(env)

#     policy_kwargs = dict(
#         net_arch=[256, 256],
#         activation_fn=nn.SELU,
#     )

#     model = DQN(
#         'MultiInputPolicy',
#         env,
#         learning_rate=0.005,
#         batch_size=200,
#         train_freq=(1, 'episode'),
#         buffer_size=10000,
#         exploration_initial_eps=1.0,
#         exploration_final_eps=0.05,
#         exploration_fraction=0.99931,
#         verbose=0,
#         device='auto',
#         policy_kwargs=policy_kwargs,
#         replay_buffer_class=HerReplayBuffer,
#         replay_buffer_kwargs=dict(
#             goal_selection_strategy='future',
#             n_sampled_goal=4,
#         )
#     )

#     callback = PlottingCallback(save_path='./data')
#     model.learn(total_timesteps=200000, log_interval=100, callback=callback)

#     success_rate = evaluate_agent(model, env, num_episodes=10)
#     print(f"Success Rate: {success_rate * 100:.2f}%")
