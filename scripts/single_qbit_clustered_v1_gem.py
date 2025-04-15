import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # Import SubprocVecEnv
from stable_baselines3.her import HerReplayBuffer
import torch.nn as nn
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback # Use EvalCallback for better evaluation
import matplotlib.pyplot as plt


# --- Constants from Paper ---
TARGET_AGF = 0.99
MAX_EPISODE_LENGTH = 130
SPARSE_REWARD_PENALTY = -1.0 / MAX_EPISODE_LENGTH

# --- Gate Definitions ---
gate_descriptions = ["V1", "V2", "V3"]
gate_matrices = [
    (1/np.sqrt(5)) * np.array([[1, 2j], [2j, 1]], dtype=complex),  # V1
    (1/np.sqrt(5)) * np.array([[1, 2], [-2, 1]], dtype=complex),   # V2
    (1/np.sqrt(5)) * np.array([[1+2j, 0], [0, 1-2j]], dtype=complex)  # V3
]

def get_haar_random_unitary(dim=2):
    """Generates a Haar-random unitary matrix of dimension dim."""
    z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q @ np.diag(ph)

def average_gate_fidelity(U_target, U_n):
    """Calculates the average gate fidelity between two unitary matrices."""
    d = U_target.shape[0]
    fidelity = np.abs(np.trace(np.linalg.inv(U_target) @ U_n))**2
    # Formula for AGF for SU(d) - see Nielsen & Chuang Eq. 8.109 or similar sources
    # Or simpler relation: AGF = (fidelity + d) / (d^2 + d) ??? Check this relation source
    # Let's use the process fidelity relation F_pro = |Tr(U_target^dagger U_n)|^2 / d^2
    # AGF = (d * F_pro + 1) / (d + 1)
    process_fidelity = np.abs(np.trace(np.conj(U_target).T @ U_n))**2 / d**2
    agf = (d * process_fidelity + 1) / (d + 1)
    # Clip agf to avoid potential floating point issues slightly above 1
    return np.clip(agf, 0.0, 1.0)

# Define the QuantumCompilerEnv class
class QuantumCompilerEnv(gym.Env):
    # Important: Set metadata for HER
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, gate_set, target_agf=TARGET_AGF, max_steps=MAX_EPISODE_LENGTH):
        super().__init__()
        self.gate_set = gate_set
        self.target_agf = target_agf
        self.max_steps = max_steps
        self.num_gates = len(self.gate_set)

        # Observation space: Flattened complex matrix elements (real and imaginary)
        # Current unitary U_n (achieved_goal)
        # Target unitary U_target (desired_goal)
        # Relative unitary O_n = U_n^dagger * U_target (observation)
        # Bounds are approximately [-1, 1] for normalized unitaries
        flat_shape = (8,) # 2x2 matrix -> 4 complex numbers -> 8 floats
        self.observation_space = spaces.Dict({
            # Observation: Relative difference O_n = U_n^dagger * U_target
            # Using U_n^dagger instead of inv(U_n) is safer numerically for unitaries
            'observation': spaces.Box(low=-1.5, high=1.5, shape=flat_shape, dtype=np.float32), # Slightly wider bounds just in case
            'achieved_goal': spaces.Box(low=-1.5, high=1.5, shape=flat_shape, dtype=np.float32),
            'desired_goal': spaces.Box(low=-1.5, high=1.5, shape=flat_shape, dtype=np.float32)
        })
        self.action_space = spaces.Discrete(self.num_gates)

        self.target_U = None
        self.U_n = None
        self.current_step = 0

    def _flatten_complex_matrix(self, matrix):
        """Flattens a 2x2 complex matrix into an 8-element float array."""
        return np.concatenate([matrix.real.flatten(), matrix.imag.flatten()]).astype(np.float32)

    def _unflatten_float_array(self, flat_array):
        """Reconstructs a 2x2 complex matrix from an 8-element float array."""
        real_part = flat_array[:4].reshape(2, 2)
        imag_part = flat_array[4:].reshape(2, 2)
        return real_part + 1j * imag_part

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        self.target_U = get_haar_random_unitary()
        # print(f"Resetting env. New Target U:\n{self.target_U}") # Debug
        observation = self._get_observation()
        info = {'is_success': self._is_success(self.U_n, self.target_U)} # Include success info
        return observation, info

    def step(self, action):
        # Ensure action is valid
        if not self.action_space.contains(action):
             raise ValueError(f"Invalid action: {action}")

        gate_index = int(action) # Ensure action is integer index
        gate = self.gate_set[gate_index]

        # Apply the gate: U_{n+1} = Gate * U_n
        # IMPORTANT: Paper likely uses convention U_n = G_n * ... * G_1 * I
        # Check if the paper implies action applies on left or right.
        # Assuming U_n evolves by applying the next gate G_{n+1} as U_{n+1} = G_{n+1} * U_n
        self.U_n = gate @ self.U_n

        self.current_step += 1

        observation = self._get_observation()
        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], {}) # Info not used here
        is_success = self._is_success(self.U_n, self.target_U)
        terminated = is_success # Episode ends if goal is reached
        truncated = (self.current_step >= self.max_steps) # Episode ends if max steps reached

        # info dict for HER (is_success is crucial) and monitoring
        info = {'is_success': is_success}

        # Debugging:
        # if terminated or truncated:
        #      agf = average_gate_fidelity(self.target_U, self.U_n)
        #      print(f"Step {self.current_step}: Action {action}, AGF: {agf:.4f}, Success: {is_success}, Term: {terminated}, Trunc: {truncated}")

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # O_n = U_n^dagger * U_target (relative transform needed)
        # U_n^dagger is conjugate transpose, inv(U_n) for unitary
        relative_U = np.conj(self.U_n).T @ self.target_U
        obs = self._flatten_complex_matrix(relative_U)

        # Goals for HER
        achieved_goal = self._flatten_complex_matrix(self.U_n)
        desired_goal = self._flatten_complex_matrix(self.target_U)

        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }

    def _is_success(self, U_achieved, U_desired):
        """Check if achieved AGF meets the target."""
        agf = average_gate_fidelity(U_desired, U_achieved)
        return agf >= self.target_agf

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Sparse Reward Function (as described in the paper for HRC gates)
        # Input can be batched (from HER buffer) or single (from step)
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal[None, :] # Add batch dimension if single input
        if desired_goal.ndim == 1:
            desired_goal = desired_goal[None, :]

        batch_size = achieved_goal.shape[0]
        rewards = np.full(batch_size, SPARSE_REWARD_PENALTY, dtype=np.float32) # Default penalty

        for i in range(batch_size):
            u_achieved = self._unflatten_float_array(achieved_goal[i])
            u_desired = self._unflatten_float_array(desired_goal[i])
            if self._is_success(u_achieved, u_desired):
                rewards[i] = 0.0 # Success reward

        # Return scalar if batch_size is 1 (for step), array otherwise (for HER)
        return rewards[0] if batch_size == 1 and rewards.shape[0] == 1 else rewards


# --- Evaluation Function ---
def evaluate_agent(model, eval_env, num_episodes=100):
    """Evaluates the agent's success rate and average episode length."""
    total_success = 0
    total_length = 0
    solved_lengths = []

    for i in range(num_episodes):
        obs, info = eval_env.reset()
        terminated = False
        truncated = False
        ep_length = 0
        # Need to access the underlying env for target_U if using VecEnv wrapper
        # target_U = eval_env.get_attr("target_U")[0]
        # Or, just use the goal from the observation dict
        desired_goal_flat = obs['desired_goal']
        target_U = eval_env.envs[0]._unflatten_float_array(desired_goal_flat) # Access underlying env method

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_length += 1

        # Check success using the environment's state AFTER the last step
        # achieved_goal_flat = obs['achieved_goal'] # This is the state AFTER the step
        # current_U = eval_env.envs[0]._unflatten_float_array(achieved_goal_flat)
        # Need the actual final U_n from the environment instance
        current_U = eval_env.envs[0].U_n # Access final state directly
        is_success = eval_env.envs[0]._is_success(current_U, target_U)

        if is_success:
            total_success += 1
            solved_lengths.append(ep_length)
        total_length += ep_length
        # Optional: Print progress
        # if (i + 1) % (num_episodes // 10) == 0:
        #      print(f"Eval Episode {i+1}/{num_episodes} - Success: {is_success}, Length: {ep_length}")


    success_rate = total_success / num_episodes
    avg_length = total_length / num_episodes
    avg_solved_length = np.mean(solved_lengths) if solved_lengths else 0

    print("-" * 30)
    print(f"Evaluation Results ({num_episodes} episodes):")
    print(f"Success Rate: {success_rate:.4f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Average Solved Episode Length: {avg_solved_length:.2f}")
    print(f"Length Distribution (Solved): Mean={np.mean(solved_lengths):.2f}, Median={np.median(solved_lengths):.2f}, Min={np.min(solved_lengths) if solved_lengths else 'N/A'}, Max={np.max(solved_lengths) if solved_lengths else 'N/A'}")
    print("-" * 30)

    # Plot length distribution (like Fig 3a)
    if solved_lengths:
        plt.figure()
        plt.hist(solved_lengths, bins=range(1, MAX_EPISODE_LENGTH + 1), density=True, alpha=0.7)
        plt.title(f"Distribution of Solved Circuit Lengths (Success Rate: {success_rate:.2f})")
        plt.xlabel("Number of Gates")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.5)
        plt.savefig("solved_length_distribution_hrc.png")
        plt.close()
    else:
        print("No episodes solved, cannot plot length distribution.")

    return success_rate, avg_solved_length


if __name__ == "__main__":
    # --- Hyperparameters (Closer to Paper Table II) ---
    N_ENVS = 8 # Number of parallel environments
    LOG_DIR = "./tensorboard_logs_hrc/"
    MODEL_SAVE_PATH = "dqn_her_hrc_model"
    os.makedirs(LOG_DIR, exist_ok=True)

    # Environment creation function
    # Use DummyVecEnv for easier debugging, SubprocVecEnv for potential speedup
    vec_env = make_vec_env(lambda: QuantumCompilerEnv(gate_set=gate_matrices),
                           n_envs=N_ENVS,
                           vec_env_cls=DummyVecEnv) # Or SubprocVecEnv

    # Policy Network Kwargs (matching paper)
    policy_kwargs = dict(
        net_arch=[128, 128], # Two hidden layers as per paper
        activation_fn=nn.SELU # SELU activation as per paper
        # SB3 default initializers are often good, but paper specified lecun/glorot.
        # If needed, customization requires a custom policy class. Start with defaults.
    )

    # HER Replay Buffer Kwargs
    replay_buffer_kwargs=dict(
            n_sampled_goal=4,             # Standard value, paper's custom strategy might differ
            goal_selection_strategy='future', # Standard SB3 strategy
            # online_sampling=True, # Default is True, usually good
            # handle_timeout_termination=False # Treat timeout as failure, important for sparse rewards
        )

    # DQN Model Hyperparameters (matching paper where possible)
    model = DQN(
        'MultiInputPolicy', # Needed for Dict observation space
        vec_env,
        learning_rate=0.0001,        # Paper value
        batch_size=200 * N_ENVS,         # Paper value * n_envs ? Or just 200 total? Let's try 200 total. Needs clarification. Usually scales with n_envs. Let's try 200.
        buffer_size=500_000,       # Paper value
        learning_starts=10_000,     # Start learning after collecting some experience. Paper doesn't specify, use a reasonable value.
        gamma=0.99,                # Standard discount factor (Paper doesn't specify)
        # train_freq=(1, 'episode'), # Paper value - VERY IMPORTANT FOR DQN+HER performance
        train_freq=1,              # Check SB3 docs: integer means train every X steps. Tuple (freq, unit). Let's try every 1 step and adjust if needed. Paper says "every 1 episode" - SB3 might need (N_ep_steps, 'step') approx or callback. Let's try (1, 'step') first.
        gradient_steps=1,          # How many gradient steps per training call. Default 1.
        # Epsilon decay: Paper uses 0.99931 factor. SB3 uses linear decay via fraction.
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05, # Standard final value
        # exploration_fraction=0.1,   # Decay over 10% of total steps. Adjust based on convergence speed. Paper's decay is slow. Maybe 0.2 or 0.3?
        exploration_fraction=(4340 * N_ENVS) / 20_000_000, # Try to match paper's decay steps (calculated in thought process)
        target_update_interval=500, # How often to update target network. Common value. Paper doesn't specify.
        seed=42,                   # Reproducibility
        policy_kwargs=policy_kwargs,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device='cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
    )

    print(f"Using device: {model.device}")
    print("Model Hyperparameters:")
    print(f"  Learning Rate: {model.learning_rate}")
    print(f"  Batch Size: {model.batch_size}")
    print(f"  Buffer Size: {model.buffer_size}")
    print(f"  Train Freq: {model.train_freq}")
    print(f"  Exploration Fraction: {model.exploration_fraction}")
    print(f"  Policy Kwargs: {model.policy_kwargs}")
    print(f"  Replay Buffer Kwargs: {model.replay_buffer_kwargs}")


    # Setup evaluation environment (single instance, not vectorized)
    # Use Monitor to track episodic stats like success rate in TensorBoard
    eval_env_single = Monitor(QuantumCompilerEnv(gate_set=gate_matrices))
    eval_callback = EvalCallback(eval_env_single,
                                 best_model_save_path='./best_model_hrc/',
                                 log_path='./eval_logs_hrc/',
                                 eval_freq=max(50_000 // N_ENVS, 500), # Evaluate periodically
                                 n_eval_episodes=50, # Evaluate on 50 episodes
                                 deterministic=True,
                                 render=False)

    # Training
    TOTAL_TIMESTEPS = 20_000_000 # As in your original code
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    log_interval=1000, # Log training stats every 1000 calls to learn()
                    callback=eval_callback, # Use EvalCallback for robust evaluation
                    reset_num_timesteps=False # Set to False if continuing training
                   )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save the final model regardless of errors
        print(f"Saving final model to {MODEL_SAVE_PATH}.zip")
        model.save(MODEL_SAVE_PATH)
        vec_env.close() # Close the vectorized environment

    # Final Evaluation (using the best model saved by EvalCallback)
    print("\nLoading best model for final evaluation...")
    try:
        # Load the best model saved by the callback
        best_model = DQN.load(os.path.join('./best_model_hrc/', 'best_model'), env=None) # Load without env first
        # Create a single evaluation environment
        final_eval_env = DummyVecEnv([lambda: Monitor(QuantumCompilerEnv(gate_set=gate_matrices))])
        best_model.set_env(final_eval_env) # Set the environment for the loaded model
        evaluate_agent(best_model, final_eval_env, num_episodes=1000) # Evaluate on 1000 episodes
        final_eval_env.close()
    except FileNotFoundError:
        print("Best model not found. Evaluating the final model instead.")
        # Load the final model
        final_model = DQN.load(MODEL_SAVE_PATH, env=None)
        final_eval_env = DummyVecEnv([lambda: Monitor(QuantumCompilerEnv(gate_set=gate_matrices))])
        final_model.set_env(final_eval_env)
        evaluate_agent(final_model, final_eval_env, num_episodes=1000)
        final_eval_env.close()
    except Exception as e:
        print(f"An error occurred during final evaluation: {e}")
        import traceback
        traceback.print_exc()

    print("Script finished.")