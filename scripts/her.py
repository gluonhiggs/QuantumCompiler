import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN, HerReplayBuffer
import torch.nn as nn
from torch.nn import SELU # Import SELU directly if needed, or use string "selu"
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback # Import EvalCallback
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time

# --- Constants ---
MAX_EPISODE_LENGTH = 130
# Define tolerance for distance metric (e.g., 0.1 or 0.2)
# Note: 0.99 AGF roughly corresponds to distance < 0.2
DISTANCE_TOLERANCE = 0.1
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

# --- Environment Definition ---
class QuantumCompilerEnv(gym.Env):
    """
    Gym environment for quantum compiling using HRC gates, HER,
    relative state observation, and distance-based sparse reward.
    """
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, gate_set, tolerance=DISTANCE_TOLERANCE, max_steps=MAX_EPISODE_LENGTH):
        super().__init__()
        self.gate_set = gate_set
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.num_gates = len(self.gate_set)

        # Observation space: Dict for HER
        flat_shape = (8,) # 2x2 matrix -> 4 complex numbers -> 8 floats
        self.observation_space = spaces.Dict({
            # Observation: Relative difference O_n = U_n^dagger * U_target
            'observation': spaces.Box(low=-1.5, high=1.5, shape=flat_shape, dtype=np.float32),
            # Goal: Target unitary U_target
            'desired_goal': spaces.Box(low=-1.5, high=1.5, shape=flat_shape, dtype=np.float32),
            # Achieved Goal: Current unitary U_n
            'achieved_goal': spaces.Box(low=-1.5, high=1.5, shape=flat_shape, dtype=np.float32)
        })
        self.action_space = spaces.Discrete(self.num_gates)

        # Internal state
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

    def _get_obs_dict(self):
        """Gets the observation dictionary for the current state."""
        # Use conjugate transpose for inverse of unitary (more stable)
        relative_U = np.conj(self.U_n).T @ self.target_U
        observation = self._flatten_complex_matrix(relative_U)
        achieved_goal = self._flatten_complex_matrix(self.U_n)
        desired_goal = self._flatten_complex_matrix(self.target_U)

        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        # Generate a new target unless one was set externally (for eval)
        if not hasattr(self, 'target_U') or self.target_U is None:
            self.target_U = get_haar_random_unitary()
        observation_dict = self._get_obs_dict()
        # Provide success info (important for Monitor wrapper and callbacks)
        info = {'is_success': self._is_success(self.U_n, self.target_U)}
        return observation_dict, info

    def set_target_unitary(self, target_U):
        """Sets a specific target unitary, e.g., for evaluation."""
        self.target_U = target_U
        # Reset internal state for the new target
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)

    def _get_distance(self, U_n_mat, target_U_mat):
        """Calculates Frobenius norm distance ||U_n - target_U||."""
        return np.linalg.norm(U_n_mat - target_U_mat, 'fro') # Use Frobenius 'fro' explicitly

    def _is_success(self, U_n_mat, target_U_mat):
        """Checks if the distance is within tolerance."""
        distance = self._get_distance(U_n_mat, target_U_mat)
        return distance <= self.tolerance

    def step(self, action):
        # Ensure action is valid
        if not self.action_space.contains(action):
             raise ValueError(f"Invalid action: {action}")

        gate_index = int(action)
        gate = self.gate_set[gate_index]
        # Apply gate: U_{n+1} = G_{n+1} @ U_n (standard convention)
        self.U_n = gate @ self.U_n
        self.current_step += 1

        observation_dict = self._get_obs_dict()
        # Reward is computed based on the *final* state relative to the goal
        reward = self.compute_reward(observation_dict['achieved_goal'],
                                     observation_dict['desired_goal'],
                                     {}) # info not needed by reward func itself

        # Check termination conditions
        is_success = self._is_success(self.U_n, self.target_U)
        terminated = is_success # End if goal reached
        truncated = (self.current_step >= self.max_steps) # End if max steps reached

        # Info dict for HER and monitoring
        info = {'is_success': is_success}
        # Add distance for potential logging/debugging
        # info['distance'] = self._get_distance(self.U_n, self.target_U)

        return observation_dict, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Computes sparse reward based on distance for HER.
        Reward is 0 if distance <= tolerance, otherwise SPARSE_REWARD_PENALTY.
        Handles single inputs or batches from HER buffer.
        """
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal[None, :] # Add batch dim
        if desired_goal.ndim == 1:
            desired_goal = desired_goal[None, :] # Add batch dim

        batch_size = achieved_goal.shape[0]
        rewards = np.full(batch_size, SPARSE_REWARD_PENALTY, dtype=np.float32) # Default penalty

        # Unflatten matrices for distance calculation
        achieved_U = self._unflatten_float_array(achieved_goal.reshape(-1, 8)) # Reshape needed if batched
        desired_U = self._unflatten_float_array(desired_goal.reshape(-1, 8))

        # Calculate distances for the batch
        distances = np.array([self._get_distance(achieved_U[i], desired_U[i]) for i in range(batch_size)])

        # Apply reward condition
        rewards[distances <= self.tolerance] = 0.0

        # Return scalar if batch_size=1, otherwise return array for HER
        return rewards[0] if batch_size == 1 and rewards.shape[0] == 1 else rewards

# --- Environment Creation Helper ---
def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = QuantumCompilerEnv(gate_set=gate_matrices, tolerance=DISTANCE_TOLERANCE, max_steps=MAX_EPISODE_LENGTH)
        # Important: Monitor wraps the env *before* VecEnv
        # Use unique seeds for each process
        env = Monitor(env, filename=f"{LOG_DIR}/monitor_{rank}")
        env.reset(seed=seed + rank)
        return env
    # set_random_seed(seed) # Not needed here, env.reset handles seed
    return _init

# --- Evaluation Function (Adapted for Distance) ---
def evaluate_agent(model, eval_env_fn, num_episodes=100, output_filename=None):
    """Evaluates the agent using distance metric."""
    # Create a single evaluation environment
    eval_env = eval_env_fn(0)() # Call the function returned by make_env

    success_count = 0
    total_solved_length = 0
    results_buffer = []

    start_time = time.time()
    print(f"Starting evaluation for {num_episodes} episodes...")

    output_file = None
    if output_filename:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        output_file = open(output_filename, 'w', buffering=8192)

    for i in tqdm(range(num_episodes), desc="Evaluating"):
        target_U = get_haar_random_unitary()
        eval_env.set_target_unitary(target_U) # Set the specific target
        obs, info = eval_env.reset() # Reset for the new target

        terminated = False
        truncated = False
        gate_sequence_indices = []
        ep_length = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            gate_sequence_indices.append(action.item()) # Store index
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_length += 1 # Count steps based on env interaction

        # Get final state and calculate distance
        final_U = eval_env.U_n
        distance = eval_env._get_distance(final_U, target_U)
        is_success = distance <= eval_env.tolerance

        if is_success:
            success_count += 1
            total_solved_length += ep_length # Use actual steps taken

        # Log detailed results if output file is provided
        if output_file:
            result = {
                "episode": i,
                "success": bool(is_success),
                "sequence_length": ep_length,
                "target_unitary": [[str(elem) for elem in row] for row in target_U.tolist()],
                "approximated_unitary": [[str(elem) for elem in row] for row in final_U.tolist()],
                "final_distance": float(distance),
                "sequence": [gate_descriptions[idx] for idx in gate_sequence_indices]
            }
            results_buffer.append(json.dumps(result))
            if len(results_buffer) >= 100: # Write in chunks
                output_file.write('\n'.join(results_buffer) + '\n')
                results_buffer.clear()

    # Write any remaining results
    if output_file and results_buffer:
        output_file.write('\n'.join(results_buffer) + '\n')
    if output_file:
        output_file.close()

    end_time = time.time()
    success_rate = success_count / num_episodes
    average_solved_length = total_solved_length / success_count if success_count > 0 else 0

    print("-" * 30)
    print(f"Evaluation Finished ({time.time() - start_time:.2f}s)")
    print(f"Success Rate: {success_rate:.4f} ({success_count}/{num_episodes})")
    print(f"Average Solved Episode Length: {average_solved_length:.2f}")
    print(f"Distance Tolerance Used: {eval_env.tolerance}")
    print("-" * 30)

    # Close the single eval env
    eval_env.close()

    return success_rate, average_solved_length


if __name__ == "__main__":
    # --- Configuration ---
    N_ENVS = 8 # Number of parallel environments (adjust based on CPU cores)
    TOTAL_TIMESTEPS = 20_000_000 # Adjust as needed (paper used ~20M)
    LOG_DIR = "./sb3_logs_hrc_dist/"
    MODEL_SAVE_PATH = f"{LOG_DIR}/dqn_her_hrc_dist_model"
    EVAL_FREQ = max(50_000 // N_ENVS, 500) # Evaluation frequency in training steps
    N_EVAL_EPISODES = 50 # Number of episodes for evaluation during training
    FINAL_EVAL_EPISODES = 1000 # Number of episodes for final evaluation
    USE_SUBPROC_VEC_ENV = True # Use SubprocVecEnv for true parallelism if N_ENVS > 1
    SEED = 42 # For reproducibility

    # Create Log directory
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Create Vectorized Environment ---
    print(f"Creating {N_ENVS} parallel environments...")
    vec_env_cls = SubprocVecEnv if USE_SUBPROC_VEC_ENV and N_ENVS > 1 else DummyVecEnv
    vec_env = make_vec_env(lambda rank: make_env(rank, SEED),
                           n_envs=N_ENVS,
                           seed=SEED,
                           vec_env_cls=vec_env_cls)

    # --- Define Model Hyperparameters (closer to paper, adjusted for SB3) ---
    policy_kwargs = dict(
        net_arch=[128, 128], # Paper's architecture
        activation_fn=nn.SELU # Paper's activation
    )

    replay_buffer_kwargs=dict(
            n_sampled_goal=4,             # Standard HER value
            goal_selection_strategy='future', # Standard HER strategy
            # online_sampling=True, # Default
            handle_timeout_termination=False # Important: Treat timeout (max_steps) as not successful for reward calc
        )

    # Calculate learning_starts based on buffer size and environments
    # Ensure enough steps are taken to fill buffer significantly before learning
    learning_starts = max(50_000, N_ENVS * MAX_EPISODE_LENGTH * 5) # e.g., wait for ~5 episodes per env * buffer depth

    model = DQN(
        'MultiInputPolicy',         # Necessary for Dict observation space
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,         # Paper value (0.0001)
        # batch_size=200 * N_ENVS,    # Scale batch size with n_envs? Or keep fixed? Paper: 200. Try fixed 200 first.
        batch_size=200,
        buffer_size=500_000,        # Paper value
        learning_starts=learning_starts, # Start learning after significant exploration
        gamma=0.99,                 # Standard discount factor
        # train_freq=(1, 'episode'),# Paper value. Tricky in SB3 DQN. Often (1, 'step') works well with HER.
        train_freq=(1, 'step'),     # Train every N steps (common with HER)
        gradient_steps=1,           # Default: How many gradients per training step
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,   # Decay epsilon over first 20% of total steps (adjust if needed)
        target_update_interval=500, # How often to update target network (adjust if needed)
        seed=SEED,                  # For reproducibility
        verbose=1,                  # Print training progress
        tensorboard_log=LOG_DIR,    # Log for TensorBoard
        device='cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
    )

    print(f"Using device: {model.device}")
    print("Model Hyperparameters:")
    print(f"  Learning Rate: {model.learning_rate}")
    print(f"  Batch Size: {model.batch_size}")
    print(f"  Buffer Size: {model.buffer_size}")
    print(f"  Learning Starts: {model.learning_starts}")
    print(f"  Train Freq: {model.train_freq}")
    print(f"  Exploration Fraction: {model.exploration_fraction}")
    print(f"  Target Update Interval: {model.target_update_interval}")
    print(f"  Policy Kwargs: {model.policy_kwargs}")
    print(f"  Replay Buffer Kwargs: {model.replay_buffer_kwargs}")
    print(f"  Tolerance (Distance): {DISTANCE_TOLERANCE}")

    # --- Setup Evaluation Callback ---
    # Create a *separate* function for the eval environment to avoid state issues
    eval_env_fn_callback = lambda rank: make_env(rank + N_ENVS, SEED) # Use different rank/seed offset
    # Use Monitor wrapper for EvalCallback to get episodic statistics
    eval_callback = EvalCallback(make_vec_env(eval_env_fn_callback, n_envs=1, vec_env_cls=DummyVecEnv), # Eval callback needs a VecEnv
                                 best_model_save_path=f'{LOG_DIR}/best_model/',
                                 log_path=f'{LOG_DIR}/eval_logs/',
                                 eval_freq=EVAL_FREQ,
                                 n_eval_episodes=N_EVAL_EPISODES,
                                 deterministic=True,
                                 render=False,
                                 warn=False) # Suppress warnings about Monitor wrapper

    # --- Training ---
    print(f"\nStarting training for {TOTAL_TIMESTEPS} timesteps...")
    start_train_time = time.time()
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    log_interval=1000, # Log training progress every 1000 calls
                    callback=eval_callback,
                    reset_num_timesteps=False # Set to True if starting fresh run
                   )
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save the final model
        final_model_path = f"{MODEL_SAVE_PATH}_final.zip"
        print(f"\nSaving final model to {final_model_path}")
        model.save(final_model_path)
        print(f"Training finished in {(time.time() - start_train_time)/3600:.2f} hours.")
        # Close the vectorized environment
        vec_env.close()

    # --- Final Evaluation ---
    print("\n--- Final Evaluation ---")
    eval_output_filename = f"{LOG_DIR}/final_evaluation_results_dist.jsonl"
    # Load the best model saved by the callback
    try:
        best_model_path = os.path.join(f'{LOG_DIR}/best_model/', 'best_model.zip')
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            model_to_eval = DQN.load(best_model_path, env=None) # Load structure first
        else:
            print("Best model not found, loading final model.")
            model_to_eval = DQN.load(final_model_path, env=None)
    except Exception as e:
         print(f"Error loading model: {e}. Using final model.")
         model_to_eval = DQN.load(final_model_path, env=None)

    # Create a fresh eval environment function for final eval
    eval_env_fn_final = lambda rank: make_env(rank + N_ENVS + 1, SEED) # Use different rank/seed offset
    evaluate_agent(model_to_eval, eval_env_fn_final, num_episodes=FINAL_EVAL_EPISODES, output_filename=eval_output_filename)

    print("\nScript finished.")