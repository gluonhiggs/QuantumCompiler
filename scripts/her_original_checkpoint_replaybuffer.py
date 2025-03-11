import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import torch
import torch.nn as nn

from stable_baselines3 import DQN
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

CHECKPOINT_DIR = "./checkpoints/"
CHECKPOINT_PREFIX = "her_dqn"

def rotation_gate(axis, angle):
    """Return a 2x2 complex unitary for rotation of 'angle' about the given axis."""
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

def get_universal_gates():
    """Example universal gate set: small rotations about X, Y, Z (± angles)."""
    gate_descriptions = []
    gate_matrices = []
    angle = np.pi / 128  # Adjust as needed
    for axis in ['x', 'y', 'z']:
        for sign in [+1, -1]:
            gate = rotation_gate(axis, sign * angle)
            desc = f"R{axis}{'+' if sign>0 else '-'}{angle:.3f}"
            gate_descriptions.append(desc)
            gate_matrices.append(gate)
    return gate_matrices, gate_descriptions

def random_unitary_haar():
    """Generate a random 2x2 unitary via the Haar measure."""
    # One simple approach: use the QR decomposition of a random complex matrix
    z = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    phase = d / np.abs(d)
    return q @ np.diag(phase)

class SingleQubitGoalEnv(gym.Env):
    """
    A goal-based environment for decomposing a random 2x2 unitary using a discrete set of gates.
    - observation['observation'] = the current partial product (U_n) flattened (real+imag)
    - observation['achieved_goal'] = same as above
    - observation['desired_goal'] = the random target unitary (real+imag)
    """
    def __init__(self, gate_set, max_steps=130, accuracy=0.95):
        super().__init__()
        self.gate_set = gate_set  # List of 2x2 complex gates
        self.max_steps = max_steps
        self.accuracy = accuracy
        
        # Discrete actions: pick an index from the gate set
        self.action_space = spaces.Discrete(len(gate_set))
        
        # Each 2x2 complex matrix flattened -> 4 real + 4 imag = 8 dims
        # Our observation is a Dict: {observation, desired_goal, achieved_goal}
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-2, high=2, shape=(8,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-2, high=2, shape=(8,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-2, high=2, shape=(8,), dtype=np.float32),
        })
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Random target unitary from the Haar measure
        self.target_U = random_unitary_haar()
        
        # Start with identity
        self.current_U = np.eye(2, dtype=complex)
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.steps += 1
        
        # Multiply current U by the chosen gate
        gate = self.gate_set[action]
        self.current_U = self.current_U @ gate
        
        # Check success
        fidelity = self.compute_fidelity(self.current_U, self.target_U)
        done = bool(fidelity >= self.accuracy or self.steps >= self.max_steps)
        
        # HER expects compute_reward to be called externally as well,
        # but we can compute a "step reward" for logging or shaping:
        reward = self.compute_reward(
            achieved_goal=self._flatten_complex(self.current_U),
            desired_goal=self._flatten_complex(self.target_U),
            info={}
        )
        
        obs = self._get_obs()
        info = {
            "is_success": fidelity >= self.accuracy
        }
        return obs, reward, done, False, info
    
    def _get_obs(self):
        """Return the Dict observation required by HER."""
        obs = self._flatten_complex(self.current_U)
        goal = self._flatten_complex(self.target_U)
        return {
            "observation": obs,
            "achieved_goal": obs,
            "desired_goal": goal
        }
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Required by HER. It's called on entire batches, so we must handle shape (batch_size, 8).
        Returns 0 if fidelity >= accuracy, else (fidelity - 1).
        """
        # shape might be (8,) for a single sample or (batch_size, 8) for a batch
        if achieved_goal.ndim == 1:
            # Expand dims to treat it as batch_size=1
            achieved_goal = achieved_goal[np.newaxis, :]
            desired_goal = desired_goal[np.newaxis, :]
        
        batch_size = achieved_goal.shape[0]
        rewards = np.zeros(batch_size, dtype=np.float32)
        
        for i in range(batch_size):
            U_achieved = self._unflatten_complex(achieved_goal[i])
            U_target = self._unflatten_complex(desired_goal[i])
            fidelity = self.compute_fidelity(U_achieved, U_target)
            if fidelity >= self.accuracy:
                rewards[i] = 0
            else:
                # partial shaping
                rewards[i] = fidelity - 1.0
        return rewards if batch_size > 1 else rewards[0]
    
    def compute_fidelity(self, U, V):
        """
        For 2x2 unitaries, define a simple fidelity measure.
        Example: 1 - ||U - V||_2 / 2 (bounded in [0,1]) or use |Tr(U^\dagger V)|/2
        """
        # We'll do trace-based measure:
        # fidelity = |Tr(U^\dagger V)| / 2
        # return np.abs(np.trace(np.dot(U.conj().T, V))) / 2.0
    
        # Or, if you prefer the Euclidean norm:
        return 1 - np.linalg.norm(U - V) / 2.0
    
    def _flatten_complex(self, U):
        """Flatten 2x2 complex into length-8 real vector."""
        return np.concatenate([U.real.flatten(), U.imag.flatten()]).astype(np.float32)
    
    def _unflatten_complex(self, arr):
        """Rebuild 2x2 complex from length-8 real vector."""
        arr = arr.reshape(-1)
        real = arr[:4].reshape(2, 2)
        imag = arr[4:].reshape(2, 2)
        return real + 1j * imag

def get_latest_checkpoint():
    """Finds the most recent checkpoint in the CHECKPOINT_DIR."""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    checkpoints = [
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith(".zip")
    ]
    return max(checkpoints, key=os.path.getctime) if checkpoints else None

def train_her_agent():
    from stable_baselines3 import DQN
    from stable_baselines3.her import HerReplayBuffer
    
    gate_matrices, gate_descriptions = get_universal_gates()
    
    # Create and wrap environment
    env = SingleQubitGoalEnv(gate_set=gate_matrices, max_steps=130, accuracy=0.9)
    
    # Optionally wrap in Monitor for logging
    env = Monitor(env)
    
    # DQN hyperparams
    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=nn.ReLU
    )
    
    # HER replay buffer params
    replay_buffer_kwargs = dict(
        n_sampled_goal=4,             # how many HER re-labeled goals
        goal_selection_strategy='future',  # 'future', 'episode', or 'final'
        # online_sampling=True,         # sample goals on the fly
    )
    
    latest_checkpoint = get_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        model = DQN.load(
            latest_checkpoint,
            env=env,
            replay_buffer_class=HerReplayBuffer,  # ✅ Fix: Restore replay buffer
            replay_buffer_kwargs=replay_buffer_kwargs
        )
        model.set_env(env)
    else:
        # Create DQN model with HER
        model = DQN(
            policy="MultiInputPolicy",  # Must be MultiInputPolicy for Dict obs
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            buffer_size=100000,
            learning_rate=1e-4,
            batch_size=256,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device='auto',  # 'cpu' or 'cuda'
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            exploration_fraction=0.3,  # fraction of training to reduce eps
            train_freq=(1, "episode"), # train after each episode
            target_update_interval=1000,
            max_grad_norm=10
        )
    # Define a checkpoint callback to save the model every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,  # Save model every 10,000 steps
        save_path=CHECKPOINT_DIR,  # Directory to store checkpoints
        name_prefix=CHECKPOINT_PREFIX
    )

    # Train for some timesteps
    total_timesteps = 200_000_000  # Increase as needed
    model.learn(total_timesteps=total_timesteps, 
                callback=checkpoint_callback,
                reset_num_timesteps=False)
    
    # Save
    model.save("her_dqn_single_qubit")
    return model, env, gate_descriptions

def evaluate_agent(model, env, n_evals=10):
    success_count = 0
    for i in range(n_evals):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        # Record the chosen gates
        gate_sequence = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            gate_sequence.append(action)
            obs, reward, done, _, info = env.step(action)
            steps += 1
        
        fidelity = env.compute_fidelity(env.current_U, env.target_U)
        success = (fidelity >= env.accuracy)
        success_count += int(success)
        
        print(f"Episode {i+1}/{n_evals} - Steps: {steps}, Fidelity: {fidelity:.4f}, Success={success}")
        if success:
            # Show the gate sequence
            # Possibly you have a 'gate_descriptions' array
            # gate_descriptions[action_index]
            pass
    print(f"Success Rate: {success_count/n_evals:.2%}")

if __name__ == "__main__":
    model, env, gate_desc = train_her_agent()
    evaluate_agent(model, env, n_evals=20)
