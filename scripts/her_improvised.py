import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3 import DQN, HerReplayBuffer

filename = os.path.splitext(os.path.basename(__file__))[0]

# Generate Haar random target unitary matrix
def get_haar_random_unitary():
    z = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q @ np.diag(ph)

def rotation_gate(axis, angle):
    """Return a 2x2 complex unitary for a rotation of 'angle' about the given axis."""
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
        
# 1. Enhanced Gate Definitions
def get_advanced_gate_set():
    """Combines small rotations and Clifford gates"""
    gates = []
    descriptions = []
    
    # Small rotations
    for angle in [np.pi/64, np.pi/32]:
        for axis in ['x', 'y', 'z']:
            gates.append(rotation_gate(axis, angle))
            gates.append(rotation_gate(axis, -angle))
            descriptions += [f"R{axis}+{angle:.3f}", f"R{axis}-{angle:.3f}"]
    
    # Clifford gates
    clifford_gates = [
        np.array([[1, 0], [0, 1]], dtype=complex),        # I
        np.array([[0, 1], [1, 0]], dtype=complex),        # X
        np.array([[0, -1j], [1j, 0]], dtype=complex),     # Y
        np.array([[1, 0], [0, -1]], dtype=complex),       # Z
        np.array([[1, 1], [1, -1]]/np.sqrt(2)),            # H
        np.array([[1, 0], [0, 1j]], dtype=complex)        # S
    ]
    gates += clifford_gates
    descriptions += ['I', 'X', 'Y', 'Z', 'H', 'S']
    
    return gates, descriptions

# 2. Transformer-based Policy Network
class TransformerFeaturesExtractor(nn.Module):
    def __init__(self, observation_space, features_dim=256):
        super().__init__()
        self.embed_dim = 64
        self.num_heads = 4
        self.features_dim = features_dim
        
        # Unitary matrix encoder
        self.unitary_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            TransformerEncoderLayer(128, self.num_heads, dim_feedforward=256),
            TransformerEncoderLayer(128, self.num_heads, dim_feedforward=256)
        )
        
        # Temporal transformer for sequence processing
        self.temporal_transformer = TransformerEncoder(
            TransformerEncoderLayer(128, self.num_heads, dim_feedforward=256),
            num_layers=2
        )
        
        self.projection = nn.Linear(128, features_dim)
        
    def forward(self, observations):
        # Process desired goal
        goal_embed = self.unitary_encoder(observations['desired_goal'])
        
        # Process current state
        state_embed = self.unitary_encoder(observations['observation'])
        
        # Combine using temporal attention
        combined = torch.stack([state_embed, goal_embed], dim=1)
        transformer_out = self.temporal_transformer(combined)
        
        return self.projection(transformer_out.mean(dim=1))

# 3. Curriculum Learning Environment
class CurriculumQuantumCompilerEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.difficulty = 0
        self.max_difficulty = 5
        self.success_threshold = 0.8
        self.success_history = []
        
    def step(self, action):
        obs, reward, terminated, done, info = self.env.step(action)
        
        if done and info.get('is_success', False):
            self.success_history.append(1)
        elif done:
            self.success_history.append(0)
            
        # Update difficulty based on recent success rate
        if len(self.success_history) > 100:
            success_rate = np.mean(self.success_history[-100:])
            if success_rate > self.success_threshold:
                self.increase_difficulty()
                
        return obs, reward, terminated, done, info
    
    def increase_difficulty(self):
        if self.difficulty < self.max_difficulty:
            self.difficulty += 1
            self.env.max_steps = int(self.env.max_steps * 0.8)
            print(f"Increased difficulty to {self.difficulty}, new max steps: {self.env.max_steps}")
            
    def reset(self, seed=None, options=None):  # ✅ Fix: Add seed argument
        super().reset(seed=seed)  # ✅ Set random seed if provided

        # Generate simpler unitaries at lower difficulty levels
        if self.difficulty == 0:
            self.env.target_U = rotation_gate('z', np.pi/8)
        elif self.difficulty == 1:
            self.env.target_U = rotation_gate('x', np.pi/4) @ rotation_gate('z', np.pi/8)
        else:
            self.env.target_U = get_haar_random_unitary()
            
        obs = self.env.reset()  # ✅ Fix: Ensure this returns `(obs, info)`

        if isinstance(obs, tuple) and len(obs) == 2:
            return obs  # ✅ Already correct format (obs, info)
        else:
            return obs, {}


# 4. Enhanced Quantum Environment
class EnhancedQuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, accuracy=0.99, max_steps=130):
        super().__init__()
        self.gate_set = gate_set
        self.accuracy = accuracy
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(gate_set))
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(8,)),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(8,)),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(8,))
        })
        self.reset()
        
    def reset(self, seed=None, options=None):  # ✅ Fix: Add seed argument
        super().reset(seed=seed)
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        self.target_U = get_haar_random_unitary()
        return self._get_obs(), {}
    
    def step(self, action):
        """Applies the given action (gate index) to the current unitary matrix."""
        self.current_step += 1
        gate = self.gate_set[action]  # Get the gate corresponding to the action
        self.U_n = gate @ self.U_n  # Apply the gate

        obs = self._get_obs()  # Get new observation
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        terminated = reward >= 10
        done = self.current_step >= self.max_steps or reward >= 10  # Stop if max steps reached or perfect match
        info = {"is_success": reward >= 10}  # Track success metric

        return obs, reward, terminated, done, info  # ✅ Fix: Ensure `step()` returns (obs, reward, done, info)

    def _get_obs(self):
        def flatten_unitary(U):
            return np.concatenate([U.real.flatten(), U.imag.flatten()])
            
        return {
            "observation": flatten_unitary(self.U_n),
            "desired_goal": flatten_unitary(self.target_U),
            "achieved_goal": flatten_unitary(self.U_n)
        }
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # Enhanced reward function
        U_achieved = self._unflatten(achieved_goal)
        U_target = self._unflatten(desired_goal)
        
        fidelity = np.abs(np.trace(U_target.conj().T @ U_achieved)) / 2
        step_penalty = -0.01 * (self.current_step / self.max_steps)
        depth_penalty = -0.1 * (self.current_step / self.max_steps)
        
        reward = (
            5.0 * fidelity + 
            10.0 * (fidelity >= self.accuracy) +
            step_penalty +
            depth_penalty
        )
        return reward
    
    def _unflatten(self, array):
        real = array[:4].reshape(2, 2)
        imag = array[4:].reshape(2, 2)
        return real + 1j*imag

# 5. Hybrid Initialization with Expert Demonstrations
class ExpertBufferInitializer:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        
    def solovay_kitaev_decomposition(self, U, depth=3):
        """Simplified expert decomposition (placeholder)"""
        # Implement actual decomposition here
        return [0] * min(10, int(np.pi / 0.1))  # Dummy sequence
    
    def add_expert_trajectories(self, n_trajectories=1000):
        for _ in range(n_trajectories):
            target_U = get_haar_random_unitary()
            actions = self.solovay_kitaev_decomposition(target_U)
            
            self.env.envs[0].target_U = target_U  # Set target in single-env instance
            obs = self.env.reset()
            
            for action in actions:
                next_obs, reward, done, info = self.env.step([action])  # Wrap action in list for vectorized env
                
                # Store transition in HER replay buffer
                self.model.replay_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=np.array([action]),  # Ensure action is stored as NumPy array
                    reward=np.array([reward]),  # Convert reward to NumPy array
                    done=np.array([done])
                )

                obs = next_obs  # Update observation for next step
                
                if done:
                    break
                    
# 7. Enhanced Evaluation Module
class QuantumCompilerTester:
    def __init__(self, model, gate_descriptions, accuracy_threshold=0.95):
        self.model = model
        self.gate_descriptions = gate_descriptions
        self.accuracy_threshold = accuracy_threshold
        self.results = []
        
    def evaluate(self, n_tests=1000, max_steps=200, save_path=None):
        """Evaluate the agent on fresh random unitaries"""
        # Create evaluation environment
        gates, _ = get_advanced_gate_set()
        env = EnhancedQuantumCompilerEnv(gates, accuracy=self.accuracy_threshold, max_steps=max_steps)
        env = DummyVecEnv([lambda: env])
        
        success_count = 0
        total_fidelity = 0
        total_length = 0
        progress_bar = tqdm(total=n_tests, desc="Evaluating")
        
        for _ in range(n_tests):
            obs = env.reset()
            done = [False]
            current_step = 0
            gate_sequence = []
            
            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
                gate_sequence.append(action[0])
                current_step += 1
                
            # Calculate final fidelity
            fidelity = self._calculate_fidelity(env, obs)
            success = fidelity >= self.accuracy_threshold
            success_count += success
            total_fidelity += fidelity
            total_length += current_step if success else 0
            
            # Store results
            self._store_result(gate_sequence, fidelity, current_step, success, env)
            progress_bar.update(1)
            progress_bar.set_postfix({
                "Success Rate": f"{success_count/(_+1):.2%}",
                "Avg Fidelity": f"{total_fidelity/(_+1):.2f}"
            })
            
        progress_bar.close()
        self._save_results(save_path)
        return self._generate_report(n_tests, success_count, total_fidelity, total_length)
    
    def _calculate_fidelity(self, env, obs):
        """Extract final fidelity from environment"""
        achieved = env.envs[0]._unflatten(obs['achieved_goal'][0])
        target = env.envs[0]._unflatten(obs['desired_goal'][0])
        return np.abs(np.trace(target.conj().T @ achieved)) / 2
    
    def _store_result(self, sequence, fidelity, steps, success, env):
        """Store detailed decomposition results"""
        result = {
            "sequence": [self.gate_descriptions[a] for a in sequence],
            "fidelity": float(fidelity),
            "steps": steps,
            "success": success,
            "target_unitary": self._complex_to_json(env.envs[0].target_U),
            "result_unitary": self._complex_to_json(env.envs[0].U_n)
        }
        self.results.append(result)
    
    def _complex_to_json(self, matrix):
        """Convert complex matrix to JSON-serializable format"""
        return {
            "real": matrix.real.tolist(),
            "imag": matrix.imag.tolist()
        }
    
    def _save_results(self, save_path):
        """Save results to JSON file"""
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.results, f, indent=2)
                
    def _generate_report(self, n_tests, success_count, total_fidelity, total_length):
        """Generate final performance report"""
        return {
            "success_rate": success_count / n_tests,
            "average_fidelity": total_fidelity / n_tests,
            "average_success_length": total_length / success_count if success_count > 0 else 0,
            "total_tests": n_tests,
            "successful_tests": success_count
        }

# 8. Visualization Tools
def plot_results(report, save_path=None):
    """Visualize evaluation metrics"""
    plt.figure(figsize=(15, 5))
    
    # Success Rate
    plt.subplot(1, 3, 1)
    plt.bar(['Success', 'Failure'], 
            [report['success_rate'], 1-report['success_rate']])
    plt.title(f"Success Rate: {report['success_rate']:.2%}")
    
    # Fidelity Distribution
    plt.subplot(1, 3, 2)
    plt.hist([r['fidelity'] for r in self.results], bins=20)
    plt.title("Fidelity Distribution")
    plt.xlabel("Fidelity")
    plt.ylabel("Count")
    
    # Sequence Length vs Fidelity
    plt.subplot(1, 3, 3)
    lengths = [r['steps'] for r in self.results]
    fidelities = [r['fidelity'] for r in self.results]
    plt.scatter(lengths, fidelities, alpha=0.3)
    plt.title("Sequence Length vs Fidelity")
    plt.xlabel("Sequence Length")
    plt.ylabel("Fidelity")
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

# 6. Training Pipeline
def train():
    # Create environment
    gates, descriptions = get_advanced_gate_set()
    env = EnhancedQuantumCompilerEnv(gates, accuracy=0.95, max_steps=200)
    env = CurriculumQuantumCompilerEnv(env)
    env = DummyVecEnv([lambda: env])
    
    # Initialize model
    policy_kwargs = {
        "features_extractor_class": TransformerFeaturesExtractor,
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        "activation_fn": nn.ReLU
    }
    
    model = DQN(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        policy_kwargs=policy_kwargs,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Initialize with expert demonstrations
    expert_init = ExpertBufferInitializer(env, model)
    expert_init.add_expert_trajectories(1000)
    
    # Callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10000
    )
    
    # Train
    model.learn(
        total_timesteps=1_000_000,
        callback=[eval_callback],
        progress_bar=True
    )
    
    # Save model
    model.save("quantum_compiler_agent")
    
if __name__ == "__main__":
    train()