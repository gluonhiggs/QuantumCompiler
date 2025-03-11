import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# Basis gates with parameterized options
def get_basis_gates():
    """Single-qubit gates from BASIS_GATES, focusing on key parameterized and fixed gates."""
    gate_descriptions = ["rx", "ry", "rz", "u1", "u3", "x", "y", "z", "h", "s", "t"]
    param_counts = [1, 1, 1, 1, 3, 0, 0, 0, 0, 0, 0]  # Number of angles per gate
    return gate_descriptions, param_counts

def apply_gate(gate_idx, angles, gate_descriptions):
    """Apply a gate with given angles."""
    gate_name = gate_descriptions[gate_idx]
    if gate_name == "rx":
        θ = angles[0]
        return np.array([[np.cos(θ/2), -1j*np.sin(θ/2)], [-1j*np.sin(θ/2), np.cos(θ/2)]], dtype=complex)
    elif gate_name == "ry":
        θ = angles[0]
        return np.array([[np.cos(θ/2), -np.sin(θ/2)], [np.sin(θ/2), np.cos(θ/2)]], dtype=complex)
    elif gate_name == "rz":
        θ = angles[0]
        return np.array([[np.exp(-1j*θ/2), 0], [0, np.exp(1j*θ/2)]], dtype=complex)
    elif gate_name == "u1":
        λ = angles[0]
        return np.array([[1, 0], [0, np.exp(1j*λ)]], dtype=complex)
    elif gate_name == "u3":
        θ, φ, λ = angles
        return np.array([[np.cos(θ/2), -np.exp(1j*λ)*np.sin(θ/2)], 
                         [np.exp(1j*φ)*np.sin(θ/2), np.exp(1j*(φ+λ))*np.cos(θ/2)]], dtype=complex)
    elif gate_name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    elif gate_name == "y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif gate_name == "z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    elif gate_name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    elif gate_name == "s":
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    elif gate_name == "t":
        return np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

def random_unitary_haar():
    z = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    phase = d / np.abs(d)
    return q @ np.diag(phase)

class SingleQubitHybridEnv(gym.Env):
    def __init__(self, gate_descriptions, param_counts, max_steps=10, accuracy=0.95, lambda_penalty=0.1):
        super().__init__()
        self.gate_descriptions = gate_descriptions
        self.param_counts = param_counts
        self.max_steps = max_steps
        self.accuracy = accuracy
        self.lambda_penalty = lambda_penalty
        
        # Hybrid action space: discrete gate choice + continuous angles (max 3 params)
        self.action_space = spaces.Dict({
            "gate": spaces.Discrete(len(gate_descriptions)),
            "angles": spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32)
        })
        
        self.observation_space = spaces.Box(low=-2, high=2, shape=(8,), dtype=np.float32)  # Flattened current U
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.target_U = random_unitary_haar()
        self.current_U = np.eye(2, dtype=complex)
        return self._flatten_complex(self.current_U), {}
    
    def step(self, action):
        self.steps += 1
        gate_idx = action["gate"]
        angles = action["angles"][:self.param_counts[gate_idx]]  # Use only required angles
        
        gate = apply_gate(gate_idx, angles, self.gate_descriptions)
        self.current_U = self.current_U @ gate
        
        fidelity = self.compute_fidelity(self.current_U, self.target_U)
        done = bool(fidelity >= self.accuracy or self.steps >= self.max_steps)
        
        penalty = self.lambda_penalty * (self.steps / self.max_steps)
        reward = fidelity - 1 - penalty if fidelity < self.accuracy else 0
        
        obs = self._flatten_complex(self.current_U)
        info = {"is_success": fidelity >= self.accuracy}
        return obs, reward, done, False, info
    
    def compute_fidelity(self, U, V):
        return np.abs(np.trace(np.dot(U.conj().T, V))) / 2.0
    
    def _flatten_complex(self, U):
        return np.concatenate([U.real.flatten(), U.imag.flatten()]).astype(np.float32)

# Variational optimization
def variational_optimize(U_initial, U_target, gate_descriptions, param_counts, sequence):
    def fidelity_loss(params):
        U = np.eye(2, dtype=complex)
        param_idx = 0
        for gate_idx in sequence:
            n_params = param_counts[gate_idx]
            gate = apply_gate(gate_idx, params[param_idx:param_idx+n_params], gate_descriptions)
            U = U @ gate
            param_idx += n_params
        return -np.abs(np.trace(np.dot(U.conj().T, U_target))) / 2.0
    
    total_params = sum(param_counts[gate_idx] for gate_idx in sequence)
    initial_params = np.zeros(total_params)
    bounds = [(-np.pi, np.pi)] * total_params
    result = minimize(fidelity_loss, initial_params, method='L-BFGS-B', bounds=bounds)
    return result.x

# AQ* search
def aq_star_search(model, env, gate_descriptions, param_counts, max_depth=10):
    from heapq import heappush, heappop
    
    start_U = np.eye(2, dtype=complex)
    target_U = env.target_U
    queue = [(0, 0, [], start_U)]  # (cost, steps, sequence, unitary)
    visited = set()
    
    while queue:
        _, steps, seq, U = heappop(queue)
        if steps > max_depth:
            continue
        
        fidelity = env.compute_fidelity(U, target_U)
        if fidelity >= env.accuracy:
            return seq
        
        obs = env._flatten_complex(U)
        for gate_idx in range(len(gate_descriptions)):
            # Predict angles using the model
            action, _ = model.predict(obs, deterministic=True)
            angles = action["angles"][:param_counts[gate_idx]]
            new_U = U @ apply_gate(gate_idx, angles, gate_descriptions)
            
            new_key = tuple(new_U.flatten())
            if new_key not in visited:
                visited.add(new_key)
                new_seq = seq + [gate_idx]
                new_cost = -fidelity + steps
                heappush(queue, (new_cost, steps + 1, new_seq, new_U))
    
    return seq

def train_hybrid_agent():
    gate_descriptions, param_counts = get_basis_gates()
    env = SingleQubitHybridEnv(gate_descriptions, param_counts, max_steps=10)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])  # PPO requires vectorized env
    
    # Custom policy for hybrid action space
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Separate actor and critic networks
        activation_fn=nn.ReLU
    )
    
    model = PPO(
        "MultiInputPolicy",  # Handles Dict action space
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=policy_kwargs
    )
    
    model.learn(total_timesteps=500_000)  # Increased for hybrid complexity
    model.save("ppo_hybrid_single_qubit")
    return model, env, gate_descriptions, param_counts

def evaluate_agent(model, env, gate_descriptions, param_counts, n_evals=10):
    success_count = 0
    for i in range(n_evals):
        obs, _ = env.reset()
        done = False
        steps = 0
        gate_sequence = []
        angle_sequence = []
        
        # PPO prediction
        while not done and steps < env.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            gate_idx = action["gate"]
            angles = action["angles"][:param_counts[gate_idx]]
            gate_sequence.append(gate_idx)
            angle_sequence.extend(angles)
            obs, reward, done, _, info = env.step(action)
            steps += 1
        
        # AQ* refinement (gate sequence only, angles from PPO)
        refined_sequence = aq_star_search(model, env, gate_descriptions, param_counts)
        
        # Variational optimization
        optimized_params = variational_optimize(env.current_U, env.target_U, gate_descriptions, param_counts, refined_sequence)
        final_U = np.eye(2, dtype=complex)
        param_idx = 0
        for gate_idx in refined_sequence:
            n_params = param_counts[gate_idx]
            gate = apply_gate(gate_idx, optimized_params[param_idx:param_idx+n_params], gate_descriptions)
            final_U = final_U @ gate
            param_idx += n_params
        
        fidelity = env.compute_fidelity(final_U, env.target_U)
        success = fidelity >= env.accuracy
        success_count += int(success)
        
        print(f"Episode {i+1}/{n_evals} - Steps: {len(refined_sequence)}, Fidelity: {fidelity:.4f}, Success={success}")
        if success:
            print(f"Sequence: {[gate_descriptions[idx] for idx in refined_sequence]}")
    
    print(f"Success Rate: {success_count/n_evals:.2%}")

if __name__ == "__main__":
    model, env, gate_desc, param_counts = train_hybrid_agent()
    evaluate_agent(model, env, gate_desc, param_counts, n_evals=20)