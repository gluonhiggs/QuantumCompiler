import gym
from gym import spaces
import numpy as np
from scipy.stats import unitary_group
from stable_baselines3 import PPO
import numpy as np

def rotation_gate(axis, angle):
    if axis == 'x':
        return np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)], [-1j * np.sin(angle / 2), np.cos(angle / 2)]], dtype=complex)
    elif axis == 'y':
        return np.array([[np.cos(angle / 2), -np.sin(angle / 2)], [np.sin(angle / 2), np.cos(angle / 2)]], dtype=complex)
    elif axis == 'z':
        return np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]], dtype=complex)

# Example of gate set
my_gate_set = [
    rotation_gate('x', np.pi / 128),
    rotation_gate('x', -np.pi / 128),
    rotation_gate('y', np.pi / 128),
    rotation_gate('y', -np.pi / 128),
    rotation_gate('z', np.pi / 128),
    rotation_gate('z', -np.pi / 128)
]


def generate_random_unitary():
    return unitary_group.rvs(2)  # Generates a random 2x2 unitary matrix

class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, tolerance):
        super(QuantumCompilerEnv, self).__init__()
        self.gate_set = gate_set
        self.tolerance = tolerance
        self.action_space = spaces.Discrete(len(self.gate_set))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.max_steps = 300  # Define a maximum step limit for each episode
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)  # Start with identity matrix
        self.target_U = generate_random_unitary()
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        return self._get_observation()
    
    def step(self, action):
        gate = self.gate_set[action]
        self.U_n = np.dot(gate, self.U_n)
        self.O_n = np.dot(np.linalg.inv(self.U_n), self.target_U)
        reward = self._compute_reward()
        done = self._check_done()
        obs = self._get_observation()
        info = {}
        self.current_step += 1
        return obs, reward, done, info
    
    def _get_observation(self):
        # Flatten the real and imaginary parts of O_n
        obs = np.concatenate([self.O_n.real.flatten(), self.O_n.imag.flatten()])
        return obs
    
    def _compute_reward(self):
        fidelity = self._average_gate_fidelity(self.U_n, self.target_U)
        return fidelity - 0.01 * self.current_step  # Larger reward for higher fidelity


    
    def _check_done(self):
        fidelity = self._average_gate_fidelity(self.U_n, self.target_U)

        print(f"Step: {self.current_step}, Fidelity: {fidelity}")

        if fidelity >= self.tolerance or self.current_step >= self.max_steps:
            return True
        else:
            return False
    
    def _average_gate_fidelity(self, U, V):
        return (np.abs(np.trace(np.dot(U.conj().T, V)))**2 + 2) / 6


env = QuantumCompilerEnv(gate_set=my_gate_set, tolerance=0.99)
model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003, n_steps=128, batch_size=64, device='cuda')
model.learn(total_timesteps=100000)

def evaluate_agent(model, env, num_episodes=100):
    success_count = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
        fidelity = env._average_gate_fidelity(env.U_n, env.target_U)
        if fidelity >= 0.99:  # Successful approximation based on fidelity
            success_count += 1
    return success_count / num_episodes


success_rate = evaluate_agent(model, env)
print(f'Success rate: {success_rate}')