import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN, HerReplayBuffer
import torch.nn as nn
from torch.nn import SELU
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

filename = os.path.splitext(os.path.basename(__file__))[0]

def get_HRC_gates():
    V1 = (1/np.sqrt(5)) * np.array([[1, 2j],
                                    [2j, 1]], dtype=complex)
    V2 = (1/np.sqrt(5)) * np.array([[1, 2],
                                    [-2, 1]], dtype=complex)
    V3 = (1/np.sqrt(5)) * np.array([[1+2j, 0],
                                    [0, 1-2j]], dtype=complex)
    gate_matrices = [V1, V2, V3]
    gate_descriptions = ["V1", "V2", "V3"]
    return gate_matrices, gate_descriptions

# Generate Haar random target unitary matrix
def get_haar_random_unitary():
    z = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q @ np.diag(ph)

class QuantumCompilerEnv(gym.Env):
    def __init__(self, gate_set, accuracy=0.99, max_steps=130):
        super().__init__()
        self.gate_set = gate_set
        self.accuracy = accuracy
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(self.gate_set))
        
        # Adjust observation space for HER (state + goal)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-2, high=2, shape=(8,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-2, high=2, shape=(8,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-2, high=2, shape=(8,), dtype=np.float32)
        })

        self.reset()
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.U_n = np.eye(2, dtype=complex)
        # Do not change self.target_U if it has been set
        if not hasattr(self, 'training') or self.training:
            # Generate a random target unitary
            self.target_U = get_haar_random_unitary()
        return self._get_obs_dict(), {}
    
    def set_target_unitary(self, target_U):
        self.target_U = target_U

    def step(self, action):
        gate = self.gate_set[action]
        self.U_n = np.dot(self.U_n, gate)
        reward = self.compute_reward(self._get_achieved_goal(), self._get_desired_goal(), {})
        done = self._check_done()
        obs = self._get_obs_dict()
        self.current_step += 1
        truncated = False
        return obs, reward, done, truncated, {}
    
    def _get_obs_dict(self):
        # The observation is the current unitary matrix U_n
        observation = np.concatenate([self.U_n.real.flatten(), self.U_n.imag.flatten()])
        achieved_goal = observation.copy()
        desired_goal = np.concatenate([self.target_U.real.flatten(), self.target_U.imag.flatten()])
        return {
            "observation": observation.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32)
        }
    
    def _get_achieved_goal(self):
        observation = np.concatenate([self.U_n.real.flatten(), self.U_n.imag.flatten()])
        return observation.astype(np.float32)
    
    def _get_desired_goal(self):
        desired_goal = np.concatenate([self.target_U.real.flatten(), self.target_U.imag.flatten()])
        return desired_goal.astype(np.float32)
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # Ensure inputs are numpy arrays
        achieved_goal = np.array(achieved_goal)
        desired_goal = np.array(desired_goal)
        
        # If inputs are 1D, reshape them to 2D
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal[np.newaxis, :]
            desired_goal = desired_goal[np.newaxis, :]
        
        # Reshape achieved_goal and desired_goal to batch_size x 2 x 2 complex matrices
        U_n_real = achieved_goal[:, :4]
        U_n_imag = achieved_goal[:, 4:]
        U_n = U_n_real + 1j * U_n_imag
        U_n = U_n.reshape(-1, 2, 2)
        
        target_U_real = desired_goal[:, :4]
        target_U_imag = desired_goal[:, 4:]
        target_U = target_U_real + 1j * target_U_imag
        target_U = target_U.reshape(-1, 2, 2)
        
        # Compute fidelities vectorized
        fidelities = self.compute_accuracy(U_n, target_U)
        
        # Compute rewards
        rewards = np.where(fidelities >= self.accuracy, 100, -1 / self.max_steps)
        return rewards.squeeze()

    
    def _check_done(self):
        fidelity = self.compute_accuracy(self.U_n, self.target_U)
        return fidelity >= self.accuracy or self.current_step >= self.max_steps

    def compute_accuracy(self, U, V):
        """Compute the accuracy between two unitary matrices U and V using operator norm error."""
        U = np.array(U)
        V = np.array(V)
        
        if U.ndim == 2 and V.ndim == 2:
            # Single matrix case
            operator_norm_error = np.linalg.norm(U - V, ord=2)
            accuracy = 1 - operator_norm_error
            accuracy = np.clip(accuracy, 0, 1)  # Ensure accuracy is within [0, 1]
            return accuracy
        elif U.ndim == 3 and V.ndim == 3:
            # Batch case
            operator_norm_error = np.linalg.norm(U - V, ord=2, axis=(1, 2))
            accuracy = 1 - operator_norm_error
            accuracy = np.clip(accuracy, 0, 1)
            return accuracy
        else:
            # Handle mixed dimensions
            if U.ndim == 2:
                U = U[np.newaxis, :, :]
            if V.ndim == 2:
                V = V[np.newaxis, :, :]
            operator_norm_error = np.linalg.norm(U - V, ord=2, axis=(1, 2))
            accuracy = 1 - operator_norm_error
            accuracy = np.clip(accuracy, 0, 1)
            return accuracy.squeeze()


class PlottingCallback(BaseCallback):
    def __init__(self, verbose=1, save_path=None):
        super(PlottingCallback, self).__init__(verbose)
        self.save_path = save_path  # Path to save the plots
        self.num_episodes = 0
        self.successful_episodes = 0
        self.success_rates = []
        self.average_lengths = []
        self.episode_lengths = []
        self.successful_episode_lengths = []
        self.total_steps = 0
    
    def _on_step(self) -> bool:
        dones = self.locals['dones']
        infos = self.locals['infos']
        self.total_steps += 1
        for i in range(len(dones)):
            if dones[i]:
                self.num_episodes += 1
                ep_info = infos[i].get('episode')
                if ep_info:
                    ep_length = ep_info['l']
                    self.episode_lengths.append(ep_length)
                    if ep_info['r'] == 0:
                        self.successful_episodes += 1
                        self.successful_episode_lengths.append(ep_length)
                if self.num_episodes % 100 == 0:
                    success_rate = self.successful_episodes / self.num_episodes
                    avg_length = np.mean(self.successful_episode_lengths) if self.successful_episode_lengths else 0
                    self.success_rates.append(success_rate)
                    self.average_lengths.append(avg_length)
                    if self.verbose > 0:
                        print(f"Episodes: {self.num_episodes}, Success Rate: {success_rate*100:.2f}%, Avg Length: {avg_length:.2f}")
        return True
    
    def _on_training_end(self):
        # Plot success rates and average lengths over time
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(len(self.success_rates))*100, self.success_rates, label="Success Rate")
        plt.xlabel("Episodes")
        plt.ylabel("Success Rate")
        plt.title("Success Rate over Episodes")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(len(self.average_lengths))*100, self.average_lengths, label="Average Successful Episode Length")
        plt.xlabel("Episodes")
        plt.ylabel("Average Length")
        plt.title("Average Successful Episode Length over Episodes")
        plt.legend()
        
        plt.tight_layout()
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            plt.savefig(os.path.join(self.save_path, f"{filename}.png"))
        plt.close()

def evaluate_agent(model, env, test_num=1, output_filename=None):
    success_count = 0
    total_length = 0
    buffer = []
    with open(output_filename, 'w', buffering=8192) as f:
        for _ in tqdm(range(test_num), desc="Evaluating"):
            target_U = get_haar_random_unitary()
            # Set the target in the environment
            env.set_target_unitary(target_U)
            obs, _ = env.reset()
            done = False
            truncated = False
            gate_sequence = []
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                gate_sequence.append(action)
                obs, reward, done, truncated, _ = env.step(action)
            fidelity = env.compute_accuracy(env.U_n, env.target_U)
            success = fidelity >= env.accuracy
            if success:
                success_count += 1
                total_length += env.current_step
            # Write the results to the output file
            result = {
                "success": bool(success),
                "sequence_length": env.current_step,
                "target_unitary": [[str(elem) for elem in row] for row in target_U.tolist()],
                "approximated_unitary": [[str(elem) for elem in row] for row in env.U_n.tolist()],
                "fidelity": fidelity.item(),
                "sequence": [gate_descriptions[action] for action in gate_sequence]
            }

            buffer.append(json.dumps(result))
            if len(buffer) >= 100:
                f.write('\n'.join(buffer) + '\n')
                buffer.clear()
        if buffer:
            f.write('\n'.join(buffer) + '\n')
    success_rate = success_count / test_num
    average_length = total_length / success_count if success_count > 0 else 0
    return success_rate, average_length



if __name__ == '__main__':
    # Get the HRC gate set
    gate_matrices, gate_descriptions = get_HRC_gates()
    env = QuantumCompilerEnv(gate_set=gate_matrices, accuracy=0.9, max_steps=130)
    env = Monitor(env)
    
    policy_kwargs = dict(
        net_arch=[128, 128],
        activation_fn=SELU,
    )
    
    goal_selection_strategy = 'future'  # Strategy for HER
    
    replay_buffer_kwargs = dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
    )

    model = DQN(
        policy='MultiInputPolicy',
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        batch_size=200,
        train_freq=(1, 'episode'),
        buffer_size=100000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.99951,  # Approximately matches epsilon decay 0.99931
        verbose=1,
        device='cuda',  # Change to 'cpu' if not using GPU
    )

    # Define the custom plotting callback
    # plotting_callback = PlottingCallback(save_path='./data')
    
    # Train the model
    total_timesteps = 8000000 # Adjust based on your computational resources
    # model.learn(total_timesteps=total_timesteps, callback=plotting_callback)
    model.learn(total_timesteps=total_timesteps)
    success_rate, average_length = evaluate_agent(model, env, test_num=1000, output_filename=f"./data/{filename}.jsonl")
    print(f"Success Rate: {success_rate*100:.2f}%, Average Length: {average_length:.2f}")

