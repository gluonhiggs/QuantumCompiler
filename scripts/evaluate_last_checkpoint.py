import os
from stable_baselines3 import DQN
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from her_original_checkpoint_replaybuffer import SingleQubitGoalEnv, get_universal_gates, CHECKPOINT_DIR

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

def evaluate_agent(model, env, n_evals=10):
    """Evaluates the agent on `n_evals` episodes."""
    success_count = 0
    for i in range(n_evals):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            steps += 1

        fidelity = env.compute_fidelity(env.current_U, env.target_U)
        success = (fidelity >= env.accuracy)
        success_count += int(success)

        print(f"Episode {i+1}/{n_evals} - Steps: {steps}, Fidelity: {fidelity:.4f}, Success={success}")

    print(f"Success Rate: {success_count/n_evals:.2%}")

if __name__ == "__main__":
    # Load the environment
    gate_matrices, gate_descriptions = get_universal_gates()
    env = SingleQubitGoalEnv(gate_set=gate_matrices, max_steps=130, accuracy=0.9)
    env = Monitor(env)

    # Find latest checkpoint
    latest_checkpoint = get_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        model = DQN.load(
            latest_checkpoint, 
            env=env, 
            replay_buffer_class=HerReplayBuffer,  # ✅ Ensure buffer is restored
        )
        model.set_env(env)  # Ensure env is properly set
        evaluate_agent(model, env, n_evals=20)
    else:
        print("No checkpoint found in:", CHECKPOINT_DIR)
