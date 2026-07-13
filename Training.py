import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ----------------------------
# Configuration & Hyperparameters
# ----------------------------
TRAINING_TIMESTEPS = 300000  # Adjust timesteps as needed
NUM_AGENTS = 3
HYPERPARAMS_AGENT1 = {
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
}

HYPERPARAMS_AGENT2 = {
    "learning_rate": 0.00001,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": True,
    "sde_sample_freq": 4,
}


HYPERPARAMS_AGENT3 = {
    "learning_rate": 0.001,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
}

HYPERPARAMS=[HYPERPARAMS_AGENT1,HYPERPARAMS_AGENT2,HYPERPARAMS_AGENT3]

# ----------------------------
# Training Function: Train and Save Ensemble Agents
# ----------------------------
def train_and_save():
    # Create the training environment (using 'rgb_array' for faster headless training)
    train_env = gym.make("CarRacing-v3", render_mode="rgb_array")
    train_env = DummyVecEnv([lambda: train_env])

  
    i=0
    for HYPER in HYPERPARAMS:
        print(f"Training agent {i+1}/{NUM_AGENTS} with seed {i}...")
        model = PPO("CnnPolicy", train_env, seed=i, verbose=1, **HYPER)
        model.learn(total_timesteps=TRAINING_TIMESTEPS)
        model_path = f"training_model1/ppo_agent_{i}.zip"
        model.save(model_path)
        print(f"Saved agent {i+1} to {model_path}\n")
        i=i+1

if __name__ == "__main__":
    train_and_save()
