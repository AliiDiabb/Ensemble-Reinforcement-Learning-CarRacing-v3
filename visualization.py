import os
import io
import zipfile
import torch
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from collections import Counter

NUM_AGENTS = 13


# ----------------------------
# Inspect a saved model's policy to detect input channels
# ----------------------------
def detect_input_channels(model_path):
    with zipfile.ZipFile(model_path) as z:
        with z.open("policy.pth") as f:
            buffer = io.BytesIO(f.read())
            state_dict = torch.load(buffer, map_location="cpu")

    for name, tensor in state_dict.items():
        if "cnn" in name and "weight" in name:
            # shape: (out_channels, in_channels, kernel_h, kernel_w)
            return tensor.shape[1]

    raise ValueError(f"Could not find a CNN weight layer in {model_path}")


# ----------------------------
# Load Trained Agents from Files
# ----------------------------
def load_agents():
    loaded_agents = []

    for i in range(NUM_AGENTS):
        model_path = f"training_model/ppo_agent_{i}.zip"

        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Please run the training code first.")
            continue

        print(f"Loading agent {i+1} from {model_path}...")

        in_channels = detect_input_channels(model_path)
        print(f"  Detected {in_channels} input channels for this model.")

        # Build a matching observation_space based on detected channels
        # CarRacing-v3 base image is 96x96
        obs_space = Box(low=0, high=255, shape=(in_channels, 96, 96), dtype=np.uint8)

        # Action space for CarRacing-v3 is unaffected by observation wrappers
        ref_env = gym.make("CarRacing-v3")
        act_space = ref_env.action_space
        ref_env.close()

        model = PPO.load(
            model_path,
            custom_objects={
                "observation_space": obs_space,
                "action_space": act_space,
            },
        )
        loaded_agents.append(model)

    return loaded_agents


# ----------------------------
# Ensemble Action Selection via Majority Voting
# ----------------------------
def ensemble_action(observation, agents):
    observation_batch = np.expand_dims(observation, axis=0)
    actions = []
    for agent in agents:
        action, _ = agent.predict(observation_batch, deterministic=True)
        actions.append(tuple(action[0]))
    most_common_action = Counter(actions).most_common(1)[0][0]
    return np.array(most_common_action)


# ----------------------------
# Evaluation Function with Visualization
# ----------------------------
def evaluate_ensemble(agents):
    eval_env = gym.make("CarRacing-v3", render_mode="human")
    obs, info = eval_env.reset()

    for _ in range(1800):
        action = ensemble_action(obs, agents)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            obs, info = eval_env.reset()

    eval_env.close()


if __name__ == "__main__":
    agents = load_agents()
    if agents:
        print("Evaluating the ensemble...")
        evaluate_ensemble(agents)
    else:
        print("No agents were loaded. Cannot run evaluation.")