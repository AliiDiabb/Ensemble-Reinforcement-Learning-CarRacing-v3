import os
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from collections import Counter

NUM_AGENTS = 13  # Ensure this matches the number of saved models

def load_agents():
    loaded_agents = []
    for i in range(NUM_AGENTS):
        model_path = f"training_model/ppo_agent_{i}.zip"
        if os.path.exists(model_path):
            print(f"Loading agent {i+1} from {model_path}...")
            model = PPO.load(model_path)
            loaded_agents.append(model)
        else:
            print(f"Model file {model_path} not found. Please check your saved models.")
    return loaded_agents

def ensemble_action(observation, agents):
    # Ensure the observation has a batch dimension
    observation_batch = np.expand_dims(observation, axis=0)
    actions = []
    for agent in agents:
        action, _ = agent.predict(observation_batch, deterministic=True)
        actions.append(tuple(action[0]))  # Remove batch dimension
    most_common_action = Counter(actions).most_common(1)[0][0]
    return np.array(most_common_action)

def evaluate_and_log(agents, num_episodes=10):
    # Create an evaluation environment (using rgb_array mode so we don't open a window)
    eval_env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    # Create a SummaryWriter to log evaluation metrics
    writer = SummaryWriter("./eval_tensorboard_logs")
    
    for episode in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = ensemble_action(obs, agents)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        # Log episode reward to TensorBoard
        writer.add_scalar("Evaluation/EpisodeReward", total_reward, episode)
        print(f"Episode {episode+1}: Reward = {total_reward}")
    
    writer.close()
    eval_env.close()
    print("Evaluation metrics logged to './eval_tensorboard_logs'.")
    print("To view the graphs, run: tensorboard --logdir=./eval_tensorboard_logs")

if __name__ == "__main__":
    agents = load_agents()
    if agents:
        evaluate_and_log(agents)
