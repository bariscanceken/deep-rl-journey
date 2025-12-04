import gymnasium as gym
from stable_baselines3 import PPO
import os

model_name = "ppo-LunarLander-v3"
env_id = "LunarLander-v3"

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = gym.make(env_id, render_mode=)

print("model creating...")
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)

print("train starting...")
model.learn(total_timesteps=100000)

save_path = f"models/{model_name}"
model.save(save_path)

print(f"model saved to: {save_path}.zip")
