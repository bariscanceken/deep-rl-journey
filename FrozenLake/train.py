import numpy as np
import gymnasium as gym
import random
import os
import pickle as pickle
import pygame 

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode= "human")

print("_____OBSERVATION SPACE_____ \n")
print("Observation Space", env.observation_space)
print("Sample observation", env.observation_space.sample())

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())

state_space = env.observation_space.n
print("There are ", state_space, " possible states")

action_space = env.action_space.n
print("There are ", action_space, " possible actions")

def initialize_q_table(state_space, action_space):
  Qtable = np.zeros((state_space, action_space))
  return Qtable

Qtable_frozenlake = initialize_q_table(state_space, action_space)

def greedy_policy(Qtable, state):
  action = np.argmax(Qtable[state][:])
  return action

def epsilon_greedy_policy(Qtable, state, epsilon):
  random_num = random.uniform(0,1)
  if random_num > epsilon:
    action = greedy_policy(Qtable, state)
  else:
    action = env.action_space.sample()

  return action

 
n_training_episodes = 10000  
learning_rate = 0.7         

 
n_eval_episodes = 100       
 
env_id = "FrozenLake-v1"    
max_steps = 99               
gamma = 0.95                  
eval_seed = []               

 
max_epsilon = 1.0             
min_epsilon = 0.05            
decay_rate = 0.0005           

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
  for episode in range(n_training_episodes):
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    state, info = env.reset()
    step = 0
    terminated = False
    truncated = False


    for step in range(max_steps):
      action = epsilon_greedy_policy(Qtable, state, epsilon)

      new_state, reward, terminated, truncated, info = env.step(action)

      Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

      if terminated or truncated:
        break

      state = new_state
  return Qtable

n_training_episodes = 10000
learning_rate = 0.7
n_eval_episodes = 100
env_id = "FrozenLake-v1"
max_steps = 99
gamma = 0.95
eval_seed = []
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005
 
env = gym.make(env_id, map_name="4x4", is_slippery=False)

state_space = env.observation_space.n
action_space = env.action_space.n 
Qtable_frozenlake = np.zeros((state_space, action_space))
Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)
Qtable_frozenlake

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
  episode_rewards = []
  for episode in range(n_eval_episodes):
    if seed:
      state, info = env.reset(seed=seed[episode])
    else:
      state, info = env.reset()
    step = 0
    truncated = False
    terminated = False
    total_rewards_ep = 0

    for step in range(max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = greedy_policy(Q, state)
      new_state, reward, terminated, truncated, info = env.step(action)
      total_rewards_ep += reward

      if terminated or truncated:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


def save_q_table(Qtable, filename="q-table-frozenlake.pkl"):

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "wb") as f:
        pickle.dump(Qtable, f)
        
    print(f"saved {save_path}")
    return save_path

see_episodes = 3
env_see = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

for episode in range(see_episodes):
  state, info = env_see.reset()
  terminated = False
  truncated = False

for step in range(max_steps):
  env_see.render()
  action = greedy_policy(Qtable_frozenlake, state)
  new_state, reward, terminated, truncated, info = env_see.step(action)
  state = new_state
  if terminated or truncated:
    break
print(f"episode {episode + 1} completed. Reward: {reward}")

save_q_table(Qtable_frozenlake)