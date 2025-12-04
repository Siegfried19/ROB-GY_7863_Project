import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
import register_envs   
# or: from go2_env import Go2EnvMoonFly

# 加载模型（zip 文件）
# model = PPO.load("./output/sb3_fly/finishied/sb3_fly_but_cannot_landing.zip", device="cpu")
model = PPO.load("./output/sb3_fly/sb3_fly_500000_steps.zip", device="cpu")
# 创建环境
env = gym.make("Go2FlyingingGround-v0")

obs, info = env.reset()
reward_sum = 0
print(obs.shape)

for step in range(200000):

    # SB3 的 predict 必须接 obs，返回 action
    action, _ = model.predict(obs, deterministic=False)

    #action = env.action_space.sample()
    # Gymnasium step API
    obs, reward, terminated, truncated, info = env.step(action)
    reward_sum += reward
    if terminated:
        print(info, reward_sum)
        reward_sum = 0
      
    time.sleep(0.01)
    # 渲染（你可以用自己的）
    env.render()
    # env.my_render()   # 如果你想用自己的渲染

    # episode 结束，自动 reset
    if terminated or truncated:
        obs, info = env.reset()

env.close()
