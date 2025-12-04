import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
import register_envs   
# or: from go2_env import Go2EnvMoonFly

# 加载模型（zip 文件）
model = PPO.load("./output/sb3_fly/sb3_fly_500000_steps.zip", device="cpu")

# 创建环境
env = gym.make("Go2FlyingingGround-v0")

obs, info = env.reset()
acu_reward = 0.0
acu_r_escape_plane = 0.0
acu_r_escape_height = 0.0
acu_r_pose = 0.0
acu_r_jet = 0.0
acu_r_soft = 0.0

for step in range(200000):

    # SB3 的 predict 必须接 obs，返回 action
    action, _ = model.predict(obs, deterministic=True)

    # Gymnasium step API
    obs, reward, terminated, truncated, info = env.step(action)
    
    acu_reward += reward
    acu_r_escape_plane += info.get("r_escape_plane", 0.0)
    acu_r_escape_height += info.get("r_escape_height", 0.0)
    acu_r_pose += info.get("r_pose", 0.0)
    acu_r_jet += info.get("r_jet", 0.0)
    acu_r_soft += info.get("r_soft", 0.0)
    
    if terminated:
        print(info.get("termination_reason"), reward)
        print(f"Acu reward: {acu_reward:.2f}, escape_plane: {acu_r_escape_plane:.2f}, escape_height: {acu_r_escape_height:.2f}, pose: {acu_r_pose:.2f}, jet: {acu_r_jet:.2f}, soft: {acu_r_soft:.2f}")
    time.sleep(0.01)
    # 渲染（你可以用自己的）
    env.render()
    # env.my_render()   # 如果你想用自己的渲染

    # episode 结束，自动 reset
    if terminated or truncated:
        obs, info = env.reset()
        acu_reward = 0.0
        acu_r_escape = 0.0
        acu_r_pose = 0.0
        acu_r_jet = 0.0
        acu_r_soft = 0.0

env.close()
