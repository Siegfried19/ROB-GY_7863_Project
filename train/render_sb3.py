import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
import register_envs   
# or: from go2_env import Go2EnvMoonFly

# 加载模型（zip 文件）
# model = PPO.load("./output/sb3_fly/finishied/sb3_fly_but_cannot_landing.zip", device="cpu")
model = PPO.load("./output/sb3_fly/can_track_bad_landing_x_vara_vel/sb3_fly_7000000_steps.zip", device="cpu")
# 创建环境
env = gym.make("Go2FlyingingGround-v0")

obs, info = env.reset()
reward_sum = 0
reward_each = {"pos":0,
                "vel":0,
                "ori":0,
                "tau":0,
                "land":0}
epi_len =0
print(obs.shape)

for step in range(200000):

    # SB3 的 predict 必须接 obs，返回 action
    action, _ = model.predict(obs, deterministic=False)

    #action = env.action_space.sample()
    # Gymnasium step API
    obs, reward, terminated, truncated, info = env.step(action)
    reward_each["pos"]+=info["pos_rd"]
    reward_each["vel"]+=info["vel_rd"]
    reward_each["ori"]+=info["ori_rd"]
    reward_each["tau"]+=info["tau_rd"]
    reward_each["land"]+=info["land_rd"]
   

    reward_sum += reward
    epi_len += 1
    if terminated:
        print("======")
        print(info["termination_reason"])
        print("episode len",epi_len,"episode reward",reward_sum)
        print(reward_each)
        epi_len, reward_sum = 0,0
        reward_each = {"pos":0,
                "vel":0,
                "ori":0,
                "tau":0,
                "land":0}
        obs, info = env.reset()

    time.sleep(0.01)
    # 渲染（你可以用自己的）
    env.render()
    # env.my_render()   # 如果你想用自己的渲染



env.close()
