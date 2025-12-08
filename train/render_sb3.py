import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
import register_envs   
# or: from go2_env import Go2EnvMoonFly
import csv
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

class TrajectoryLogger:
    def __init__(self, filename="traj.csv"):
        self.filename = filename
        self.buffer = []   # 缓存，每轮 episode 内的所有数据

        # 文件不存在 → 创建表头
        if not os.path.exists(filename):
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t",
                    "x","y","z",
                    "vx","vy","vz",
                    "roll","pitch","yaw"
                ])

    def record(self, data):
        """保存到缓冲区，不立即写文件"""
        t  = data.time
        x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
        vx, vy, vz = data.qvel[0], data.qvel[1], data.qvel[2]

        quat = data.qpos[3:7]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)

        # 加入 buffer
        self.buffer.append([
            t, x, y, z, vx, vy, vz, roll, pitch, yaw
        ])

    def save_if_good(self, reward_sum):
        """episode 结束时调用：reward 好才保存，否则丢弃"""
        if reward_sum > 4000:
            print(f"Saving episode (reward={reward_sum}) ...")
            with open(self.filename, "a", newline="") as f:
                writer = csv.writer(f)
                for row in self.buffer:
                    writer.writerow(row)
        else:
            print(f"Discard episode (reward={reward_sum})")

        # 清空缓存
        self.buffer = []


# 加载模型（zip 文件）
# model = PPO.load("./output/sb3_fly/finishied/sb3_fly_but_cannot_landing.zip", device="cpu")
model = PPO.load("./output/sb3_fly/can_track_good_landing_x_vara_vel/sb3_fly_7000000_steps.zip", device="cpu")
# 创建环境

env = gym.make("Go2FlyingGround-v0")


obs, info = env.reset()
reward_sum = 0
reward_each = {"pos":0,
                "vel":0,
                "ori":0,
                "tau":0,
                "land":0}
epi_len =0
print(obs.shape)
logger = TrajectoryLogger("traj_trench.csv")
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

        #logger.save_if_good(reward_sum)
        epi_len, reward_sum = 0,0
        reward_each = {"pos":0,
                "vel":0,
                "ori":0,
                "tau":0,
                "land":0}
        obs, info = env.reset()


    # 渲染（你可以用自己的）
    env.render()
    logger.record(env.unwrapped.data)   # 记录数据但不写文件
    # env.my_render()   # 如果你想用自己的渲染



env.close()
