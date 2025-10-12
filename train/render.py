import torch
import gym
import numpy as np
from model import Actor
import time
import register_envs   # 确保 Go2Env 已经注册
from scipy.spatial.transform import Rotation as R
import pandas as pd
from get_ref_action import get_ref_torque
import mujoco
env = gym.make("Go2JumpingGround-v0")

env.model.opt.timestep = 0.005 
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

policy = Actor(obs_dim, act_dim)
policy.load_state_dict(torch.load("output/checkpoints_moon_walking_deephole/actor_max.pth", map_location="cpu"))
policy.eval()


obs= env.reset()
done = False
records = []
refernces = []




for step in range(10000):   # 运行2000步
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        action = policy.choose_action(obs_tensor)

    obs, reward, done, info = env.step(action)
    
   
    env.my_render()  
    if info!=None: 
        print(info)
    if done:
        obs = env.reset()


    qw, qx, qy, qz = env.data.qpos[3:7]
    roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)
    ref_angle, ref_ctrl = env.get_ref()
    print(pitch)
    record = {
        "time": env.data.time,
        **{f"qpos_{i}": env.data.qpos[7+i] for i in range(12)},
        **{f"torque_{i}": env.data.actuator_force[i] for i in range(12)},
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        **{f"action_{i}": action[i] for i in range(env.num_actions)},
    }
    records.append(record)

    refernce = {
        "time": env.data.time,
        **{f"ref_angle_{i}": ref_angle[i] for i in range(12)},
        **{f"ref_torque_{i}": ref_ctrl[i] for i in range(12)},
    }
    refernces.append(refernce)


# df = pd.DataFrame(records)
# df.to_csv("go2_moon_log.csv", index=False)
# print("saved to go2_moon_log.csv")


# df_ref = pd.DataFrame(refernces)
# df_ref.to_csv("go2_moon_reference_log.csv", index=False)
# print("saved to go2_moon_reference_log.csv")

env.close()
