import gymnasium as gym
import torch
import numpy as np
import argparse
from parameters import *
from PPO import Ppo
from collections import deque
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback,CallbackList,EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb
from go2_env import Go2EnvMoonFly
import wandb  
import register_envs
from stable_baselines3.common.monitor import Monitor
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Go2FlyingingGround-v0",
                    help='save path')
parser.add_argument('--save_path', type=str, default="output/sb3_fly",
                    help='save path')
parser.add_argument('--resume', type=bool, default=False,
                    help='whether to resume training')
parser.add_argument('--resume_path', type=str, default="./output/sb3_fly/sb3_fly_but_cannot_landing.zip",
                    help='path to resume training')
args = parser.parse_args()




def make_env():
    def _init():
        env = gym.make(args.env_name)
        env = Monitor(env)           # <<< 必须加！
        return env
    return _init



def main():
    wandb.init(
        project="Go2Moon_SB3",
        name="jumping",
        sync_tensorboard=True,
    )



    NUM_ENVS = 8# 并行环境数量，可根据 CPU 调整
    vec_env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])


    # ----------------------------
    # 5. Configure PPO model
    # ----------------------------
    if args.resume:
        print("resume training from:", args.resume_path)
      
        model = PPO.load(args.resume_path, device="cuda")
        model.set_env(vec_env)


    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            device="cuda",              # 使用 GPU
            n_steps=2048,               # 每次 rollout 步数（越大训练越稳定）
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            tensorboard_log = args.save_path+"/logs",
        )


    checkpoint_cb = EveryNTimesteps(
        n_steps=500000,
        callback=CheckpointCallback(
            save_freq=1,   
            save_path=args.save_path,
            name_prefix="sb3_fly"
        )
    )

    callback_list = CallbackList([
        checkpoint_cb,
        WandbCallback(
            gradient_save_freq=1000,
            model_save_path=args.save_path+"/models/",
            verbose=2
        )
    ])


    model.learn(
        total_timesteps=40_000_000,           # 训练 200 万步
        callback=callback_list,
    )

    model.save(args.save_path+"/sb3_fly_final")
    wandb.finish()

if __name__ == "__main__":
    main()