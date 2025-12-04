import gymnasium as gym
import torch
import numpy as np
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback,CallbackList,EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb
import os

from go2_env import Go2EnvMoonFly
import register_envs

from parameters import *
from PPO import Ppo
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Go2FlyingingGround-v0",
                    help='save path')
parser.add_argument('--save_path', type=str, default="output/sb3_fly",
                    help='save path')
parser.add_argument('--resume', type=bool, default=False,
                    help='whether to resume training')
parser.add_argument('--resume_iter', type=str, default='max',
                    help='iteration to resume training')
parser.add_argument('--xml_path', type=str, default="../unitree_go2/scene_moon_jet.xml", 
                    help='path to mujoco xml')
args = parser.parse_args()

PARAM_RANGES = {
    "foot_slide": (0.03, 0.2),
    "foot_spin": (0.001, 0.02),
    "foot_roll": (0.001, 0.02),
    "crater_size": (0.4, 0.7),
    "crater_depth": (0.5, 1.5),
    "flat_ratio": (0.2, 0.4),
}

def make_env(rank, config):
    def _init():
        env = gym.make(
                    args.env_name, 
                    xml_path=args.xml_path,
                    foot_friction=config["foot_friction"], # 这是一个列表
                    body_friction=config["body_friction"], # 这是一个标量 (数值)
                    crater_config={
                        "size": config["crater_size"],
                        "depth": config["crater_depth"],
                        "flat_ratio": config["flat_ratio"]
                    },
                    rank=rank
                )
        env = Monitor(env)
        return env
    return _init
        
def main():
    wandb.init(
        project="Go2Moon_SB3",
        name="flying",
        sync_tensorboard=True,
    )

    # Numbers of parallel environments
    NUM_ENVS = 32
    env_configs = []
    
    # Configure different environment parameters for each env
    np.random.seed(42)
    print(f"{'='*20} Generating Random Environments {'='*20}")
    for i in range(NUM_ENVS):
        f_slide = np.random.uniform(*PARAM_RANGES["foot_slide"])
        f_spin  = np.random.uniform(*PARAM_RANGES["foot_spin"])
        f_roll  = np.random.uniform(*PARAM_RANGES["foot_roll"])
        foot_fric_vector = [f_slide, f_spin, f_roll]
        
        body_fric_scalar = f_slide
        
        c_size  = np.random.uniform(*PARAM_RANGES["crater_size"])
        c_depth = np.random.uniform(*PARAM_RANGES["crater_depth"])
        c_flat  = np.random.uniform(*PARAM_RANGES["flat_ratio"])
        
        env_config = {
            "foot_friction": foot_fric_vector,  # [slide, spin, roll]
            "body_friction": body_fric_scalar,  # equal to slide
            "crater_size":   c_size,
            "crater_depth":  c_depth,
            "flat_ratio":    c_flat,
        }
        
        env_configs.append(env_config)
        print(f"[Env {i}] Foot Fric: {np.round(foot_fric_vector, 3)} | "
              f"Body Fric: {body_fric_scalar:.3f} | "
              f"Depth: {c_depth:.2f}m")
    print(f"{'='*60}")
    
    # Generate vectorized environments
    env_fns = [make_env(i, env_configs[i]) for i in range(NUM_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    # Configure PPO model
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
        total_timesteps=200000000,           # 训练 200 万步
        callback=callback_list,
    )

    model.save(args.save_path+"/sb3_fly_final")
    wandb.finish()

if __name__ == "__main__":
    main()