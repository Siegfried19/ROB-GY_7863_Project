import gym
import torch
import numpy as np
import argparse
from parameters import *
from PPO import Ppo
from collections import deque
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import make_vec_env, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb
from go2_env import Go2EnvMoonFly
import wandb  


parser = argparse.ArgumentParser()

parser.add_argument('--save_path', type=str, default="output",
                    help='save path')
parser.add_argument('--resume', type=bool, default=False,
                    help='whether to resume training')
parser.add_argument('--resume_iter', type=str, default='max',
                    help='iteration to resume training')
args = parser.parse_args()



# ----------------------------
# 2. Create single env function
# ----------------------------
def make_env():
    """
    Stable Baselines3 requires env to be created inside a function.
    """
    def _init():
        env = Go2EnvMoonFly(xml_path="../unitree_go2/scene_moon.xml")
        env = Monitor(env)  # record episode return, length, etc.
        return env
    return _init


# ----------------------------
# 3. Initialize wandb
# ----------------------------
wandb.init(
    project="Go2Moon_SB3",
    name="flying",
    sync_tensorboard=True,
)


# ----------------------------
# 4. Create vectorized environment
# ----------------------------
NUM_ENVS = 8  # 并行环境数量，可根据 CPU 调整
vec_env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])


# ----------------------------
# 5. Configure PPO model
# ----------------------------
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
    tensorboard_log="./tensorboard_log/",
)


# ----------------------------
# 6. Checkpoint callback (自动存模型)
# ----------------------------
checkpoint_cb = CheckpointCallback(
    save_freq=50000,                    # 每隔多少 step 保存一次
    save_path= args.save_path,
    name_prefix="sb3_fly",
)


# ----------------------------
# 7. Training
# ----------------------------
model.learn(
    total_timesteps=2_000_000,           # 训练 200 万步
    callback=[
        checkpoint_cb,
        WandbCallback(
            gradient_save_freq=1000,
            model_save_path="./models/",
            verbose=2
        )
    ]
)

model.save(args.save_path+"/sb3_fly_final")
wandb.finish()