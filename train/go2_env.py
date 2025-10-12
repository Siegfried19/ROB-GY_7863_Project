import gym
from gym import spaces
import mujoco
import numpy as np
import mujoco.viewer
from reward import compute_reward,compute_reward_jump
from scipy.spatial.transform import Rotation as R  
from get_ref_action import get_ref_torque
class Go2EnvMoon(gym.Env):
    def __init__(self, xml_path="../unitree_go2/scene_moon.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.num_actions = self.model.nu    # 动作数
        self.num_obs = 36               # 可自由定义观测维度
        self.viewer = None
        self.num_envs = 16
        # 定义 action/observation 空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        obs = self.get_observations()
        return obs
    
    def step(self, action):
        self.ref_angle, self.ref_ctrl = get_ref_torque(self.model,self.data)
      
        # 将动作转化为关节控制力矩
        ctrl = np.clip(action, -1, 1)
        max_torque = self.model.actuator_ctrlrange[:12, 1]
 
        self.data.ctrl[:12] =  self.ref_ctrl + max_torque * ctrl[:12]
        self.data.ctrl[12:] = 0
        # 执行仿真一步
        mujoco.mj_step(self.model, self.data)

        obs = self.get_observations()
        reward = self._get_reward(obs)
        done, info = self._check_done(obs)
        
        return obs, reward, done,info 

    def get_observations(self):
        # 示例：返回位置 + 速度
        obs = np.concatenate([
        self.data.qpos[7:],    # 跳过 base 自由度的 7 (xyz + quat)
        self.data.qvel[6:],    # 跳过 base 线+角速度
        self.data.actuator_force[:12],
        ])

        return obs 
    
    def _get_reward(self,obs):
        # vx = self.data.qvel[0]   # X方向速度（前进方向）
        # vy = self.data.qvel[1]   # Y方向速度（侧移）
     
        # # 简单奖励
        # reward = vx - 0.5 * abs(vy)
        # if done:
        #     reward -= 5.0
        done, _ = self._check_done(obs)
      
        reward = compute_reward(self.data, done)
        return reward

    def _check_done(self, obs):
        qw, qx, qy, qz = self.data.qpos[3:7]
        roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)
        z = self.data.qpos[2]

        # if z < 0.12:
        #     return True, "fell_down"

        if abs(roll) > 0.7 or abs(pitch) > 1.0 or abs(yaw) > 1.5:
            return True, "unstable_orientation"

        if z > 5.0:
            return True, "too_high"
        
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return True, "nan_error"

        return False, None

    def my_render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def get_ref(self):
        return self.ref_angle, self.ref_ctrl
class Go2Env(gym.Env):
    def __init__(self, xml_path="../unitree_go2/scene.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.num_actions = self.model.nu    # 动作数
        self.num_obs = 46               # 可自由定义观测维度
        self.viewer = None
        self.num_envs = 16
        # 定义 action/observation 空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        obs = self.get_observations()
        return obs
    
    def step(self, action):
        # 将动作转化为关节控制力矩
        ctrl = np.clip(action, -1, 1)
        max_torque = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = ctrl * max_torque

        # 执行仿真一步
        mujoco.mj_step(self.model, self.data)

        obs = self.get_observations()
        reward = self._get_reward(obs)
        done = self._check_done(obs)

        return obs, reward, done, {}

    def get_observations(self):
        # 示例：返回位置 + 速度
        obs = np.concatenate([
        self.data.qpos[7:],    # 跳过 base 自由度的 7 (xyz + quat)
        self.data.qvel[6:],    # 跳过 base 线+角速度
        self.data.actuator_force[:12],
        self.data.sensor('imu_quat').data,
        self.data.sensor('imu_gyro').data,
        self.data.sensor('imu_acc').data,
        ])

        return obs 
    
    def _get_reward(self,obs):
        # vx = self.data.qvel[0]   # X方向速度（前进方向）
        # vy = self.data.qvel[1]   # Y方向速度（侧移）
     
        # # 简单奖励
        # reward = vx - 0.5 * abs(vy)
        # if done:
        #     reward -= 5.0
        done = self._check_done(obs)
      
        reward = compute_reward(self.data, done)
        return reward

    def _check_done(self, obs):
        # 如果狗倒地
        if self.data.qpos[2] < 0.2:
            return True
        return False

    def my_render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
