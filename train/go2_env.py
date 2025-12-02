import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import mujoco.viewer
from reward import compute_reward_walk,compute_reward_fly
from scipy.spatial.transform import Rotation as R  
from get_ref_action import get_ref_torque


class Go2EnvMoonWalk(gym.Env):
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

    def reset(self, *, seed=None, options=None):
    # 必写
        super().reset(seed=seed)

        # Reset mujoco
        mujoco.mj_resetData(self.model, self.data)

        obs = self.get_observations()
        info = {}

        return obs, info

    
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
      
        reward = compute_reward_walk(self.data, done)
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

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def get_ref(self):
        return self.ref_angle, self.ref_ctrl
class Go2EnvMoonFly(gym.Env):
    def __init__(self, xml_path="../unitree_go2/scene_moon_jet.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.num_actions = self.model.nu    # 动作数
        self.num_obs = 46               # 可自由定义观测维度
        self.viewer = None
  
        # 定义 action/observation 空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
    # 必写
        super().reset(seed=seed)

        # Reset mujoco
        mujoco.mj_resetData(self.model, self.data)

        obs = self.get_observations()
        info = {}

        return obs, info
    def step(self, action):
        action = np.clip(action, -1, 1)

        # ----------------------------------------------------
        # 1. 关节角控制（前12维） angles → torque via PD
        # ----------------------------------------------------
        joint_low  = self.model.jnt_range[:12, 0]
        joint_high = self.model.jnt_range[:12, 1]

        # 把 [-1,1] action 转成实际角度
        target_angles = joint_low + (action[:12] + 1) * 0.5 * (joint_high - joint_low)

        current_angles = self.data.qpos[7:19]
        current_vel    = self.data.qvel[6:18]

        kp = 40.0
        kd = 0.6

        joint_torque = kp * (target_angles - current_angles) - kd * current_vel

        # 限制在 actuator torque 范围内
        joint_max = self.model.actuator_ctrlrange[:12, 1]
        joint_torque = np.clip(joint_torque, -joint_max, joint_max)


        # ----------------------------------------------------
        # 2. 火箭推力控制（后 4 维）
        # ----------------------------------------------------
        jet_action = action[12:]
        jet_max = self.model.actuator_ctrlrange[12:, 1]

        jet_norm = (jet_action + 1) / 2.0    # [-1,1] → [0,1]
        jet_force = jet_norm * jet_max


        # ----------------------------------------------------
        # 3. 合并 ctrl
        # ----------------------------------------------------
        self.data.ctrl[:12] = joint_torque
        self.data.ctrl[12:16] = jet_force

        # 执行仿真
        mujoco.mj_step(self.model, self.data)

        # ----------------------------------------------------
        # 4. 返回
        # ----------------------------------------------------
        obs = self.get_observations()
        reward = self._get_reward(obs)
        terminated, reason = self._check_done(obs)
        truncated = False  # 你暂时还没有时间截断机制

        info = {"termination_reason": reason}

        return obs, reward, terminated, truncated, info

       
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
      
        return obs.astype(np.float32)
    
    def _get_reward(self,obs):
        done,info = self._check_done(obs)
        reward = compute_reward_fly(self.data, done, info)
        return reward

    def _check_done(self, obs):
        qw, qx, qy, qz = self.data.qpos[3:7]
        roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)

        z = self.data.qpos[2]
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        vz = self.data.qvel[2]
        CRATER_RADIUS = 2
        dist_xy = np.sqrt(x**2 + y**2)
        escaped = dist_xy > CRATER_RADIUS

        if escaped and z < 0.35 and abs(vz) < 0.3 and abs(roll)<0.5 and abs(pitch)<0.5:
            return True, "success_landing"

        if abs(roll) > 0.5 or abs(pitch) > 0.5:
           return True, "unstable_orientation"

        if z > 3.0:
            return True, "too_high"

        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return True, "nan_error"

        return False, None

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
