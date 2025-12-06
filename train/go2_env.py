import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import mujoco.viewer
from reward import compute_reward_walk,compute_reward_fly,compute_reward_refence_fly
from scipy.spatial.transform import Rotation as R  
from get_ref_action import get_ref_torque



class Go2EnvMoonWalk(gym.Env):
    def __init__(self, xml_path="../unitree_go2/scene_moon.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.num_actions = self.model.nu    # 动作数
        self.num_obs = 44              # 可自由定义观测维度
        self.viewer = None
        self.num_envs = 16
        # 定义 action/observation 空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)


    def reset(self, *, seed=None, options=None):
    # 必写
        super().reset(seed=seed)

        # Reset mujoco
        KEYFRAME_ID = 0   # 如果这是你 XML 的第一个 keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, KEYFRAME_ID)


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
        self.num_obs = 33             # 可自由定义观测维度
        self.viewer = None
  
        # 定义 action/observation 空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)
        self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "init")
        self.is_land = False

    def reset(self, *, seed=None, options=None):
    # 必写
        super().reset(seed=seed)
        self.data
        # Reset mujoco
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)

        # 通常再 forward 一下保证所有派生量（矩阵、接触等）更新
     
        self.is_land = False

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
        abad_index = [0, 3, 6, 9]
        target_angles [abad_index] = 0.0

        current_angles = self.data.qpos[7:19]
        current_vel    = self.data.qvel[6:18]
        t = self.data.time
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
        z = self.data.qpos[2]

        # 例如超过 1.0m 之后禁止继续喷火
        if z > 1.0:
            jet_max = jet_max*0.2     


        jet_norm = (jet_action + 1) / 2.0    # [-1,1] → [0,1]
        jet_force = jet_norm * jet_max
        # print("action",action)
        # print("joint_torque",joint_torque)
        # print("jet_force",jet_force)

        # ----------------------------------------------------
        # 3. 合并 ctrl
        # ----------------------------------------------------
        if t<0.05:
            joint_torque = 0
            jet_force = 0

        self.data.ctrl[:12] = joint_torque
        self.data.ctrl[12:16] = jet_force

        # 执行仿真
        mujoco.mj_step(self.model, self.data)

        # ----------------------------------------------------
        # 4. 返回
        # ----------------------------------------------------
        obs = self.get_observations()
        rewards,reward_each = self._get_reward(obs)
        terminated, reason = self._check_done(obs)
        truncated = False  # 暂时还没有时间截断机制
        info = {"termination_reason": reason,
                "pos_rd":reward_each[0],
                "vel_rd":reward_each[1],
                "ori_rd":reward_each[2],
                "tau_rd":reward_each[3],
                "land_rd":reward_each[4]}


        return obs, rewards, terminated, truncated, info

       
    def get_observations(self):
        data = self.data

        # Base pos & orientation
        x, y, z = data.qpos[:3]
        qw, qx, qy, qz = data.qpos[3:7]

        # Base velocity
        vx, vy, vz = data.qvel[:3]
        wx, wy, wz = data.qvel[3:6]

        # # Task info
        # dist_xy = np.sqrt(x*x + y*y)
        # radial_dir_x = x / (dist_xy + 1e-6)
        # radial_dir_y = y / (dist_xy + 1e-6)
   

        # --- Leg joint states (hip only) ---
        # Go2: 12 joints, order depends on your XML
        # Example: qpos[7:19] contains 12 joint angles
        joint_angles = data.qpos[7:19]         # length=12
        joint_vel    = data.qvel[6:18]         # length=12

        # Select hip joints only (0,1 for each leg) → total 8 dims
        hip_idx = [0,1, 3,4, 6,7, 9,10]         # adjust index mapping accordingly
        hip_angles = joint_angles[hip_idx]
        hip_vels   = joint_vel[hip_idx]

        # Last action (for smooth control)
        last_action = data.actuator_force[12:]
        obs = np.concatenate([
            np.array([
                x, y, z,
                qw, qx, qy, qz,
                vx, vy, vz,
                wx, wy, wz,
            ], dtype=np.float32),

            hip_angles.astype(np.float32),
            hip_vels.astype(np.float32),
            last_action
        ])

        return obs.astype(np.float32)

        # print("=====obvervation state shape==========")
        # print(self.data.qpos[7:].shape)
        # print(self.data.qvel[6:].shape)    # 跳过 base 线+角速度
        # print(self.data.actuator_force.shape)
        # print(self.data.sensor('imu_quat').data.shape)
        # print(self.data.sensor('imu_gyro').data.shape)
        # print(self.data.sensor('imu_acc').data.shape)
    
    def _get_reward(self,obs):
        done,info = self._check_done(obs)

        rewards,reward_each = compute_reward_refence_fly(self.data, done,self.is_land )
        return rewards,reward_each

    def _check_done(self, obs):
        qw, qx, qy, qz = self.data.qpos[3:7]
        t = self.data.time
        roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)

        z = self.data.qpos[2]
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        vz = self.data.qvel[2]
        # CRATER_RADIUS = 2
        # dist_xy = np.sqrt(x**2 + y**2)
        # escaped = dist_xy > CRATER_RADIUS
   

        # if escaped and z < 0.5 and abs(vz) < 0.3 and abs(roll)<0.5 and abs(pitch)<0.5: # land termiate
        #     return True, "success_landing"
    
        # if 3.2> x > 2.8 and z < 3.4 and abs(roll) < 0.5 and abs(pitch) < 0.5 and abs(yaw) < 0.5 :
        #    self.is_land = True
        #    return True, "landing"
    
        # if abs(roll) > 0.7 or abs(pitch) > 0.7 or abs(yaw) > 0.7:
        #    return True, "unstable_orientation"

        # if x>3.5 or z > 7.0:
        #     return True, "too_far"
        
        if t> 3.0:
            return True, "too_long"
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return True, "nan_error"

        return False, None

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_down")
            # self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            # self.viewer.cam.fixedcamid = cam_id
        self.viewer.sync()
