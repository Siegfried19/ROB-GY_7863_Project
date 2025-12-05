import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation as R 
 
from get_ref_action import get_ref_torque
from reward import compute_reward_walk,compute_reward_fly

# Now we create terrian here
def get_crater_height_map(size=512, crater_size=0.3, crater_depth=1.0, flat_ratio=0.2, seed=42):
    
    # 创建网络坐标
    X, Y = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    
    # 计算半径
    img_radius = size / 2.0
    R_outer = crater_size * img_radius
    R_inner = flat_ratio * R_outer

    # 计算距离
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # 初始平面为 1.0 (高地)
    H = np.ones_like(r, dtype=np.float32)
    
    # 坑底为 0.0
    H[r <= R_inner] = 0.0
    
    # 过渡区 (Smoothstep)
    mask_transition = (r > R_inner) & (r < R_outer)
    denom = R_outer - R_inner
    if denom < 1e-6: denom = 1e-6
    
    t = (r[mask_transition] - R_inner) / denom
    smooth = t * t * (3.0 - 2.0 * t) # smoothstep
    H[mask_transition] = smooth
    
    # 应用深度
    Z = H * crater_depth

    Z = np.flipud(Z)
    return Z.ravel().astype(np.float32)
    
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
    def __init__(
        self, 
        xml_path="../unitree_go2/scene_moon_valley_jet.xml",
        foot_friction=None,
        body_friction=None,
        valley_width=None,
        rank=0
        ):
        
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.rank = rank
        
        if valley_width is None:
            self.valley_width = 3.0
        else:
            self.valley_width = valley_width
        
        if foot_friction is None or body_friction is None:
            self.foot_friction = [0.8, 0.02, 0.01]
            self.body_friction = 0.4
        else:
            self.foot_friction = foot_friction
            self.body_friction = body_friction
        
        self._modify_physics(self.foot_friction, self.body_friction)
        self._modify_terrain(self.valley_width)
            
        self.data = mujoco.MjData(self.model)
        
        # 定义 action/observation 空间
        self.num_actions = self.model.nu    # 动作数
        self.num_obs = 46               # 可自由定义观测维度
        self.viewer = None
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset mujoco
        mujoco.mj_resetData(self.model, self.data)
        
        # Initial stable pose
        self.data.qpos[0] = 1.5
        self.data.qpos[1] = 0.0 
        self.data.qpos[2] = 3.35
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        
        init_joint_angles = np.array([0, 0.9, -1.57] * 4, dtype=np.float32)
        self.data.qpos[7:19] = init_joint_angles
        self.data.qpos[7:19] += np.random.uniform(-0.05, 0.05, 12)
        
        # Simulate a few steps to settle down
        settle_steps = 100
        
        kp = 60.0
        kd = 3.0
        
        for _ in range(settle_steps):
            # 计算保持初始姿态所需的力矩
            current_angles = self.data.qpos[7:19]
            current_vel    = self.data.qvel[6:18]
            
            # PD 控制: Target 是 init_joint_angles
            torque = kp * (init_joint_angles - current_angles) - kd * current_vel
            
            # 限制力矩
            max_torque = self.model.actuator_ctrlrange[:12, 1]
            torque = np.clip(torque, -max_torque, max_torque)
            
            # 写入控制 (注意：Jet 推力设为 0)
            self.data.ctrl[:12] = torque
            self.data.ctrl[12:16] = 0.0 # 关掉火箭
            
            mujoco.mj_step(self.model, self.data)
    
        self.data.qvel[:6] = 0.0
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
        
        # 限制hip的角度
        # hip_limit = 0.1745
        # target_angles[0] = np.clip(target_angles[0], -hip_limit, hip_limit) # FL_hip
        # target_angles[3] = np.clip(target_angles[3], -hip_limit, hip_limit) # FR_hip
        # target_angles[6] = np.clip(target_angles[6], -hip_limit, hip_limit) # RL_hip
        # target_angles[9] = np.clip(target_angles[9], -hip_limit, hip_limit) # RR_hip
        
        target_angles[0] = 0
        target_angles[3] = 0
        target_angles[6] = 0
        target_angles[9] = 0

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
        reward, r_cross, r_pose, r_jet, r_soft = self._get_reward(obs)
        terminated, reason = self._check_done(obs, self.valley_width)
        truncated = False  # 你暂时还没有时间截断机制

        info = {"termination_reason": reason,
                "r_cross": r_cross,
                "r_pose": r_pose,
                "r_jet": r_jet,
                "r_soft": r_soft}

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
    
    def _modify_physics(self, foot_friction, body_friction):
        """
        简化版：基于排除法修改摩擦力
        """
        # 1. 先把 4 只脚的 ID 找出来
        foot_names = ["FL", "FR", "RL", "RR"]
        foot_ids = []
        for name in foot_names:
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if fid != -1:
                foot_ids.append(fid)

        # 2. 遍历所有 Geom
        for geom_id in range(self.model.ngeom):
            
            # --- 排除 1: 视觉物体 (Visual) ---
            # contype == 0 表示不参与碰撞，改了也没意义，直接跳过
            if self.model.geom_contype[geom_id] == 0:
                continue

            # --- 排除 2: 地面 (WorldBody) ---
            # 在 MuJoCo 中，挂在 <worldbody> 下的物体（如 floor），body_id 永远是 0
            # 只要跳过 body_id 为 0 的，就绝对不会改到地面
            if self.model.geom_bodyid[geom_id] == 0:
                continue

            # --- 剩下的肯定都是机器人身上的部件了 ---

            if geom_id in foot_ids:
                # 情况 A: 是脚 -> 设置 3D 摩擦力 [滑动, 扭转, 滚动]
                self.model.geom_friction[geom_id] = np.array(foot_friction, dtype=np.float64)
            else:
                # 情况 B: 是机器人的其他部位 (大腿、小腿、机身等) -> 设为身体摩擦力
                # 只修改滑动摩擦 (索引0)
                self.model.geom_friction[geom_id, 0] = body_friction
            
    def _modify_terrain(self, width):
        geom_name = "platform_end"
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        
        if geom_id != -1:
            half_len_x = self.model.geom_size[geom_id, 0]
            start_edge = 2.0
            new_left_edge = start_edge + width
            new_center_x = new_left_edge + half_len_x
            self.model.geom_pos[geom_id, 0] = new_center_x
            
    def _get_reward(self,obs):
        done,info = self._check_done(obs, self.valley_width)
        reward, r_cross, r_pose, r_jet, r_soft = compute_reward_fly(self.data, done, info, self.valley_width)
        return reward, r_cross, r_pose, r_jet, r_soft

    def _check_done(self, obs, valley_width):
        qw, qx, qy, qz = self.data.qpos[3:7]
        roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)

        z = self.data.qpos[2]
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        vz = self.data.qvel[2]
        cross = x > (2.0 + valley_width)
   

        if cross and z < 3.4 and abs(vz) < 0.3 and abs(roll)<0.5 and abs(pitch)<0.5: # land termiate
            return True, "success_landing"

        if abs(roll) > 0.7 or abs(pitch) > 0.9:
           return True, "unstable_orientation"
       
        if z < 2:
            return True, "fallen_in_gap"

        if z > 10:
            return True, "too_high"

        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return True, "nan_error"

        return False, None

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        