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
        xml_path="../unitree_go2/scene_moon_jet.xml",
        foot_friction=None,
        body_friction=None,
        crater_config=None,
        rank=0
        ):
        
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.rank = rank
        
        if crater_config is None:
            self.crater_config = {
                "size": 0.4, 
                "depth": 1.0, 
                "flat_ratio": 0.4
            }
        else:
            self.crater_config = crater_config
        
        if foot_friction is None or body_friction is None:
            self.foot_friction = [0.1, 0.005, 0.001]
            self.body_friction = 0.5
        else:
            self.foot_friction = foot_friction
            self.body_friction = body_friction
        
        self._modify_physics(self.foot_friction, self.body_friction)
        self._modify_terrain(self.crater_config)
            
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
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0 
        self.data.qpos[2] = 0.35
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
            
    def _modify_terrain(self, config):
        hfield_name = "moon_hf" # 请确保 scene_moon_jet.xml 里 hfield 叫这个名字
        hfield_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
        
        if hfield_id != -1:
            # 获取 XML 中定义的网格尺寸
            nrow = self.model.hfield_nrow[hfield_id]
            ncol = self.model.hfield_ncol[hfield_id]
            
            # 生成新数据
            # 使用 rank 作为种子的一部分，确保不同环境生成不同的微小噪声(如果有)
            # 或者即使参数相同，也可以通过 seed 引入差异
            new_data = get_crater_height_map(
                size=nrow, # 假设 nrow == ncol
                crater_size=config.get("size", 0.3),
                crater_depth=config.get("depth", 1.0),
                flat_ratio=config.get("flat_ratio", 0.2),
                seed=100 + self.rank 
            )
            
            # 写入内存
            start_addr = self.model.hfield_adr[hfield_id]
            expected_len = nrow * ncol
            
            if len(new_data) == expected_len:
                self.model.hfield_data[start_addr : start_addr + expected_len] = new_data
            else:
                print(f"[Error] Generated terrain size {len(new_data)} != Model hfield size {expected_len}")
        else:
            print(f"[Warning] HField '{hfield_name}' not found. Terrain modification skipped.")
    
    def _get_reward(self,obs):
        done,info = self._check_done(obs)
        reward = compute_reward_fly(self.data, done, info, self.crater_config)
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
   

        if escaped and z < 0.35 and abs(vz) < 0.3 and abs(roll)<0.5 and abs(pitch)<0.5: # land termiate
            return True, "success_landing"

        if abs(roll) > 0.7 or abs(pitch) > 0.9:
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
        
    # def _get_robot_body_ids(self, root_name="base"):
    #     root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    #     if root_id == -1:
    #         print(f"[Warning] Robot root body '{root_name}' not found.")
    #         return set()

    #     # 1. 构建父子关系表 (Parent -> Children List)
    #     # model.body_parentid[i] 存储了第 i 个 body 的父级 ID
    #     tree = {}
    #     for i in range(self.model.nbody):
    #         parent = self.model.body_parentid[i]
    #         if parent not in tree:
    #             tree[parent] = []
    #         tree[parent].append(i)

    #     # 2. BFS (广度优先搜索) 遍历获取整个子树的所有 ID
    #     robot_body_ids = set()
    #     queue = [root_id]
        
    #     while queue:
    #         curr_id = queue.pop(0)
    #         robot_body_ids.add(curr_id) # 加入白名单
            
    #         # 如果当前节点有子节点，将子节点加入队列继续查找
    #         if curr_id in tree:
    #             queue.extend(tree[curr_id])
                
    #     return robot_body_ids

    
    # def _modify_physics(self, foot_friction, body_friction):
    #     """
    #     foot_friction: [sliding, torsional, rolling] (3维列表)
    #     body_friction: float (标量)
    #     """
    #     # 1. 获取白名单 (属于机器人的所有 Body ID)
    #     robot_ids = self._get_robot_body_ids("base")

    #     # 2. 找出 4 只脚的 Geom ID，用于特殊处理
    #     foot_names = ["FL", "FR", "RL", "RR"]
    #     foot_geom_ids = set()
    #     for name in foot_names:
    #         fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    #         if fid != -1:
    #             foot_geom_ids.add(fid)
    #         else:
    #             print(f"[Warning] Foot geom '{name}' not found.")

    #     # 3. 遍历模型中所有的 Geom (包括机器人、地面、背景物体)
    #     for geom_id in range(self.model.ngeom):
            
    #         # --- 过滤 A: 跳过视觉几何体 (Visual Geoms) ---
    #         # contype==0 表示不参与碰撞检测，改了也没用
    #         if self.model.geom_contype[geom_id] == 0:
    #             continue

    #         # --- 过滤 B: 安全检查 ---
    #         # 查一下这个 geom 挂在哪个 body 下面？
    #         owner_body_id = self.model.geom_bodyid[geom_id]
            
    #         # 如果这个 body 不在机器人的白名单里 (比如它是 floor, body_id=0)，直接跳过！
    #         if owner_body_id not in robot_ids:
    #             continue

    #         # --- 4. 应用修改 ---
    #         if geom_id in foot_geom_ids:
    #             # 情况 1: 是脚 -> 设置全套 3D 摩擦力
    #             # 注意：必须确保传入的是 numpy array 或 list
    #             self.model.geom_friction[geom_id] = np.array(foot_friction, dtype=np.float64)
    #         else:
    #             # 情况 2: 是机器人的其他部位 (机身、大腿、小腿、髋部) -> 设置滑动摩擦力
    #             # 只修改第 0 个分量 (Sliding)，保持 Torsional/Rolling 为 XML 里的默认值
    #             self.model.geom_friction[geom_id, 0] = body_friction
