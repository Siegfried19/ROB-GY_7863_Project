import numpy as np
from scipy.spatial.transform import Rotation as R  
from dataclasses import dataclass


@dataclass
class LunarJumpCfg:
    z0: float = 0.95          # 站立高度
    g: float = 1.62            # 月球重力
    h_peak: float = 3.0        # 抬升高度
    L: float = 5.0             # 水平跨越距离
    T_buffer: float = 0.4      # 着地后缓冲时间（可选）

class LunarJumpRef:
    def __init__(self, cfg=LunarJumpCfg()):
        self.cfg = cfg
        self.v0z = float(np.sqrt(2*cfg.g*cfg.h_peak))
        self.t_up = self.v0z / cfg.g
        self.T = 2*self.t_up
        self.v0x = cfg.L / self.T

    def ref(self, t):
        c = self.cfg
        if t <= self.T:
            x = self.v0x * t-4
            z = c.z0 + self.v0z*t - 0.5*c.g*t*t
            vx = self.v0x
            vz = self.v0z - c.g*t
        else:
            # 简单缓冲：靠近地面期望（也可做 quintic）
            x = self.v0x * self.T  # 水平到 5m 后保持
            z = c.z0
            vx = 0.0               # 着地后期望停下（如需继续跑可改为 vx_ref=目标速度）
            vz = 0.0
        return x, z, vx, vz
    
@dataclass
class RewardJumpCfg:
    w_z: float = 3.0
    w_v: float = 2.0
    w_ori: float = 1.2
    w_land: float = 2.0
    lam_tau: float = 1e-3
    sz: float = 0.2
    sv: float = 0.2
    sori : float = 0.2
class RewardWalkCfg:
    vx_target = 1.0
    w_v, w_lat, w_ori, w_tau, w_delta, w_slip = 2.0, 0.5, 1.2, 1.0, 0.05, 0.5
    sigma_v, sigma_y, sigma_w, sigma_s = 0.3, 0.2, 0.6, 0.1
    kp, kr = 6.0, 6.0
    lam_tau, lam_delta = 0.002, 0.05
    r_done = -20.0

cfg = RewardWalkCfg()

def quat_to_euler_xyz(q):  # q = [w, x, y, z]
    # 转为 (x,y,z,w) 以适配 scipy
    rot = R.from_quat([q[1], q[2], q[3], q[0]])
    roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
    return roll, pitch, yaw

def compute_reward_walk(data, done):
    # 基座速度：qvel[0:3] 线速度; qvel[3:6] 角速度（MuJoCo惯例）
    vx, vy, wz = data.qvel[0], data.qvel[1], data.qvel[5]
    # 姿态
    qw, qx, qy, qz = data.qpos[3:7]
    roll, pitch, yaw = quat_to_euler_xyz([qw, qx, qy, qz])

    # (1) 前向速度
    rv = np.exp(-((vx - cfg.vx_target)**2) / (cfg.sigma_v**2))
    # (2) 横移/偏航
    rlat = np.exp(-(vy**2) / (cfg.sigma_y**2)) * np.exp(-(wz**2) / (cfg.sigma_w**2))
    # (3) 姿态
    rori = np.exp(-(cfg.kp * pitch**2 + cfg.kr * roll**2))
    # (4) 能耗（这里用扭矩 L1）
    tau = data.actuator_force[:].copy()  # 或 data.qfrc_actuator
    rtau = -cfg.lam_tau * np.sum(np.abs(tau))
 

    # (6) 防滑（可选：需计算接触足端切向速度，平地可先置 0）
    rslip = 0.0
   
    r = (cfg.w_v*rv + cfg.w_lat*rlat + cfg.w_ori*rori +
         cfg.w_tau*rtau + cfg.w_slip*rslip)

    if done:
        r += cfg.r_done
    return float(r)

def compute_reward_refence_fly(data, done, ref = LunarJumpRef(), rw= RewardJumpCfg()):
    t = data.time

    z  = float(data.qpos[2])
    x  = float(data.qpos[0])
    vx = float(data.qvel[0])
    vz = float(data.qvel[2])
    qw, qx, qy, qz = data.qpos[3:7]
    roll, pitch, _ = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', False)
  
    x_ref, z_ref, vx_ref, vz_ref = ref.ref(t)
 
    # 误差（仅用 z/v 跟踪；如需更严格也可加入 x 跟踪项）
    rz = np.exp(-(((z - z_ref)**2)+(x-x_ref)**2) / (rw.sz**2))
    rv = np.exp(-(((vx - vx_ref)**2 + (vz - vz_ref)**2)) / (rw.sv**2))
    rori = np.exp(-(roll**2 + pitch**2)/ (rw.sori**2))

    # 靠近地面时鼓励小竖直速度（软着陆）
    near_ground = (ref.cfg.z0 - 0.03) < z < (ref.cfg.z0 + 0.03)
    rland = np.exp(-abs(vz)) if near_ground else 0.0

    # 能耗惩罚
    tau = np.abs(np.array(data.actuator_force[12:], dtype=float)).sum()
    r_tau = -rw.lam_tau * tau
    #print(rz,rv,rori,r_tau)
   
    reward = rw.w_z*rz + rw.w_v*rv + rw.w_ori*rori #+  r_tau+ rw.w_land*rland 
    if done:
        reward -= 20.0
    return float(reward)
    
def compute_reward_fly(data, done, reason, vally_width):
    x, y, z = data.qpos[:3]
    vx, vy, vz = data.qvel[:3]
    cross = x > (2 + vally_width)
    high_enough = z > 7.0

    # Let it jump high
    r_high = 2.0 * np.exp(- (z - 7.0)**2 / (2 * 2**2))
    r_pos = -0.5 * (x - 1.5)**2 - 0.5 * y**2
    
    r_cross = r_high + r_pos
    # if cross == False:
    #     r_cross = (x - 1.5)*2.0
    # else:
    #     r_cross = 0
        
    # Pose stability
    qw, qx, qy, qz = data.qpos[3:7]
    roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz')
    r_pose = -1.0 * abs(pitch) - 0.5 * abs(roll)

    # et energy penalty
    motor = data.ctrl[0:12]
    jet = data.ctrl[12:16]
    r_jet = -0.000001 * np.sum(jet**2) - 0.000001 * np.sum(motor**2)

    # ------------------------
    # D) Soft landing reward (only after escape)
    # ------------------------
    r_soft = 0.0
    if cross:
        # 1. slow vertical speed
        r_soft_v = -2.0 * max(0, abs(vz) - 0.5)

        # 2. height control
        target_h = 3.4
        r_soft_h = -3.0 * abs(z - target_h)

        # 3. landing stability
        r_soft_pose = -2.0 * (abs(roll) + abs(pitch))

        r_soft = r_soft_v + r_soft_h + r_soft_pose
    r_soft = 0.0
    # ------------------------
    # E) small alive reward
    # ------------------------
    r_alive = 0.001

    reward = r_cross + r_pose + r_jet + r_soft + r_alive + cross*10.0

    # ------------------------
    # F) terminal bonus
    # ------------------------
    if done:
        if reason == "success_landing":
            reward += 1500     # 高奖励，鼓励逃出+落地
        if reason == "take_off_success":
            reward += 500
        else:
            reward -= 50

    return reward, r_cross, r_pose, r_jet, r_soft

# def compute_reward_fly2(data, done, reason):
#     x, y, z = data.qpos[:3]
#     vx, vy, vz = data.qvel[:3]

#     dist_xy = np.sqrt(x*x + y*y)
#     CRATER_RADIUS = 2.0
#     escaped = dist_xy > CRATER_RADIUS

#     # ------------------------
#     # A) Escape reward (outward speed)
#     # ------------------------
#     r_forward = 1.0 * vx

#     # ------------------------
#     # B) Pose stability
#     # ------------------------
#     qw, qx, qy, qz = data.qpos[3:7]
#     roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz')
#     r_pose = -2.0 * (abs(pitch) + abs(roll))
#     r_y = -1.0 * abs(y)
#     r_yaw = 1.0 * np.cos(yaw)    # yaw=0 → +1，偏离变小
#     r_pose = r_pose + r_y + r_yaw
#     # ------------------------
#     # C) Jet energy penalty
#     # ------------------------
#     jet = data.ctrl[12:16]
#     r_jet = -0.0 * np.sum(jet * jet)

#     # ------------------------
#     # D) Soft landing reward (only after escape)
#     # ------------------------
#     r_soft = 0.0
#     if escaped:
#         r_forward = 0
#         target_h = 0.3

#         # 垂直速度（只在接近地面时强惩罚）
#         w = np.clip((0.5 - z) / 0.5, 0, 1)
#         p_vz =  0.5 * w * abs(vz)

#         # 水平速度
#         p_hvel = 0.5 * (abs(vx) + abs(vy))

#         # 姿态
#         p_pose = 2.0 * (abs(roll) + abs(pitch))

#         # 高度偏差
#         p_height = 0.5 * abs(z - target_h)

#         # 总惩罚（越接近目标 → 越少惩罚）
#         r_soft = -(p_vz + p_hvel + p_pose + p_height)
#         print(r_soft)
     

#     # ------------------------
#     # E) small alive reward
#     # ------------------------
#     r_alive = 0.001
#     # print("r_forward",r_forward)
#     # print("r_pose",r_pose)
#     reward = r_forward + r_pose + r_jet + r_soft + r_alive

#     # ------------------------
#     # F) terminal bonus
#     # ------------------------
#     if done:
#         if escaped:
#             reward += 500
#         if reason == "success_landing":
#             reward += 1500     # 高奖励，鼓励逃出+落地
#             print("landing!!!!!!!!!")   
#         else:
#             reward -= 1000

#     return reward