import numpy as np
from scipy.spatial.transform import Rotation as R  
from dataclasses import dataclass


@dataclass
class LunarJumpCfg:
    z0: float = 0.5          # 站立高度
    g: float = 3.71          # 月球重力
    h_peak: float = 1.0        # 抬升高度
    L: float = 3.0             # 水平跨越距离
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
            x = self.v0x * t
            z = c.z0 + self.v0z*t - 0.5*c.g*t*t
            vx = self.v0x
            vz = self.v0z - c.g*t
        else:
            # 简单缓冲：靠近地面期望（也可做 quintic）
          
            z = c.z0
            vx = 1.0               # 着地后期望停下（如需继续跑可改为 vx_ref=目标速度）
            vz = 0.0
            x = self.v0x * self.T +vx*(t - self.T) # 水平到 3m 后保持
        return x, z, vx, vz
    
@dataclass
class RewardJumpCfg:
    w_z: float = 3.0
    w_v: float = 2.0
    w_ori: float = 1.2
    w_land: float = 2.0
    lam_tau: float = 1e-3
    sz: float = 0.5
    sv: float = 0.5
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

def compute_reward_refence_fly(data, done, reason, ref = LunarJumpRef(), rw= RewardJumpCfg()):
    t = data.time

    z  = float(data.qpos[2])
    x  = float(data.qpos[0])
    vx = float(data.qvel[0])
    vz = float(data.qvel[2])
    qw, qx, qy, qz = data.qpos[3:7]
    roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', False)
  
    x_ref, z_ref, vx_ref, vz_ref = ref.ref(t)
 
    # 误差（仅用 z/v 跟踪；如需更严格也可加入 x 跟踪项）
    rz = np.exp(-(((z - z_ref)**2)+(x-x_ref)**2) / (rw.sz**2))
    rv = - 0.5 * ( (vx - vx_ref)**2 + (vz - vz_ref)**2 )
    rori = np.exp(-(roll**2 + pitch**2)/ (rw.sori**2))+1.0 * np.cos(yaw) 
    # print("ref",x_ref, z_ref, vx_ref, vz_ref )
    # print("robot",x,z,vx,vz)
    # print("reward", rz,rv)
    # 靠近地面时鼓励小竖直速度（软着陆）
    near_ground = (ref.cfg.z0 - 0.3) < z < (ref.cfg.z0 + 0.3)
    rland = np.exp(-abs(vz)) if near_ground else 0.0

    # 能耗惩罚
    tau = np.abs(np.array(data.actuator_force[12:], dtype=float)).sum()
    r_tau = -rw.lam_tau * tau
    #print(rz,rv,rori,r_tau)
    r_alive = 0.001
    reward = rw.w_z*rz + rw.w_v*rv + rw.w_ori*rori + r_tau+ rw.w_land*rland +r_alive
    
 
    if done:
        reward -= 500
    return float(reward)
    
def compute_reward_fly(data, done, reason):
    x, y, z = data.qpos[:3]
    vx, vy, vz = data.qvel[:3]

    dist_xy = np.sqrt(x*x + y*y)
    CRATER_RADIUS = 2.0
    escaped = dist_xy > CRATER_RADIUS

    # ------------------------
    # A) Escape reward (outward speed)
    # ------------------------
    # 逃出坑奖励
    r_escape = 2.0 * dist_xy

    # 接近坑边时的减速权重
    w_edge = np.clip((dist_xy - CRATER_RADIUS*0.5) / (CRATER_RADIUS*0.5), 0, 1)

    # 奖励：在坑中央不减速；接近坑边时速度越小越好
    r_slow_x = w_edge * -abs(vx) 
    r_slow_z = w_edge * -abs(vz)

    r_flight = r_escape + r_slow_x + r_slow_z
    # ------------------------
    # B) Pose stability
    # ------------------------
    qw, qx, qy, qz = data.qpos[3:7]
    roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz')
    r_pose = -2.0 * (abs(pitch) + abs(roll))
    r_y = -1.0 * abs(y)
    r_yaw = 1.0 * np.cos(yaw)    # yaw=0 → +1，偏离变小
    r_pose = r_pose + r_y + r_yaw
    # ------------------------
    # C) Jet energy penalty
    # ------------------------
    jet = data.ctrl[12:16]
    r_jet = -1.0* np.mean(jet)/50

    # ------------------------
    # D) Soft landing reward (only after escape)
    # ------------------------
    r_soft = 0.0
    if escaped:
        r_flight = 0
        target_h = 0.3

        # ----------------------------
        # 1) 垂直速度奖励：落地时越接近 0 越好
        # ----------------------------

     
        if z > target_h+0.5: #
            r_descend = np.clip(-vz, -2.0, 3.0)   # vz<0 才有奖励
        else:
            r_descend =   np.exp(-3.0 * abs(vz))

        # ----------------------------
        # 2) 水平速度奖励：越慢越奖励
        # ----------------------------
        r_hvel = np.exp(-(abs(vx) + abs(vy)))

        # ----------------------------
        # 3) 姿态奖励：roll pitch 越接近 0 越好
        # ----------------------------
        r_pose = np.exp(-2.0 * (abs(roll) + abs(pitch)))

        # ----------------------------
        # 4) 高度奖励：落地高度越接近 target_h 越好
        # ----------------------------
        r_height = np.exp(-5.0 * abs(z - target_h))

        # ----------------------------
        # 总落地奖励（全正）
        # ----------------------------
        r_soft = (
            1.0 * r_descend +
            1.0 * r_hvel +
            1.0 * r_pose +
            1.0 * r_height
        )

        # print("===============")
        # print("landing reward:", r_soft)
        # print("postion",x,y,z)
        # print("vz reward:", r_descend, "hvel reward:", r_hvel)
        # print("pose reward:", r_pose, "height reward:", r_height)


    # ------------------------
    # E) small alive reward
    # ------------------------
    r_alive = 0.001
    # print("r_forward",r_forward)
    # print("r_pose",r_pose)
    if escaped:
        reward = r_jet + r_soft + r_alive
    else:
        reward = r_flight + r_pose + r_jet + r_soft + r_alive
    # ------------------------
    # F) terminal bonus
    # ------------------------
    if done:
        if escaped:
            reward += 500
        if reason == "success_landing":
            reward += 1500     # 高奖励，鼓励逃出+落地
            print("landing!!!!!!!!!")   
        else:
            reward -= 1000

    return reward
