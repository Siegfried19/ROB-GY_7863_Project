import time
import numpy as np
import mujoco
import mujoco.viewer

# ====== 加载模型 ======
MODEL_PATH = "unitree_go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ====== 参数 ======
# CPG
freq = 1.2                                 # 步频 Hz（更低更稳，1.0~1.5）
phase = np.array([0.0, np.pi, np.pi, 0.0]) # 对角步态：FL, FR, RL, RR
amp_hip   = 0.25                            # 髋外展幅度 (roll)
amp_thigh = 0.55                            # 大腿抬摆幅度 (pitch)
amp_knee  = 0.75                            # 小腿弯曲幅度

# PD
kp = np.array([100.0] * model.nu)           # 关节刚度（40~60）
kd = np.array([10.0]  * model.nu)           # 关节阻尼（5~8）

# 姿态反馈（越大越抗倾覆，但太大步态会僵硬）
k_roll_to_hip   = 0.5                       # roll 修正 → 髋roll
k_pitch_to_thigh= 0.5                       # pitch 修正 → 大腿pitch

# 支撑相判定
CONTACT_THRESH = 5.0                        # 足端竖直力 N
SUPPORT_THIGH  = 0.8                        # 支撑位（略屈膝）
SUPPORT_KNEE   = -1.6

# 初始站立姿
q_offset = np.array([0.0, 0.9, -1.5] * 4)

# 限幅/平滑
ctrl_min = model.actuator_ctrlrange[:, 0].copy()
ctrl_max = model.actuator_ctrlrange[:, 1].copy()
alpha_smooth = 0.3                          # 扭矩平滑（0.2~0.4）

# ====== 工具函数 ======
def quat_to_euler(qw, qx, qy, qz):
    # xyz 欧拉角（MuJoCo基座四元数）
    sinr_cosp = 2*(qw*qx + qy*qz)
    cosr_cosp = 1 - 2*(qx*qx + qy*qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2*(qw*qy - qz*qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2*(qw*qz + qx*qy)
    cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

legs = ["FL", "FR", "RL", "RR"]
def get_contact_forces(model, data):
    """返回4足竖直力数组 [FL, FR, RL, RR]"""
    foot_forces = np.zeros(4)
    for c in range(data.ncon):
        contact = data.contact[c]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        pair = (g1 or ""), (g2 or "")
        for i, leg in enumerate(legs):
            foot_name = leg + "_foot"
            if foot_name in pair:
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, f6)  # [fx,fy,fz, tx,ty,tz]
                foot_forces[i] += f6[2]
    return foot_forces

def cpg_reference(t):
    """生成12维参考关节角（按 3*i+{0,1,2} -> [hip_roll, thigh_pitch, knee]）"""
    q_ref = np.zeros(model.nu)
    for i in range(4):
        phi = 2*np.pi*freq*t + phase[i]
        q_ref[3*i + 0] = 0               # hip roll
        q_ref[3*i + 1] = q_offset[1] + amp_thigh * np.sin(phi)   # thigh pitch
        q_ref[3*i + 2] = -1.8 #q_offset[2] + amp_knee  * np.sin(phi+np.pi/2)  # knee
    return q_ref

# ====== 初始站立 ======
data.qpos[:] = 0  # 先清空
data.qpos[2] = 0.27  # base 高度
data.qpos[3:7] = [1, 0, 0, 0]  # base 姿态 (w,x,y,z)
data.qpos[7:] = np.array([
    0.0, 0.9, -1.8,   # FR
    0.0, 0.9, -1.8,   # FL
    0.0, 0.9, -1.8,   # RR
    0.0, 0.9, -1.8    # RL
])
mujoco.mj_forward(model, data)


# ====== 主循环 ======
with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()
    last_torque = np.zeros(model.nu)

    while viewer.is_running():
        t = time.time() - t0

        # 1) CPG 参考
        q_ref = cpg_reference(t)

        # # 2) 姿态反馈（抗倾覆）
        # qw, qx, qy, qz = data.qpos[3:7]
        # roll, pitch, _ = quat_to_euler(qw, qx, qy, qz)
        # # 对所有腿：用 roll 修正髋roll，用 pitch 修正大腿pitch
        # q_ref[0::3] -= k_roll_to_hip    * roll   # hip roll
        # q_ref[1::3] -= k_pitch_to_thigh * pitch  # thigh pitch

        # # 3) 接触检测：支撑相“锁定”更稳
        # foot_fz = get_contact_forces(model, data)
        # for i in range(4):
        #     if foot_fz[i] > CONTACT_THRESH:
        #         # 该脚在支撑：给它一个稳定支撑位
        #         q_ref[3*i + 1] = SUPPORT_THIGH
        #         q_ref[3*i + 2] = SUPPORT_KNEE

        # 4) PD 力矩 + 限幅 + 平滑
        q  = data.qpos[7:7+model.nu]
        qd = data.qvel[6:6+model.nu]
        qd[0],qd[6] = 0,0
      
        torque = kp * (q_ref - q) - kd * qd
        torque = np.clip(torque, ctrl_min, ctrl_max)
        torque = alpha_smooth * torque + (1 - alpha_smooth) * last_torque
        last_torque = torque.copy()

        # 5) 施加控制并步进
        data.ctrl[:] = torque
        time.sleep(0.1)
        mujoco.mj_step(model, data)
        viewer.sync()
