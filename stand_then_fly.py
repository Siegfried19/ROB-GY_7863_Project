import os
import mujoco
import mujoco.viewer
import numpy as np
import cvxpy
import time
import keyboard_controller
import matplotlib.pyplot as plt

LEG = ["FL","FR","RL","RR"]
site_id     = {leg: None for leg in LEG}
hip_bid     = {leg: None for leg in LEG}
hip_jid     = {leg: None for leg in LEG}    # HAA
thigh_jid   = {leg: None for leg in LEG}    # HFE
calf_jid    = {leg: None for leg in LEG}    # KFE
thr_act_id  = {leg: None for leg in LEG}

#PD gains for leg joint
kp_hip, kd_hip = 60.0, 4.0
kp_thigh, kd_thigh = 80.0, 6.0
kp_calf, kd_calf = 60.0, 4.0

# PID gains for body position and orientation
Kv = np.diag([10.0, 10.0, 10.0])
Ki_v = np.diag([0.3, 0.3, 0.4])
KR = np.diag([90.0, 120.0, 120.0])
Kw = np.diag([3.6, 3.6, 3.8])
# KI_tau = np.diag([0.3, 0.3, 0.3])
KI_tau = np.diag([1, 1, 1])

# Thruster optimization parameters
rho = 0.1                       # 平滑
theta_max = np.deg2rad(25.0)    # 喷口最大等效偏转
u_max = 200.0                   # 见 xml 的 ctrlrange 上限

g = None

# Controller parameters
f_prev = {leg: np.zeros(3) for leg in LEG}  # 上次喷口力向量，模是力
eta_v = np.zeros(3) # 积分误差速度
eta_tau = np.zeros(3)   # 积分误差角速度
error_log = []  # (time, ev) samples for plotting

# Utility functions
def get_R_from_xmat(xmat9):
    R = np.array(xmat9, dtype=float).reshape(3,3)
    return R

def vee(Rerr):
    # vee( 0.5 (R_d^T R - R^T R_d) )
    return np.array([Rerr[2,1]-Rerr[1,2], Rerr[0,2]-Rerr[2,0], Rerr[1,0]-Rerr[0,1]])*0.5

def hat(w):
    x, y, z = w
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]])
    
def Rz(dpsi):
    c, s = np.cos(dpsi), np.sin(dpsi)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], dtype=float)

def actuator_for_joint(model, joint_id):
    """Find actuator index that drives the given joint id. Returns -1 if none."""
    for a in range(model.nu):
        if model.actuator_trnid[a, 0] == joint_id:
            return a
    return -1

# Setup gravity
def setup_gravity(model):
    global g
    g = model.opt.gravity
    print(f"Gravity: {g}")

# Setup IDs
def setup_ids(model):
    for leg in LEG:
        site_id[leg]    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_rocket_site")
        hip_bid[leg]    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip")
        hip_jid[leg]    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_hip_joint")
        thigh_jid[leg]  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_thigh_joint")
        calf_jid[leg]   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_calf_joint")
        thr_act_id[leg] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_rocket_thruster")
        
# PD control for joint
def pd_for_joint(model, data, joint_id, target_pos):
    qpos_idx = model.jnt_qposadr[joint_id]
    qvel_idx = model.jnt_dofadr[joint_id]
    
    for leg in LEG:
        if joint_id == hip_jid[leg]:
            kp, kd = 60.0, 4.0        
            break
        elif joint_id == thigh_jid[leg]:
            kp, kd = 80.0, 6.0
            break
        elif joint_id == calf_jid[leg]:
            kp, kd = 60.0, 4.0
            break
    
    err = target_pos - data.qpos[qpos_idx]
    derr = -data.qvel[qvel_idx]
    return kp * err + kd * derr

# SOCP优化求解，保证为突优化
def solve_thrusters_SOCP(Fd_body, tau_d, r_list, a_list, u_max_list,
                         f_prev=None, theta_max=np.deg2rad(25), rho=0.1,
                         lamF=1000.0, lamT=1000.0):
    """
    同步优化 4 个喷口的“方向+大小”。输入/输出都在机体系。
    r_list: 4x3 机体系力矩臂
    a_list: 4x3 机体系锥轴（由当前 thigh 朝向的 -z 得到）
    """
    n = 4
    f = [cvxpy.Variable(3) for _ in range(n)]
    epsF = cvxpy.Variable(3)   # 合力松弛
    epsT = cvxpy.Variable(3)   # 力矩松弛

    cons = []
    cons += [cvxpy.sum(f) == Fd_body + epsF]
    # 力矩平衡：∑ r×f = τ + epsT
    torque_cols = [cvxpy.hstack([ r_list[i][1]*f[i][2] - r_list[i][2]*f[i][1],
                               r_list[i][2]*f[i][0] - r_list[i][0]*f[i][2],
                               r_list[i][0]*f[i][1] - r_list[i][1]*f[i][0] ]) for i in range(n)]
    cons += [cvxpy.sum(torque_cols, axis=0) == tau_d + epsT]

    # 圆锥可达 + 幅值上限
    tmax = np.tan(theta_max)
    for i in range(n):
        ai = a_list[i] / (np.linalg.norm(a_list[i]) + 1e-9)
        # 平行分量与垂直分量
        f_par = cvxpy.sum(cvxpy.multiply(ai, f[i]))            # = f_i^T a_i
        f_perp = f[i] - f_par * ai
        cons += [cvxpy.norm(f_perp, 2) <= tmax * (-f_par)]
        cons += [f_par <= 0]                              # 朝“向下”
        cons += [cvxpy.norm(f[i], 2) <= u_max_list[i]]

    # 目标：总推力 + 平滑 + 松弛罚
    obj_terms = [cvxpy.norm(fi, 2) for fi in f]
    if f_prev is not None:
        for i in range(n):
            obj_terms.append(rho * cvxpy.norm(f[i] - f_prev[i], 2))
    obj_terms += [lamF * cvxpy.norm(epsF, 2), lamT * cvxpy.norm(epsT, 2)]
    prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(obj_terms)), cons)
    prob.solve(solver=cvxpy.ECOS, warm_start=True, verbose=False)

    f_val = [fi.value if fi.value is not None else np.zeros(3) for fi in f]
    return np.array(f_val), epsF.value, epsT.value


# Flight controller
'''
1. 外环 - 计算期望的机体力和力矩 - 所有速度以及位置控制都是基于当前坐标系的
2. 内环 - 优化计算喷口推力（机体系） - 如何保证是突优化？
R_d: 期望机体系在世界系的旋转矩阵
'''
def control(model, data, vd_body, rotation, J0, omega_d_desired=np.zeros(3), omegadot_d_desired=np.zeros(3)):
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    
    # 质量、惯量与姿态
    m0        = model.body_subtreemass[base_bid]
    com_world = data.subtree_com[base_bid].copy()
    # J0_world  = data.cinert[base_bid].reshape(3, 3)   # 惯量(about COM) in world
    R_world_body = get_R_from_xmat(data.xmat[base_bid])        # body→world
    Rd = Rz(rotation[2])
    
    # 惯量转到 body：J_body = R^T J_world R
    # J0_body = R_world_body.T @ J0_world @ R_world_body
    
    # 速度
    omega_body = R_world_body.T @ data.cvel[base_bid][0:3].copy()              # ω_body
    v_body     = R_world_body.T @ data.cvel[base_bid][3:6].copy()              # v_body (COM linear vel

    # 重力
    g_body = R_world_body.T @ g
    
    # 计算期望力
    ev = vd_body - v_body 
    global eta_v
    eta_v = np.clip(eta_v + ev * model.opt.timestep, -2.0, 2.0)
    
    Fd_body = m0 * (Kv @ ev + Ki_v @ eta_v) - m0 * g_body
    Fd_world = R_world_body @ Fd_body
    
    # 计算期望力矩
    eR_des = R_world_body.T @ Rd @ vee(Rd.T @ R_world_body)   # 在 “期望机体系 desired-body” 下的旋转误差
    omega_d_in_body = R_world_body.T @ Rd @ omega_d_desired
    
    omega_d_in_body = R_world_body.T @ Rd @ omega_d_desired
    eomega_body = omega_body - omega_d_in_body
    global eta_tau
    eta_tau = np.clip(eta_tau + eR_des * model.opt.timestep, -10.0, 10.0)
    
    # 前馈项：把 \dot{ω}_d 从期望系转到 body
    omegadot_d_in_body = R_world_body.T @ Rd @ omegadot_d_desired
    coriolis_like = hat(omega_body) @ (R_world_body.T @ Rd @ omega_d_desired)  # hat(ω) R^T R_d ω_d
    ff_term = - J0 @ (coriolis_like - omegadot_d_in_body)

    tau_d = (- KR @ eR_des - Kw @ eomega_body + np.cross(omega_body, J0 @ omega_body)
             + ff_term + KI_tau @ eta_tau)

    B_list = []  # 喷口轴
    r_list = []  # 喷口力矩臂
    a_list = []  # 喷口转动范围
    z_body_body = R_world_body.T @ data.xmat[base_bid][6:9]
    for leg in LEG:
        sid = site_id[leg]
        # site 世界姿态/位置
        z_site_world = data.xmat[sid][6:9]            # site 的 +z 轴（世界）
        pos_site_world = data.xpos[sid]
        # 转到机体系
        bB = R_world_body.T @ z_site_world   # 喷口轴（机体系）
        rB = R_world_body.T @ (pos_site_world - data.subtree_com[base_bid])    # 机体质心到 site 的向量
        B_list.append(bB)
        r_list.append(rB)
        a_list.append(z_body_body / (np.linalg.norm(z_body_body) + 1e-9)) # 机体的 +z 轴（世界）
        
    # test
    F_world  = R_world_body @ Fd_body
    tau_world = R_world_body @ tau_d
    
    data.xfrc_applied[base_bid, 0:3] = F_world
    data.xfrc_applied[base_bid, 3:6] = tau_world
    # print(f"F_world: {Fd_body + m0 * g_body}", f"ev: {ev}")
    print(tau_d)
    error_log.append((float(data.time), ev.copy()))

    # 优化计算喷口推力
    u_max_list = [u_max for _ in LEG]
    f_star, eF, eT = solve_thrusters_SOCP(Fd_body, tau_d, r_list, a_list, u_max_list,
                                        f_prev=[f_prev[leg] for leg in LEG],
                                        theta_max=theta_max, rho=rho)
    u_cmd = np.linalg.norm(f_star, axis=1)
    b_body = (f_star.T / (u_cmd + 1e-9)).T  # 4x3 单位方向
    # 反解关节角度
    for i, leg in enumerate(LEG):
        R_world_hip  = get_R_from_xmat(data.xmat[hip_bid[leg]])
        R_body_hip   = R_world_body.T @ R_world_hip
        d = R_body_hip.T @ b_body[i]
        qHAA = np.arctan2(d[1], d[0])
        qHFE = np.arctan2(np.hypot(d[0], d[1]), -d[2])

    # # 关节 PD 控制以及火箭推力控制
    # kp, kd = 40.0, 2.0  # 起步增益，按需要调
    # for leg in LEG:
    #     j1, j2 = hip_jid[leg], thigh_jid[leg]
    #     q1adr = model.jnt_qposadr[j1]; q2adr = model.jnt_qposadr[j2]
    #     v1adr = model.jnt_dofadr[j1]; v2adr = model.jnt_dofadr[j2]
    #     # 期望角 - 当前角
    #     e1 = q1_cmd[leg] - data.qpos[q1adr]
    #     e2 = q2_cmd[leg] - data.qpos[q2adr]
    #     tau1 = kp*e1 - kd*data.qvel[v1adr]
    #     tau2 = kp*e2 - kd*data.qvel[v2adr]
    #     # 写到电机：你的 joint 执行器是 torque motor（class "abduction"/"hip"）
    #     act1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_hip")
    #     act2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_thigh")
    #     data.ctrl[act1] = tau1
    #     data.ctrl[act2] = tau2
    #     # 喷口推力
    #     data.ctrl[thr_act_id[leg]] = u_cmd[leg]
    
    
def save_virtual_control_error_plot(path="virtual_control_error.png"):
    if not error_log:
        print("No virtual control error data recorded; skipping plot.")
        return
    times = np.array([entry[0] for entry in error_log])
    errors = np.stack([entry[1] for entry in error_log])

    plt.figure()
    plt.plot(times, errors[:, 0], label="vx error")
    plt.plot(times, errors[:, 1], label="vy error")
    plt.plot(times, errors[:, 2], label="vz error")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity error [m/s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved virtual control error plot to {path}")


def stand_then_fly(model_path=os.path.join("unitree_go2", "scene.xml")):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    
    
    target_angles = {
        "hip": 0.0,
        "thigh": 0.9,
        "calf": -1.57,
    }
    
    control_target = np.zeros(6, dtype=float)
    
    setup_ids(model)
    setup_gravity(model)
    keyboard_controller.start_listener()
       
    mujoco.mj_resetData(model, data)
    for leg in LEG:
        data.qpos[model.jnt_qposadr[hip_jid[leg]]] = target_angles["hip"]
        data.qpos[model.jnt_qposadr[thigh_jid[leg]]] = target_angles["thigh"]
        data.qpos[model.jnt_qposadr[calf_jid[leg]]] = target_angles["calf"]
    mujoco.mj_forward(model, data)
    cin = data.cinert[base_bid]
    J0 = np.array([[cin[0], cin[3], cin[4]],
                 [cin[3], cin[1], cin[5]],
                 [cin[4], cin[5], cin[2]]])
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_log = 0.0
        while viewer.is_running():
            # # Leg PD control
            # for leg in LEG:
            #     hip_j = hip_jid[leg]
            #     thigh_j = thigh_jid[leg]
            #     calf_j = calf_jid[leg]
                
            #     a_id = actuator_for_joint(model, hip_j)
            #     data.ctrl[a_id] = pd_for_joint(model, data, hip_j, target_angles["hip"])
            #     a_id = actuator_for_joint(model, thigh_j)
            #     data.ctrl[a_id] = pd_for_joint(model, data, thigh_j, target_angles["thigh"])
            #     a_id = actuator_for_joint(model, calf_j)
            #     data.ctrl[a_id] = pd_for_joint(model, data, calf_j, target_angles["calf"])
    
            # Keyboard control
            control_target = keyboard_controller.control_target.copy()
            # Flight control
            control(model, data, control_target[:3], control_target[3:6], J0)
            # print(control_target[:3])
            mujoco.mj_step(model, data)
            viewer.sync()
    save_virtual_control_error_plot()

if __name__ == "__main__":
    stand_then_fly()
