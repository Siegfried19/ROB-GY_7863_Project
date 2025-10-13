import os
import mujoco
import mujoco.viewer
import numpy as np
import cvxpy
import time
import keyboard_controller

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
Kv = np.diag([3.0, 3.0, 4.0])
Ki_v = np.diag([0.3, 0.3, 0.4])
KR = np.diag([4.0, 4.0, 4.0])
Kw = np.diag([0.6, 0.6, 0.8])
KI_tau = np.diag([0.3, 0.3, 0.3])

# Thruster optimization parameters
rho = 0.1                       # 平滑
theta_max = np.deg2rad(25.0)    # 喷口最大等效偏转
u_max = 200.0                   # 见 xml 的 ctrlrange 上限

g = None

# Controller parameters
f_prev = {leg: np.zeros(3) for leg in LEG}  # 上次喷口力向量，模是力
eta_v = np.zeros(3) # 积分误差速度
eta_tau = np.zeros(3)   # 积分误差角速度

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
'''
def control(model, data, vd_body, Rd, omega_d_desired=np.zeros(3), omegadot_d_desired=np.zeros(3)):
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    
    # 质量、惯量与姿态
    m0        = data.subtree_mass[base_bid]
    com_world = data.subtree_com[base_bid].copy()
    J0_world  = data.subtree_inertia[base_bid].reshape(3, 3)   # 惯量(about COM) in world
    R_world_body = get_R_from_xmat(data.xmat[base_bid])        # body→world
    
    # 惯量转到 body：J_body = R^T J_world R
    J0_body = R_world_body.T @ J0_world @ R_world_body
    
    # 速度
    omega_body = data.cvel[base_bid][0:3].copy()              # ω_body
    v_body     = data.cvel[base_bid][3:6].copy()              # v_body (COM linear vel
    
    # 重力
    g_body = R_world_body.T @ g
    
    # 计算期望力
    ev = vd_body - v_body 
    global eta_v
    eta_v = np.clip(eta_v + ev * data.time_step, -2.0, 2.0)
    
    Fd_body = m0 * (Kv @ ev + Ki_v @ eta_v) + m0 * g_body
    Fd_world = R_world_body @ Fd_body
    
    # 计算期望力矩
    eR_des = vee(Rd.T @ R_world_body)   # 在 “期望机体系 desired-body” 下的旋转误差
    omega_d_in_body = R_world_body.T @ Rd @ omega_d_desired
    
    omega_d_in_body = R_world_body.T @ Rd @ omega_d_desired
    eomega_body = omega_body - omega_d_in_body
    global eta_tau
    eta_tau = np.clip(eta_tau + eR_des * data.time_step, -1.0, 1.0)
    
    # 前馈项：把 \dot{ω}_d 从期望系转到 body
    omegadot_d_in_body = R_world_body.T @ Rd @ omegadot_d_desired
    coriolis_like = hat(omega_body) @ (R_world_body.T @ Rd @ omega_d_desired)  # hat(ω) R^T R_d ω_d
    ff_term = - J0_body @ (coriolis_like - omegadot_d_in_body)

    tau_d = (- KR @ eR_des - Kw @ eomega_body + np.cross(omega_body, J0_body @ omega_body) 
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
        
    

def stand_then_fly(model_path=os.path.join("unitree_go2", "scene.xml")):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    setup_ids(model)
    setup_gravity(model)
    
    target_angles = {
        "hip": 0.0,
        "thigh": 0.9,
        "calf": -1.57,
    }
    
    mujoco.mj_resetData(model, data)
    for leg in LEG:
        data.qpos[model.jnt_qposadr[hip_jid[leg]]] = target_angles["hip"]
        data.qpos[model.jnt_qposadr[thigh_jid[leg]]] = target_angles["thigh"]
        data.qpos[model.jnt_qposadr[calf_jid[leg]]] = target_angles["calf"]
    
    mujoco.mj_forward(model, data)

