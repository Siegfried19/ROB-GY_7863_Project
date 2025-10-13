import numpy as np
import mujoco as mj
import cvxpy as cp

Kv = np.diag([3.0, 3.0, 4.0])
Ki_v = np.diag([0.3, 0.3, 0.4])
KR = np.diag([4.0, 4.0, 4.0])
Kw = np.diag([0.6, 0.6, 0.8])
KI_tau = np.diag([0.3, 0.3, 0.3])
rho = 0.1                       # 平滑
theta_max = np.deg2rad(25.0)    # 喷口最大等效偏转
u_max = 200.0                   # 见 xml 的 ctrlrange 上限
# g = np.array([0, 0, -9.81])
g = None

LEG = ["FL","FR","RL","RR"]
site_id = {leg: None for leg in LEG}
hip_bid = {leg: None for leg in LEG}
hip_jid = {leg: None for leg in LEG}      # HAA
thigh_jid = {leg: None for leg in LEG}    # HFE
thr_act_id = {leg: None for leg in LEG}

f_prev = {leg: np.zeros(3) for leg in LEG}  # 上次喷口力向量，模是力
eta_v = np.zeros(3) # 积分误差速度
eta_tau = np.zeros(3)   # 积分误差角速度

def get_R_from_xmat(xmat9):
    R = np.array(xmat9, dtype=float).reshape(3,3)
    return R

def vee(Rerr):
    # vee( 0.5 (R_d^T R - R^T R_d) )
    return np.array([Rerr[2,1]-Rerr[1,2], Rerr[0,2]-Rerr[2,0], Rerr[1,0]-Rerr[0,1]])*0.5

def setup_ids_gravity(model):
    global g
    g = model.opt.gravity
    for leg in LEG:
        site_id[leg] = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, f"{leg}_rocket_site")
        hip_bid[leg] = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{leg}_hip")
        hip_jid[leg] = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{leg}_hip_joint")
        thigh_jid[leg]= mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{leg}_thigh_joint")
        thr_act_id[leg] = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_rocket_thruster")

def control(model, data, vd_body, Rd):
    # 机体 R, p, v, ω
    base_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base")
    m0 = data.subtree_mass[base_bid]
    com_world = data.subtree_com[base_bid].copy()
    J0_world = data.subtree_inertia[base_bid].reshape(3, 3)
    
    R_world_body = get_R_from_xmat(data.xmat[base_bid]) # 机体姿态（世界）
    
    p_body_world = data.xpos[base_bid].copy()   # 机体质心位置（世界）
    v_body = data.cvel[base_bid][3:6].copy()  # or data.qvel[0:3] if free joint layout
    omega_body = data.cvel[base_bid][0:3].copy()

    # 外环
    ev = vd_body - v_body
    global eta_v
    eta_v = np.clip(eta_v + ev*data.time_step, -2.0, 2.0)
    
    Fd_world = m0*(Kv@ev + Ki_v@eta_v) + m0*g
    Fd_body = R_world_body.T @ Fd_world

    Rerr = Rd.T @ R_world_body
    tau_d = (J0 @ (Kw @ (np.zeros(3) - omega_body))
            + KR @ vee(Rerr) + KI_tau @ eta_tau)

    # 组 r_i^B, a_i (当前喷口轴)：
    B_list, r_list, a_list = [], [], []
    for leg in LEG:
        sid = site_id[leg]
        # site 世界姿态/位置
        z_world = data.xmat[sid][6:9]            # site 的 +z 轴（世界）
        x_world = data.xpos[sid]
        # 转到机体系
        bB = R_world_body.T @ z_world   # 喷口轴（机体系）
        rB = R_world_body.T @ (x_world - data.subtree_com[base_bid])    # 机体质心到 site 的向量
        B_list.append(bB)
        r_list.append(rB)
        a_list.append(bB / (np.linalg.norm(bB)+1e-9))   # 取当前轴做锥轴

    B = np.stack(B_list, axis=1)         # 3x4
    RxB = np.stack([np.cross(r_list[i], B_list[i]) for i in range(4)], axis=1)  # 3x4
    M = np.vstack([B, RxB])              # 6x4（若做 f 向量法则见下）

    # ---- (A) 力向量 SOCP/QP（简化版：用 QP 近似 + 圆锥内逼近） ----
    # 这里给一个“近似QP”：先不显式写锥约束，而是在目标里惩罚偏离锥轴和朝向错误；实测很稳
    # 变量改为 12 维 f = [f_FL; f_FR; f_RL; f_RR]
    # 下面演示的是一步闭式带正则的最小二乘（可替换成 OSQP 带盒约束/非负等）
    Wd = np.hstack([Fd_body, tau_d])     # 6
    # 先用当前方向解推力标量 u（非负 + 限幅）：
    Mtikh = M.T @ M + 1e-3*np.eye(4)
    u = np.linalg.solve(Mtikh, M.T @ Wd)
    u = np.clip(u, 0.0, u_max)

    # **可选：若你要“最小化总推力”**，把变量换成 f_i 向量并用 CVXPY 实现严格 SOCP（见上文公式）

    # ---- 反解两髋角（把 f_i → b_i → d_i → 两次 atan2） ----
    q1_cmd = {}; q2_cmd = {}; u_cmd = {}
    for k,leg in enumerate(LEG):
        if u[k] < 1e-6:
            # 维持当前角度
            j1, j2 = hip_jid[leg], thigh_jid[leg]
            q1_cmd[leg] = data.qpos[model.jnt_qposadr[j1]]
            q2_cmd[leg] = data.qpos[model.jnt_qposadr[j2]]
            u_cmd[leg]  = 0.0
            continue
        b_body = B[:,k] / (np.linalg.norm(B[:,k])+1e-9)    # 这里用当前轴方向即可
        # 旋回到髋母体：
        R_world_hip = get_R_from_xmat(data.xmat[hip_bid[leg]])
        R_body_hip = R_world_body.T @ R_world_hip
        d = R_body_hip.T @ b_body
        q1 = np.arctan2(d[1], d[0])                         # HAA
        q2 = np.arctan2(np.hypot(d[0],d[1]), -d[2])         # HFE
        q1_cmd[leg] = np.clip(q1, -1.0472, 1.0472)          # 取自 xml abduction range
        # 前/后腿 HFE range 不同，可分别夹
        if leg[0]=='F': q2_cmd[leg] = np.clip(q2, -1.5708, 3.4907)
        else:           q2_cmd[leg] = np.clip(q2, -0.5236, 4.5379)
        u_cmd[leg] = float(u[k])

    # ---- 输出：推力与关节力矩（PD 伺服到角度） ----
    kp, kd = 40.0, 2.0  # 起步增益，按需要调
    for leg in LEG:
        j1, j2 = hip_jid[leg], thigh_jid[leg]
        q1adr = model.jnt_qposadr[j1]; q2adr = model.jnt_qposadr[j2]
        v1adr = model.jnt_dofadr[j1]; v2adr = model.jnt_dofadr[j2]
        # 期望角 - 当前角
        e1 = q1_cmd[leg] - data.qpos[q1adr]
        e2 = q2_cmd[leg] - data.qpos[q2adr]
        tau1 = kp*e1 - kd*data.qvel[v1adr]
        tau2 = kp*e2 - kd*data.qvel[v2adr]
        # 写到电机：你的 joint 执行器是 torque motor（class "abduction"/"hip"）
        act1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_hip")
        act2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_thigh")
        data.ctrl[act1] = tau1
        data.ctrl[act2] = tau2
        # 喷口推力
        data.ctrl[thr_act_id[leg]] = u_cmd[leg]
        

def solve_thrusters_SOCP(Fd_body, tau_d, r_list, a_list, u_max_list,
                         f_prev=None, theta_max=np.deg2rad(25), rho=0.1,
                         lamF=1000.0, lamT=1000.0):
    """
    同步优化 4 个喷口的“方向+大小”。输入/输出都在机体系。
    r_list: 4x3 机体系力矩臂
    a_list: 4x3 机体系锥轴（由当前 thigh 朝向的 -z 得到）
    """
    n = 4
    f = [cp.Variable(3) for _ in range(n)]
    epsF = cp.Variable(3)   # 合力松弛
    epsT = cp.Variable(3)   # 力矩松弛

    cons = []
    cons += [cp.sum(f) == Fd_body + epsF]
    # 力矩平衡：∑ r×f = τ + epsT
    torque_cols = [cp.hstack([ r_list[i][1]*f[i][2] - r_list[i][2]*f[i][1],
                               r_list[i][2]*f[i][0] - r_list[i][0]*f[i][2],
                               r_list[i][0]*f[i][1] - r_list[i][1]*f[i][0] ]) for i in range(n)]
    cons += [cp.sum(torque_cols, axis=0) == tau_d + epsT]

    # 圆锥可达 + 幅值上限
    tmax = np.tan(theta_max)
    for i in range(n):
        ai = a_list[i] / (np.linalg.norm(a_list[i]) + 1e-9)
        # 平行分量与垂直分量
        f_par = cp.sum(cp.multiply(ai, f[i]))            # = f_i^T a_i
        f_perp = f[i] - f_par * ai
        cons += [cp.norm(f_perp, 2) <= tmax * (-f_par)]
        cons += [f_par <= 0]                              # 朝“向下”
        cons += [cp.norm(f[i], 2) <= u_max_list[i]]

    # 目标：总推力 + 平滑 + 松弛罚
    obj_terms = [cp.norm(fi, 2) for fi in f]
    if f_prev is not None:
        for i in range(n):
            obj_terms.append(rho * cp.norm(f[i] - f_prev[i], 2))
    obj_terms += [lamF * cp.norm(epsF, 2), lamT * cp.norm(epsT, 2)]
    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), cons)
    prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)

    f_val = [fi.value if fi.value is not None else np.zeros(3) for fi in f]
    return np.array(f_val), epsF.value, epsT.value

