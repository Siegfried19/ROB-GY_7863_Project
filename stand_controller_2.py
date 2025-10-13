import os
import mujoco
import mujoco.viewer
import numpy as np

g = None

LEG = ["FL","FR","RL","RR"]
site_id     = {leg: None for leg in LEG}
hip_bid     = {leg: None for leg in LEG}
hip_jid     = {leg: None for leg in LEG}    # HAA
thigh_jid   = {leg: None for leg in LEG}    # HFE
calf_jid    = {leg: None for leg in LEG}    # KFE
thr_act_id  = {leg: None for leg in LEG}

def setup_gravity(model):
    global g
    g = model.opt.gravity
    print(f"Gravity: {g}")

def setup_ids(model):
    for leg in LEG:
        site_id[leg]    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_rocket_site")
        hip_bid[leg]    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip")
        hip_jid[leg]    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_hip_joint")
        thigh_jid[leg]  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_thigh_joint")
        calf_jid[leg]   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_calf_joint")
        thr_act_id[leg] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_rocket_thruster")

def _actuator_for_joint(model, joint_id):
    """Find actuator index that drives the given joint id. Returns -1 if none."""
    for a in range(model.nu):
        if model.actuator_trnid[a, 0] == joint_id:
            return a
    return -1

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

def stand_still(model_path=os.path.join("unitree_go2", "scene.xml")):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    setup_ids(model)
    setup_gravity(model)
    
    target_angles = {
        "hip": 0.0,
        "thigh": 0.9,
        "calf": -1.57,
    }
    
    # Initialize model state close to target pose
    mujoco.mj_resetData(model, data)
    for leg in LEG:
        data.qpos[model.jnt_qposadr[hip_jid[leg]]] = 0.5
        data.qpos[model.jnt_qposadr[thigh_jid[leg]]] = 1.2
        data.qpos[model.jnt_qposadr[calf_jid[leg]]] = -1.27
    
    mujoco.mj_forward(model, data)
    print("开启仿真窗口：保持站立（按下关闭按钮结束）")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_log = 0.0
        while viewer.is_running():
            for leg in LEG:
                hip_j = hip_jid[leg]
                thigh_j = thigh_jid[leg]
                calf_j = calf_jid[leg]
                
                a_id = _actuator_for_joint(model, hip_j)
                data.ctrl[a_id] = pd_for_joint(model, data, hip_j, target_angles["hip"])
                a_id = _actuator_for_joint(model, thigh_j)
                data.ctrl[a_id] = pd_for_joint(model, data, thigh_j, target_angles["thigh"])
                a_id = _actuator_for_joint(model, calf_j)
                data.ctrl[a_id] = pd_for_joint(model, data, calf_j, target_angles["calf"])
    
            mujoco.mj_step(model, data)
            viewer.sync()
if __name__ == "__main__":
    stand_still()