import os

import mujoco
import mujoco.viewer
import numpy as np



def _collect_leg_joints(model):
    """Collect joint ids per leg, ordered by qpos address (hip, thigh, calf)."""
    leg_joints = {}
    for j in range(model.njnt):
        name = model.joint(j).name
        if not name:
            continue
        if name.startswith("FL_"):
            leg_joints.setdefault("FL", []).append(j)
        elif name.startswith("FR_"):
            leg_joints.setdefault("FR", []).append(j)
        elif name.startswith("RL_"):
            leg_joints.setdefault("RL", []).append(j)
        elif name.startswith("RR_"):
            leg_joints.setdefault("RR", []).append(j)

    # Ensure consistent hip/thigh/calf order per leg by qpos index
    for leg in leg_joints:
        leg_joints[leg] = sorted(leg_joints[leg], key=lambda j: model.jnt_qposadr[j])
    return leg_joints


def _actuator_for_joint(model, joint_id):
    """Find actuator index that drives the given joint id. Returns -1 if none."""
    for a in range(model.nu):
        if model.actuator_trnid[a, 0] == joint_id:
            return a
    return -1


def stand_still(model_path=os.path.join("unitree_go2", "scene.xml")):
    """Hold GO2 at a stable standing pose using simple joint-space PD control."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件找不到：{model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Group joints by leg and set target standing angles (hip abduction, thigh, calf)
    leg_joints = _collect_leg_joints(model)

    # Desired static standing pose (radians)
    # Hip abduction ~0, Thigh ~0.9, Calf ~-1.57 matches the 'home' keyframe
    target_angles = {
        "hip": 0.0,
        "thigh": 0.9,
        "calf": -1.57,
    }

    # Initialize model state close to target pose
    mujoco.mj_resetData(model, data)
    for leg, joints in leg_joints.items():
        if len(joints) < 3:
            continue
        hip_j, thigh_j, calf_j = joints[:3]
        data.qpos[model.jnt_qposadr[hip_j]] = target_angles["hip"]
        data.qpos[model.jnt_qposadr[thigh_j]] = target_angles["thigh"]
        data.qpos[model.jnt_qposadr[calf_j]] = target_angles["calf"]
    mujoco.mj_forward(model, data)

    # PD gains
    kp_hip, kd_hip = 60.0, 4.0
    kp_thigh, kd_thigh = 80.0, 6.0
    kp_calf, kd_calf = 60.0, 4.0

    def pd_for_joint(joint_idx, target):
        qpos_idx = model.jnt_qposadr[joint_idx]
        qvel_idx = model.jnt_dofadr[joint_idx]
        # Choose gains based on joint role (0 hip abduction, 1 thigh, 2 calf per leg)
        kp, kd = 70.0, 5.0
        for _, joints in leg_joints.items():
            if joint_idx in joints[:3]:
                idx_in_leg = list(joints[:3]).index(joint_idx)
                if idx_in_leg == 0:
                    kp, kd = kp_hip, kd_hip
                elif idx_in_leg == 1:
                    kp, kd = kp_thigh, kd_thigh
                else:
                    kp, kd = kp_calf, kd_calf
                break
        err = target - data.qpos[qpos_idx]
        derr = -data.qvel[qvel_idx]
        return kp * err + kd * derr

    # Identify thruster actuators (general actuators at the end) and keep them off
    thruster_names = (
        "FL_rocket_thruster",
        "FR_rocket_thruster",
        "RL_rocket_thruster",
        "RR_rocket_thruster",
    )
    thruster_ids = []
    for nm in thruster_names:
        try:
            thruster_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm))
        except mujoco.Error:
            pass

    print("开启仿真窗口：保持站立（按下关闭按钮结束）")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_log = 0.0
        while viewer.is_running():
            # Compute PD torques for each joint towards the standing pose
            for leg, joints in leg_joints.items():
                if len(joints) < 3:
                    continue
                hip_j, thigh_j, calf_j = joints[:3]

                # Resolve actuators and write control values
                # Hip abduction
                a_id = _actuator_for_joint(model, hip_j)
                if a_id != -1:
                    data.ctrl[a_id] = pd_for_joint(hip_j, target_angles["hip"])

                # Thigh
                a_id = _actuator_for_joint(model, thigh_j)
                if a_id != -1:
                    data.ctrl[a_id] = pd_for_joint(thigh_j, target_angles["thigh"])

                # Calf
                a_id = _actuator_for_joint(model, calf_j)
                if a_id != -1:
                    data.ctrl[a_id] = pd_for_joint(calf_j, target_angles["calf"])

            # Ensure rockets off
            for a_id in thruster_ids:
                data.ctrl[a_id] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()

            # Periodic log
            if data.time - last_log > 1.0:
                base_pos = data.qpos[:3]
                height = float(base_pos[2])
                dist = float(np.linalg.norm(base_pos[:2]))
                print(f"t={data.time:5.2f}s  高度={height:.3f}m  偏移={dist:.3f}m")
                last_log = float(data.time)



if __name__ == "__main__":
    stand_still()
