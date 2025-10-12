import mujoco
import mujoco.viewer
import numpy as np
import os
import csv
from datetime import datetime
import keyboard_controller
from typing import List, Sequence, Tuple
import time
from scipy.spatial.transform import Rotation as R
THRUSTER_NAMES: Tuple[str, ...] = (
    "FL_rocket_thruster",
    "FR_rocket_thruster",
    "RL_rocket_thruster",
    "RR_rocket_thruster",
)

def simple_walk_demo(model_path):
    """Simple hardcoded walking demonstration for Unitree Go2"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Print available joints
    print("\nAvailable joints:")
    leg_joints = {}
    for i in range(model.njnt):
        jname = model.joint(i).name
        print(f"  [{i}] {jname}")
        
        # Group by leg
        if jname.startswith('FL_'):
            leg_joints.setdefault('FL', []).append(i)
        elif jname.startswith('FR_'):
            leg_joints.setdefault('FR', []).append(i)
        elif jname.startswith('RL_'):
            leg_joints.setdefault('RL', []).append(i)
        elif jname.startswith('RR_'):
            leg_joints.setdefault('RR', []).append(i)
            
    thruster_ids: List[int] = []
    for name in THRUSTER_NAMES:
        try:
            thruster_ids.append(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            )
        except mujoco.Error:
            raise ValueError(f"模型里找不到火箭执行器：{name}")
    
    # Sort joints by position
    for leg in leg_joints:
        leg_joints[leg] = sorted(leg_joints[leg], key=lambda j: model.jnt_qposadr[j])
    
    print(f"\nLeg joints grouped: {leg_joints}")
    
    # Reset to standing pose
    mujoco.mj_resetData(model, data)
    
    # Set initial standing configuration
    for leg, joints in leg_joints.items():
        if len(joints) >= 3:
            hip_idx = model.jnt_qposadr[joints[0]]
            thigh_idx = model.jnt_qposadr[joints[1]]
            calf_idx = model.jnt_qposadr[joints[2]]
            
            data.qpos[hip_idx] = 0.0      # Hip neutral
            data.qpos[thigh_idx] = 0.8    # Thigh forward
            data.qpos[calf_idx] = -1.5    # Calf bent
    
    mujoco.mj_forward(model, data)
 

    
    # Walking parameters
    freq = 1.5  # Walking frequency (Hz)
    max_simulation_time = 100.0  # Stop after 10 seconds
    
    # Trot gait - BACK TO ORIGINAL
    def get_phase_offset(leg_name):
        if leg_name in ['FL', 'RR']:
            return np.pi
        else:  # FR, RL
            return 0.0
    
    # Simple joint trajectories for trotting
    def get_joint_angles(t, leg_name):
        phase = 2 * np.pi * freq * t + get_phase_offset(leg_name)
        
        # Swing phase when sin(phase) > 0
        swing = np.sin(phase)
        
        # Hip: NO hip motion - this prevents rotation
        hip = 0.01 * np.sin(phase)
        
        # Thigh: NEGATE to walk forward instead of backward
        # Original was: 0.8 + 0.4 * swing (walked backward)
        # So reverse it to: use negative swing values
        thigh = 0.8 + 0.6 * swing  # Flip the direction
        
        # Calf: Keep original
        calf = -1.5 + 0.1 * np.sin(phase + np.pi / 2)

        return [hip, thigh, calf]
    


    print("Starting walk demo...")
    print("Simulation will stop after 10 seconds\n")
    dog_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "go2")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print = 0
        # viewer.cam.lookat[:] = data.body(dog_id).xpos
        viewer.cam.distance = 10
        keyboard_controller.start_listener()
        while viewer.is_running() and data.time < max_simulation_time:

            # # get world coordinate
            # dog_pos = data.xpos[dog_id]

            # # camera setting
            # viewer.cam.lookat[:] = dog_pos             
            # viewer.cam.distance = 3.0                
            # viewer.cam.elevation = -10         
            # viewer.cam.azimuth = 180                   

            t = data.time
            # ==== PREPARE DATA ROW ====
            data_row = [t]
            
            # Position (x, y, z)
            data_row.extend(data.qpos[:3].tolist())
            
            # Orientation (quaternion: w, x, y, z)
            data_row.extend(data.qpos[3:7].tolist())
            
            # Linear velocity (x, y, z)
            data_row.extend(data.qvel[:3].tolist())
            
            # Angular velocity (x, y, z)
            data_row.extend(data.qvel[3:6].tolist())
            
            # Store actions and targets
            actions = []
            targets = []
            # ==========================
            

            qw, qx, qy, qz = data.qpos[3:7]
            roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)

            # 2) 简单 PD 姿态反馈（也可只用 P）
            # 角速度在 qvel[3:6]，其中 wz = qvel[5]
            wx, wy, wz = data.qvel[3:6]
            k_p_pitch, k_d_pitch = 0.5, 0.05   # 先小一点，防抖
            k_p_roll,  k_d_roll  = 0.5, 0.05
        
            pitch_correction = np.clip(-(k_p_pitch*pitch + k_d_pitch*wy), -1.0, 1.0)
            roll_correction  = np.clip(-(k_p_roll*roll  + k_d_roll*wx),  -1.0, 1.0)

            print(pitch_correction,roll_correction)
            # Compute target angles for each leg
            for leg_name, joints in leg_joints.items():
                if len(joints) < 3:
                    continue
                
                target_angles = get_joint_angles(t, leg_name)
                        
                if leg_name in ["FL", "RL"]:
                    target_angles[0] += roll_correction      # 左侧外展(+)
                else:  # FR, RR
                    target_angles[0] -= roll_correction      # 右侧内收(-)

                # pitch：前腿加(+)，后腿减(-) —— 以抵消前俯
                if leg_name in ["FL", "FR"]:
                    target_angles[1] += pitch_correction
                else:  # RL, RR
                    target_angles[1] -= pitch_correction
                    
                # Store target angles
                targets.extend(target_angles)
                
                # Apply PD control to each joint
                leg_actions = []
                for i, joint_id in enumerate(joints[:3]):
                    # Find actuator for this joint
                    action_value = 0.0
                    for act_id in range(model.nu):
                        if model.actuator_trnid[act_id, 0] == joint_id:
                            qpos_idx = model.jnt_qposadr[joint_id]
                            qvel_idx = model.jnt_dofadr[joint_id]
                            
                            # PD control
                            kp = 100.0  # Position gain
                            kd = 10.0   # Velocity gain
                            
                            error = target_angles[i] - data.qpos[qpos_idx]
                            error_dot = -data.qvel[qvel_idx]
                            
                            action_value = kp * error + kd * error_dot
                            data.ctrl[act_id] = action_value
                            break
                    
                    leg_actions.append(action_value)
                
                actions.extend(leg_actions)
            
            # Add actions and targets to data row
            data_row.extend(actions)
            data_row.extend(targets)
            
     
            # Step simulation
            if keyboard_controller.is_active():
                for act_id in thruster_ids:
                    if act_id == 14 or act_id == 15:
                        print(f"火箭{act_id:.2f}喷射")
                        data.ctrl[act_id] = 23
                    if act_id == 12 or act_id == 13:
                        print(f"火箭{act_id:.2f}喷射")
                        data.ctrl[act_id] = 23*1.32
            else:
                data.ctrl[12:] = 0.0
            mujoco.mj_step(model, data)
            time.sleep(0.001)
            viewer.sync()
            
            # Print status every 2 seconds
            if t - last_print > 2.0:
                pos = data.qpos[:3]
                height = pos[2]
                dist = np.sqrt(pos[0]**2 + pos[1]**2)
                print(f"t={t:5.1f}s | pos=[{pos[0]:6.2f}, {pos[1]:6.2f}] | height={height:.3f}m | dist={dist:.2f}m")
                last_print = t


if __name__ == "__main__":
    # Use the path from your code

    model_path = f"unitree_go2/scene_moon.xml"
    
    simple_walk_demo(model_path)