import numpy as np

freq = 1.5 

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

def get_ref_torque(model,data):
    leg_joints = {}
    leg_torques = []
    leg_joints_ref = []
    t =data.time
    for i in range(model.njnt):
        jname = model.joint(i).name
       
        
        # Group by leg
        if jname.startswith('FL_'):
            leg_joints.setdefault('FL', []).append(i)
        elif jname.startswith('FR_'):
            leg_joints.setdefault('FR', []).append(i)
        elif jname.startswith('RL_'):
            leg_joints.setdefault('RL', []).append(i)
        elif jname.startswith('RR_'):
            leg_joints.setdefault('RR', []).append(i)
            
    for leg_name, joints in leg_joints.items(): 
        target_angles = get_joint_angles(t, leg_name)
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
            leg_joints_ref.append(target_angles[i])
            leg_torques.append(action_value)

    return leg_joints_ref, leg_torques