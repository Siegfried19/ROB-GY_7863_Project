import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
from go2_env import Go2EnvMoonFly

# 定义参数随机范围
PARAM_RANGES = {
    "foot_slide": (0.6, 1.0),      # 摩擦力范围
    "valley_width": (1.5, 4.0),    # 沟宽范围
}

def generate_random_configs(num_envs=5):
    """生成 N 组随机配置"""
    configs = []
    for i in range(num_envs):
        f_slide = np.random.uniform(*PARAM_RANGES["foot_slide"])
        foot_fric_vector = [f_slide, 0.005, 0.005]
        v_width = np.random.uniform(*PARAM_RANGES["valley_width"])
        
        config = {
            "id": i,
            "foot_friction": foot_fric_vector,
            "body_friction": 0.8,
            "valley_width": v_width,
        }
        configs.append(config)
    return configs

def verify_and_render(config):
    """手动执行 Reset 过程并渲染，不使用 RL 策略"""
    print(f"\n{'-'*60}")
    print(f"正在加载环境 ID [{config['id']}] ...")
    
    # 1. 初始化环境
    try:
        env = Go2EnvMoonFly(
            xml_path="../unitree_go2/scene_moon_valley_jet.xml", 
            foot_friction=config["foot_friction"],
            body_friction=config["body_friction"],
            valley_width=config["valley_width"],
            rank=config["id"]
        )
        # 注意：这里我们只通过 mujoco 原生方法重置数据，不调用 env.reset()
        # 因为 env.reset() 会在后台瞬间跑完 1000 步，无法可视化
        mujoco.mj_resetData(env.model, env.data)
        
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        return

    # 2. 手动设置初始姿态 (从 env.reset 中提取的逻辑)
    env.data.qpos[0] = 1.5   # x
    env.data.qpos[1] = 0.0   # y
    env.data.qpos[2] = 3.4   # z (高空)
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] # quaternion
    
    # 设置关节初始角度
    init_joint_angles = np.array([0, 0.9, -1.57] * 4, dtype=np.float32)
    env.data.qpos[7:19] = init_joint_angles
    
    # 3. 物理/几何验证打印
    print(f"  [场景参数] 沟宽: {config['valley_width']:.2f}m | 摩擦: {config['foot_friction'][0]:.2f}")
    
    # 4. 启动渲染循环
    print(f"\n📺 正在启动 MuJoCo 查看器...")
    print(f"   此时正在运行 PD 控制器 (RL 尚未接管)，观察落地稳定性...")
    print(f"   (按 ESC 关闭窗口)")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # 定义 PD 参数 (与 reset 中保持一致)
        kp = 60.0
        kd = 3.0
        
        while viewer.is_running():
            # ------------------------------------------------
            # 这里复现了 reset() 中的 settle loop
            # ------------------------------------------------
            
            # 1. 计算 PD 力矩
            current_angles = env.data.qpos[7:19]
            current_vel    = env.data.qvel[6:18]
            
            torque = kp * (init_joint_angles - current_angles) - kd * current_vel
            
            # 2. 限制力矩
            max_torque = env.model.actuator_ctrlrange[:12, 1]
            torque = np.clip(torque, -max_torque, max_torque)
            
            # 3. 写入控制
            env.data.ctrl[:12] = torque
            
            # [关键] 显式关闭火箭推力
            # 原来的 env.step(0) 会导致 (0+1)/2 = 50% 推力，这里我们手动设为 0
            env.data.ctrl[12:16] = 0.0 
            
            # 4. 物理步进
            mujoco.mj_step(env.model, env.data)
            
            # 5. 渲染同步
            viewer.sync()
            
            # 稍微加点延迟，让肉眼能看清动作 (模拟实时)
            time.sleep(env.model.opt.timestep)

    env.close()
    print("窗口已关闭。")

def main():
    num_envs = 6
    configs = generate_random_configs(num_envs)

    while True:
        print(f"\n{'='*20} 仅验证 Reset/PD 阶段 {'='*20}")
        for cfg in configs:
            print(f"ID {cfg['id']} | 宽度 {cfg['valley_width']:.2f} | 摩擦 {cfg['foot_friction'][0]:.2f}")
        
        choice = input("\n输入 ID 查看下落过程 (q 退出): ").strip()
        if choice == 'q': break
        
        if choice.isdigit() and int(choice) < num_envs:
            verify_and_render(configs[int(choice)])
        else:
            print("输入无效。")

if __name__ == "__main__":
    main()