import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
from scipy.spatial.transform import Rotation as R
from go2_env import Go2EnvMoonFly

# --- 1. 参数配置区域 (保留你的随机生成逻辑) ---
PARAM_RANGES = {
    "foot_slide": (0.6, 1.0),      # 脚底摩擦力范围
    "valley_width": (1.5, 4.0),    # 沟宽范围
}

def generate_random_configs(num_envs=6):
    """生成 N 组随机配置供选择"""
    configs = []
    for i in range(num_envs):
        f_slide = np.random.uniform(*PARAM_RANGES["foot_slide"])
        # 这里把摩擦力拼成 3维向量 [滑动, 滚动, 扭转]
        foot_fric_vector = [f_slide, 0.02, 0.01] 
        v_width = np.random.uniform(*PARAM_RANGES["valley_width"])
        
        config = {
            "id": i,
            "foot_friction": foot_fric_vector,
            "body_friction": 0.4, # 假设身体摩擦力固定，也可随机
            "valley_width": v_width,
        }
        configs.append(config)
    return configs

def get_body_state(env):
    """辅助函数：获取当前的本体状态信息"""
    # 线速度
    lin_vel = env.data.qvel[:3]
    lin_speed = np.linalg.norm(lin_vel)
    # 角速度
    ang_vel = env.data.qvel[3:6]
    ang_speed = np.linalg.norm(ang_vel)
    # 姿态
    qw, qx, qy, qz = env.data.qpos[3:7]
    roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=True)
    # 高度
    height = env.data.qpos[2]
    
    return lin_speed, ang_speed, roll, pitch, height

def verify_single_env(config):
    """
    针对选定的配置，启动 Viewer 并循环测试 reset 稳定性
    """
    print(f"\n{'-'*60}")
    print(f"🚀 正在加载环境 ID [{config['id']}] ...")
    print(f"   参数: 沟宽={config['valley_width']:.2f}m | 脚摩擦={config['foot_friction'][0]:.2f}")
    
    # 1. 初始化环境 (直接调用你的类)
    try:
        env = Go2EnvMoonFly(
            xml_path="../unitree_go2/scene_moon_valley_jet.xml", 
            foot_friction=config["foot_friction"],
            body_friction=config["body_friction"],
            valley_width=config["valley_width"],
            rank=config['id']
        )
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return

    print(f"✅ 环境创建成功，启动 Viewer...")

    # 2. 启动 Viewer 循环
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # 开启接触点可视化，方便看脚有没有穿模
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        loop_cnt = 0
        while viewer.is_running():
            loop_cnt += 1
            print(f"\n>>> [第 {loop_cnt} 次测试] 正在执行 env.reset() (含500步settle)...")
            
            # --- 核心测试点：调用 reset ---
            start_t = time.time()
            env.reset() 
            cost_t = time.time() - start_t
            
            viewer.sync() # 同步画面

            # --- 诊断：获取 Reset 刚结束时的状态 ---
            lin_v, ang_v, roll, pitch, h = get_body_state(env)
            
            print(f"  [Reset 完成] 耗时 {cost_t:.2f}s")
            print(f"  ------------------------------------------------")
            print(f"  高度 (Z)       : {h:.4f} m")
            print(f"  姿态 (R/P)     : Roll={roll:.2f}°, Pitch={pitch:.2f}°")
            print(f"  残余线速度     : {lin_v:.6f} m/s (应接近0)")
            print(f"  残余角速度     : {ang_v:.6f} rad/s")

            # 简单的自动判定
            if lin_v < 0.05 and abs(roll) < 5.0 and abs(pitch) < 5.0:
                print("  ✅ [判定] 初始化状态：稳定")
            else:
                print("  ⚠️ [判定] 初始化状态：不稳定 (速度过大或姿态倾斜)")

            # --- 观察期：运行 100 步看是否会炸 ---
            print("  [观察期] 自由运行 100 步 (Action=0)...")
            action = np.zeros(env.action_space.shape)
            action[12:] = -1.0  # 关闭喷气
            
            crashed = False
            for _ in range(100):
                if not viewer.is_running(): break
                env.step(action)
                viewer.sync()
                time.sleep(0.01) # 慢放
                
                # 实时检查是否炸飞
                if np.linalg.norm(env.data.qvel[:3]) > 2.0:
                    crashed = True
                    break
            
            if crashed:
                print("  ❌ [警告] 机器人在观察期飞出/倒下！物理接触可能存在冲突。")
                # lin_v, ang_v, roll, pitch, h = get_body_state(env)
            
                # print(f"  [Reset 完成后继续运行] 状态如下：")
                # print(f"  ------------------------------------------------")
                # print(f"  高度 (Z)       : {h:.4f} m")
                # print(f"  姿态 (R/P)     : Roll={roll:.2f}°, Pitch={pitch:.2f}°")
                # print(f"  残余线速度     : {lin_v:.6f} m/s (应接近0)")
                # print(f"  残余角速度     : {ang_v:.6f} rad/s")
            
            print(f"  --- 等待 2 秒后重新 Reset ---")
            time.sleep(2.0)
            
    # 退出 Viewer 后清理
    env.close()

def main():
    # 生成一批随机配置
    configs = generate_random_configs()

    while True:
        print(f"\n{'='*30} 环境配置选择菜单 {'='*30}")
        print(f"{'ID':<4} | {'沟宽 (m)':<10} | {'脚部摩擦':<10} | {'身体摩擦'}")
        print("-" * 50)
        for cfg in configs:
            fric = cfg['foot_friction'][0]
            print(f"{cfg['id']:<4} | {cfg['valley_width']:<10.2f} | {fric:<10.2f} | {cfg['body_friction']}")
        
        print("-" * 50)
        choice = input("请输入 ID 进行测试 (输入 q 或 exit 退出): ").strip()
        
        if choice.lower() in ['q', 'exit']:
            print("退出程序。")
            break
            
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(configs):
                # 选中后进入单环境验证函数
                verify_single_env(configs[idx])
            else:
                print("❌ ID 超出范围，请重新输入。")
        else:
            print("❌ 输入无效。")

if __name__ == "__main__":
    main()