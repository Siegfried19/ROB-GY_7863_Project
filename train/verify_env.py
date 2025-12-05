import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
from go2_env import Go2EnvMoonFly

# 定义参数随机范围
PARAM_RANGES = {
    "foot_slide": (0.6, 1.0),      # 摩擦力范围
    "valley_width": (1.5, 4.0),    # 沟宽范围 [1.5米, 4.0米]
}

def generate_random_configs(num_envs=5):
    """生成 N 组随机配置"""
    configs = []
    for i in range(num_envs):
        # 1. 随机生成摩擦力
        f_slide = np.random.uniform(*PARAM_RANGES["foot_slide"])
        foot_fric_vector = [f_slide, 0.005, 0.005]
        
        # 2. 随机生成沟宽 (使用新的变量名 valley_width)
        v_width = np.random.uniform(*PARAM_RANGES["valley_width"])
        
        config = {
            "id": i,
            "foot_friction": foot_fric_vector,
            "body_friction": 0.8,
            "valley_width": v_width,  # 直接作为顶层变量
        }
        configs.append(config)
    return configs

def verify_and_render(config):
    """创建环境，验证几何参数，并启动渲染"""
    print(f"\n{'-'*60}")
    print(f"正在加载环境 ID [{config['id']}] ...")
    
    # 1. 初始化环境 (适配你修改后的 __init__)
    try:
        env = Go2EnvMoonFly(
            # 请确保 xml_path 和你实际文件名一致 (如果之前是 velly 没改，这里保持原样)
            xml_path="../unitree_go2/scene_moon_valley_jet.xml", 
            foot_friction=config["foot_friction"],
            body_friction=config["body_friction"],
            valley_width=config["valley_width"], # <--- 关键修改：直接传参
            rank=config["id"]
        )
        env.reset()
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        print("提示：请检查 go2_env.py 的 __init__ 是否已接收 valley_width 参数")
        return

    # 2. 物理/几何验证 (Math Check)
    width = config['valley_width']
    print(f"  [目标参数]")
    print(f"  > 设定沟宽 (Valley Width): {width:.4f} m")
    print(f"  > 设定摩擦 (Friction)    : {config['foot_friction'][0]:.2f}")

    # 获取 'platform_end' 的位置
    geom_name = "platform_end"
    geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    
    if geom_id != -1:
        # 获取实际数值
        actual_x = env.model.geom_pos[geom_id][0]
        half_len = env.model.geom_size[geom_id][0]
        
        # 计算预期数值 
        # 基于 scene_moon_velly_jet.xml: 
        # 起点平台中心=0, 半长=2.0 => 右边缘在 x=2.0
        start_edge = 2.0
        expected_center = start_edge + width + half_len
        
        print(f"  [几何验证]")
        print(f"  > 起跳边缘 (固定)     : {start_edge:.1f} m")
        print(f"  > 终点平台半长        : {half_len:.1f} m")
        print(f"  > 预期中心点 X        : {expected_center:.4f} m")
        print(f"  > 实际中心点 X        : {actual_x:.4f} m")
        
        if abs(actual_x - expected_center) < 1e-4:
            print("  ✅ 验证通过: 平台位置计算正确！")
        else:
            print("  ❌ 验证失败: 平台位置不对，请检查 go2_env.py 计算逻辑！")
    else:
        print(f"  ❌ 错误: 找不到名为 '{geom_name}' 的物体")

    # 3. 检查机器狗位置
    robot_x = env.data.qpos[0]
    print(f"  [出生点验证]")
    print(f"  > 机器狗 X 坐标       : {robot_x:.4f} m (预期约 1.5)")

    # 4. 启动 3D 渲染
    print(f"\n📺 正在启动 MuJoCo 查看器...")
    print(f"   (按 ESC 关闭窗口返回菜单)")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # 发送全 0 动作，让它自然站立/下落
            env.step(np.zeros(env.action_space.shape))
            viewer.sync()
            time.sleep(0.002)

    env.close()
    print("窗口已关闭。")

def main():
    # 1. 生成一批配置
    num_envs = 5
    print(f"正在生成 {num_envs} 个随机环境配置...\n")
    configs = generate_random_configs(num_envs)

    while True:
        # 2. 打印菜单
        print(f"\n{'='*20} 环境选择菜单 {'='*20}")
        print(f"{'ID':<5} | {'Valley Width (m)':<18} | {'Friction':<10}")
        print("-" * 45)
        for cfg in configs:
            w = cfg['valley_width']
            f = cfg['foot_friction'][0]
            print(f"{cfg['id']:<5} | {w:<18.4f} | {f:<10.4f}")
        print("-" * 45)
        
        # 3. 获取用户输入
        choice = input("\n请输入要查看的环境 ID (输入 q 退出): ").strip()
        
        if choice.lower() == 'q':
            print("退出程序。")
            break
        
        if not choice.isdigit():
            print("输入无效，请输入数字 ID。")
            continue
            
        idx = int(choice)
        if 0 <= idx < num_envs:
            # 4. 进入渲染与验证
            verify_and_render(configs[idx])
        else:
            print(f"ID {idx} 超出范围，请输入 0 到 {num_envs-1} 之间的数字。")

if __name__ == "__main__":
    main()