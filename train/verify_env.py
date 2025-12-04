import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import time
from go2_env import Go2EnvMoonFly

# 复制你 main_train_sb3.py 中的参数范围，保持一致
PARAM_RANGES = {
    "foot_slide": (0.3, 1.0),
    "foot_spin":  (0.005, 0.1),
    "foot_roll":  (0.001, 0.02),
    "crater_size":   (0.2, 0.6),
    "crater_depth":  (0.5, 1.5),
    "flat_ratio":    (0.1, 0.3),
}

# --- 新增函数：用于 3D 渲染指定配置的环境 ---
def render_env_in_3d(config):
    """
    接收一个环境配置字典，重新创建环境并启动 3D 渲染窗口
    """
    print(f"\n>>> 正在启动 Env {config['rank']} 的 3D 视图...")
    print(f"    参数: 深度={config['crater_config']['depth']:.2f}, 摩擦={config['foot_friction'][0]:.2f}")
    
    # 重新实例化环境
    env = Go2EnvMoonFly(
        xml_path="../unitree_go2/scene_moon_jet.xml",
        foot_friction=config["foot_friction"],
        body_friction=config["body_friction"],
        crater_config=config["crater_config"],
        rank=config["rank"] # 关键：传入同样的 rank 确保随机地形种子一致
    )
    
    env.reset()

    print("按 [ESC] 退出查看，按空格开始/暂停物理")
    
    # 启动被动查看器
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # 这里给全 0 动作，或者你可以给一个站立动作
            action = np.zeros(env.action_space.shape)
            env.step(action)

            # 同步画面
            viewer.sync()

            # 简单的帧率控制
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    env.close()
    print(">>> 3D 视图已关闭\n")

def verify_environments():
    num_envs = 8
    
    # 准备画布：2行4列，用来画8个地形
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # 固定种子，确保复现性
    np.random.seed(2323)

    print(f"{'='*30} 开始验证环境参数 {'='*30}")

    saved_configs = []
    
    for i in range(num_envs):
        # 1. 模拟参数采样 (和 Main 函数逻辑一致)
        f_slide = np.random.uniform(*PARAM_RANGES["foot_slide"])
        f_spin  = np.random.uniform(*PARAM_RANGES["foot_spin"])
        f_roll  = np.random.uniform(*PARAM_RANGES["foot_roll"])
        foot_fric_vector = [f_slide, f_spin, f_roll]
        body_fric_scalar = f_slide # 你的绑定逻辑

        c_size  = np.random.uniform(*PARAM_RANGES["crater_size"])
        c_depth = np.random.uniform(*PARAM_RANGES["crater_depth"])
        c_flat  = np.random.uniform(*PARAM_RANGES["flat_ratio"])

        current_config = {
            "xml_path": "../unitree_go2/scene_moon_jet.xml",
            "foot_friction": foot_fric_vector,
            "body_friction": body_fric_scalar,
            "crater_config": {
                "size": c_size, 
                "depth": c_depth, 
                "flat_ratio": c_flat
            },
            "rank": i
        }
        saved_configs.append(current_config)
        
        # 2. 创建环境
        env = Go2EnvMoonFly(**current_config)

        # 3. --- 验证 A: 获取地形数据 ---
        # 找到 hfield 的地址
        hfield_name = "moon_hf"
        hfield_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
        nrow = env.model.hfield_nrow[hfield_id]
        ncol = env.model.hfield_ncol[hfield_id]
        start_addr = env.model.hfield_adr[hfield_id]
        
        # 读取内存中的高度数据
        hfield_data = env.model.hfield_data[start_addr : start_addr + nrow*ncol]
        hfield_grid = hfield_data.reshape(nrow, ncol)

        # 4. --- 验证 B: 检查摩擦力 ---
        # 获取左前脚 (FL) 的摩擦力来验证
        fl_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "FL")
        actual_foot_fric = env.model.geom_friction[fl_id]
        
        # 打印验证信息
        print(f"[Env {i}]")
        print(f"  设定地形: Size={c_size:.2f}, Depth={c_depth:.2f}")
        print(f"  设定摩擦: Foot={np.round(foot_fric_vector, 3)}")
        print(f"  实际内存: Foot={np.round(actual_foot_fric, 3)}") # 应该与设定完全一致
        
        # 5. 画图
        im = axes[i].imshow(hfield_grid, cmap='terrain', origin='lower')
        axes[i].set_title(f"Env {i}\nD={c_depth:.1f}m, F={f_slide:.2f}")
        axes[i].axis('off')
        
        # 用完记得关闭，释放内存
        env.close()

    plt.tight_layout()
    # plt.savefig("verify_terrain_results.png")
    print(f"\n{'-'*60}")
    print("验证完成！请查看生成的图片: verify_terrain_results.png")
    print("如果图片中的坑大小不一，且打印的摩擦力参数各不相同，说明环境创建成功。")
    plt.show()
    
    while True:
        try:
            user_input = input("请输入你想查看 3D 效果的环境编号 (0-7)，输入 q 退出: ")
            if user_input.lower() == 'q':
                break
            
            idx = int(user_input)
            if 0 <= idx < num_envs:
                # 调用我们在上面定义的新函数
                render_env_in_3d(saved_configs[idx])
            else:
                print(f"请输入 0 到 {num_envs-1} 之间的数字")
        except ValueError:
            print("输入无效，请输入数字或 q")
    

if __name__ == "__main__":
    verify_environments()
    