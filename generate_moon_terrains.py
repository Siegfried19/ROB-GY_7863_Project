import numpy as np
from PIL import Image

def generate_rough_moon_terrain(
    size=512,
    world_size_m=10.0,
    max_bump_height=0.10,      # 小坡高度（正向 +0.2m）
    max_pit_depth=0.15,        # 坑深度 （向下 -0.25m，不再夸张）
    smoothness=12,             # 越大越平滑，越小越崎岖(推荐8~18)
    output_path="unitree_go2/assets/moon_height_rough.png"
):
    """
    生成不平坦月壤地形 ---- 带噪声起伏、浅坑而非垂直深沟。
    真实高度保存在 Z（单位 meter），映射为 PNG 用于地形加载。
    """

    # 网格坐标
    x = np.linspace(-world_size_m/2, world_size_m/2, size)
    y = np.linspace(-world_size_m/2, world_size_m/2, size)
    X, Y = np.meshgrid(x, y)

    # ============================
    # 基础随机山丘噪声 Perlin/Fractional Brownian Motion 风格
    # ============================
    Z = np.zeros((size, size))

    for i in range(5):                                    # 多频率叠加（FBM风格山丘）
        freq = (i+1) * 0.05 * smoothness
        amp  = (max_bump_height - max_pit_depth) * (0.5**i)

        Z += amp * (
            np.sin(freq * X) * np.cos(freq * Y) +
            0.5*np.sin(freq*0.8*X+1)*np.sin(freq*0.5*Y-1)
        )

    # ============================
    # 控制高度范围：坑更浅、整体可控
    # ============================
    Z = np.clip(Z, -max_pit_depth, max_bump_height)

    # 映射到 PNG 灰度 [0,255]
    Z_img = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
    Z_img = (Z_img * 255).astype(np.uint8)

    img = Image.fromarray(Z_img)
    img.save(output_path)

    print("Saved:", output_path)
    print(f"Terrain Height Range: {Z.min():.3f}m ~ {Z.max():.3f}m")



def generate_crater_at_origin(
    radius_m=2.0,
    depth_m=1.0,
    size=512,
    world_size_m=10.0,
    output_path="unitree_go2/assets/moon_height.png"
):
    """
    生成一个以 (0,0) 为中心，指定半径和深度的圆形月球坑.
    
    参数说明:
    - radius_m: 坑的半径（米）
    - depth_m: 坑的深度（米）
    - size: height map 的分辨率 (size x size)
    - world_size_m: MuJoCo 的地面物理尺寸对应的总宽度（米）
    - output_path: 保存的 PNG 文件路径
    """
    
    # 生成坐标网格（以 heightfield 中心为 (0,0)）
    x = np.linspace(-world_size_m/2, world_size_m/2, size)
    y = np.linspace(-world_size_m/2, world_size_m/2, size)
    X, Y = np.meshgrid(x, y)

    # 计算到中心的距离
    R = np.sqrt(X**2 + Y**2)

    # 高斯形状的坑 (exp 型凹陷)
    crater = np.exp(-(R**2) / (2 * (radius_m / 2)**2))

    # 深度映射
    Z = -depth_m * crater  # crater 深度（以米为单位）

    # Normalize → [0,1]
    Z -= Z.min()
    Z /= (Z.max() + 1e-8)

    # 保存为灰度 PNG
    img = Image.fromarray((Z * 255).astype(np.uint8))
    img.save(output_path)

    print(f"Saved: {output_path} (radius={radius_m}m, depth={depth_m}m, size={size}×{size})")



def generate_rect_trench(
    x_start_m=0.25,           # 沟开始
    x_end_m=1.25,             # 沟结束（宽 2m）
    depth_m=3.0,             # 沟深度，向下为负
    size=512,
    world_size_m=10.0,
    output_path="unitree_go2/assets/moon_trench_down.png"
):
    """
    生成一条沿 y 方向延伸、地面高度为 0、沟槽高度为 -depth_m 的矩形沟槽。
    PNG 里会线性缩放到 [0,255]，但真实高度语义保持不变。
    """

    # 网格
    x = np.linspace(-world_size_m/2, world_size_m/2, size)
    y = np.linspace(-world_size_m/2, world_size_m/2, size)
    X, Y = np.meshgrid(x, y)

    # 真实高度（MuJoCo 语义）
    Z = np.zeros_like(X)                # 地面高度 = 0
    Z[(X >= x_start_m) & (X <= x_end_m)] = -depth_m   # 沟槽 = -1m

    # 将 [-depth_m, 0] 映射到 [0,255] 以存储 PNG
    Z_img = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)  # 0 → 沟底, 1 → 地面
    Z_img = (Z_img * 255).astype(np.uint8)

    img = Image.fromarray(Z_img)
    img.save(output_path)

    print("Saved:", output_path)
    print(f"Trench from x={x_start_m} to x={x_end_m}, depth={depth_m}m")
# 示例使用：
# 小而深的坑
#generate_random_moon_png(crater_size=0.6, crater_depth=0.03)

# 大而浅的坑
# generate_moon_png(crater_size=0.6, crater_depth=0.01)


#在原点位置
# generate_crater_at_origin(
#     radius_m=2.0,
#     depth_m=2.0,
#     size=512,
#     world_size_m=10.0
# )
#generate_rect_trench()
generate_rough_moon_terrain()