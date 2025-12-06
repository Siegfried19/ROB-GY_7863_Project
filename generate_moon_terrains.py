import numpy as np
from PIL import Image

def generate_random_moon_png(crater_size, crater_depth,  size=512, crater_count=5, seed=42):
 
    rng = np.random.default_rng(seed)
    Z = np.zeros((size, size), dtype=np.float32)
    X, Y = np.ogrid[:size, :size]

    for _ in range(crater_count):
        cx, cy = rng.integers(0, size, 2)
        # ✅ 坑的半径由 crater_size 控制
        r = int(rng.integers(size * crater_size / 4, size * crater_size))
        crater = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * (r / 8)**2))
        # ✅ 坑的深度由 crater_depth 控制
        Z -= crater * crater_depth

    # 归一化到 [0,1]
    Z -= Z.min()
    Z /= (Z.max() + 1e-8)

    # 转成灰度图
    img = Image.fromarray((Z * 255).astype(np.uint8))
    img.save("unitree_go2/assets/moon_height.png")
    print(f"Saved moon_height.png, shape={Z.shape}, "
          f"crater_size={crater_size}, crater_depth={crater_depth}")



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
generate_rect_trench()