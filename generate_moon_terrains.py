import numpy as np
from PIL import Image

def generate_random_moon_png(crater_size=0.5, crater_depth=0.08, size=512, crater_count=10, seed=42):
 
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

# 示例使用：
# 小而深的坑
generate_random_moon_png(crater_size=0.4, crater_depth=0.03)

# 大而浅的坑
# generate_moon_png(crater_size=0.6, crater_depth=0.01)
