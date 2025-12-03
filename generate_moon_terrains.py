import numpy as np
from PIL import Image

def generate_random_guassian_crater_png(crater_size=0.5, crater_depth=0.08, size=512, crater_count=10, seed=42):
 
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
    
def generate_single_guassian_crater_png(crater_size=0.5, crater_depth=1, size=512, crater_steepness=3, seed=42):
    Z = np.zeros((size, size), dtype=np.float32)
    X, Y = np.ogrid[:size, :size]

    cx, cy = size // 2, size // 2
    r = int(size * crater_size)/crater_steepness
    crater = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * r**2))
    Z -= crater 

    # 归一化到 [0,1]
    Z -= Z.min()
    Z /= (Z.max() + 1e-8)
    Z *= crater_depth

    # 转成灰度图
    img = Image.fromarray((Z * 255).astype(np.uint8))
    img.save("unitree_go2/assets/moon_height.png")
    print(f"Saved moon_height.png, shape={Z.shape}, "
          f"crater_size={crater_size}, crater_depth={crater_depth}")
    
def generate_single_crater_png(crater_size=0.3, crater_depth=1, size=512, flat_ratio=0.2, seed=42, out_path="unitree_go2/assets/moon_height.png" ):
    Z = np.zeros((size, size), dtype=np.float32)
    X, Y = np.ogrid[:size, :size]

    cx, cy = size // 2, size // 2
    img_radius = size / 2.0
    
    R_outer = crater_size * img_radius
    R_inner = flat_ratio * R_outer
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)

    H = np.ones_like(r, dtype=np.float32)
    H[r <= R_inner] = 0.0
    
    mask_transition = (r > R_inner) & (r < R_outer)
    t = (r[mask_transition] - R_inner) / (R_outer - R_inner)
    smooth = t * t * (3.0 - 2.0 * t)   # cubic smoothstep
    H[mask_transition] = smooth
    
    Z = H * crater_depth
    
    img = Image.fromarray((Z * 255).astype(np.uint8))
    img.save(out_path)
    # print(
    #     f"Saved moon_height.png, shape={Z.shape}, "
    #     f"crater_size={crater_size}, crater_depth={crater_depth}, "
    #     f"R_inner={R_inner:.1f}px (flat bottom), R_outer={R_outer:.1f}px (leave crater)"
    # )

# generate_random_moon_png(crater_size=0.5, crater_depth=0.02)
if __name__ == "__main__":
    generate_single_crater_png(crater_size=0.5, crater_depth=1, flat_ratio=0.3)
