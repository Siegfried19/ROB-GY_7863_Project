import numpy as np
from PIL import Image

def generate_moon_png(size=256, crater_count=20, seed=42):
    rng = np.random.default_rng(seed)
    Z = np.zeros((size, size), dtype=np.float32)

    X, Y = np.ogrid[:size, :size]
    for _ in range(crater_count):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(size // 30, size // 8)
        crater = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2*(r/2)**2))
        Z -= crater * 0.1

    # 归一化到 0~255 并转灰度图
    Z -= Z.min()
    Z /= Z.max()
    img = Image.fromarray((Z * 255).astype(np.uint8))
    img.save("moon_height.png")
    print("✅ Saved moon_height.png, shape:", Z.shape)

generate_moon_png()
