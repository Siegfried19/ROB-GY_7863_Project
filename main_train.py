import mujoco
import mujoco.viewer
import numpy as np

# 1. 加载 XML 模型（例如 humanoid.xml）
model = mujoco.MjModel.from_xml_path("unitree_go2/scene_moon.xml")

# 2. 创建仿真数据对象
data = mujoco.MjData(model)
dog_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "go2")

for i in range(3):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"step{i}")
    model.geom_pos[geom_id, 2] = 10 * np.random.uniform(1, 3)

# 4. 打开可视化窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        # 获取狗的世界坐标位置
        dog_pos = data.xpos[dog_id]

        # 设置相机位置与方向（跟随效果）
        viewer.cam.lookat[:] = dog_pos              # 相机对准狗
        viewer.cam.distance = 3.0                   # 相机与狗的距离
        viewer.cam.elevation = -10                  # 俯仰角（-10°）
        viewer.cam.azimuth = 180                    # 水平方向（180°背后视角）

        viewer.sync()
