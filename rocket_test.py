import os
from typing import List, Sequence, Tuple
import mujoco
import mujoco.viewer
import numpy as np
# ====== 修改这里来控制测试行为 ======
MODEL_PATH = os.path.join(os.path.dirname(__file__), "unitree_go2", "scene_moon.xml")
LOCK_JOINTS = True  # 改成 False 就不会锁定任何关节
JOINTS_TO_LOCK = ["root"]  # 想锁定的关节名称列表，"root" 表示整个机身
THRUST_FORCE = 2.0  # 四个火箭同时喷射的推力（牛顿）
SIM_DURATION = 50.0  # 模拟时长（秒）
# ===================================
THRUSTER_NAMES: Tuple[str, ...] = (
    "FL_rocket_thruster",
    "FR_rocket_thruster",
    "RL_rocket_thruster",
    "RR_rocket_thruster",
)
def _joint_sizes(joint_type: int) -> Tuple[int, int]:
    jt = mujoco.mjtJoint
    if joint_type == jt.mjJNT_FREE:
        return 7, 6
    if joint_type == jt.mjJNT_BALL:
        return 4, 3
    return 1, 1
def _prepare_freeze(
    model: mujoco.MjModel, data: mujoco.MjData, joint_names: Sequence[str]
):
    if not joint_names:
        return []
    name_to_id = {}
    for j in range(model.njnt):
        jname = model.joint(j).name
        if jname:
            name_to_id[jname] = j
    if not model.joint(0).name:
        name_to_id["root"] = 0
    frozen = []
    for name in joint_names:
        if name not in name_to_id:
            print(f"警告: 找不到关节 {name}")
            continue
        j_id = name_to_id[name]
        qpos_adr = model.jnt_qposadr[j_id]
        dof_adr = model.jnt_dofadr[j_id]
        qpos_count, dof_count = _joint_sizes(model.jnt_type[j_id])
        frozen.append(
            (
                slice(qpos_adr, qpos_adr + qpos_count),
                slice(dof_adr, dof_adr + dof_count),
                np.array(data.qpos[qpos_adr : qpos_adr + qpos_count], copy=True),
            )
        )
    return frozen
def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件找不到：{MODEL_PATH}")
    print(f"加载模型：{MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    # frozen = _prepare_freeze(model, data, JOINTS_TO_LOCK if LOCK_JOINTS else [])
    # if frozen:
    #     print("锁定的关节：", ", ".join(JOINTS_TO_LOCK))
    # else:
    #     print("没有锁定任何关节。")
    thruster_ids: List[int] = []
    for name in THRUSTER_NAMES:
        try:
            thruster_ids.append(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            )
        except mujoco.Error:
            raise ValueError(f"模型里找不到火箭执行器：{name}")
    print(f"四个火箭喷射力：{THRUST_FORCE:.1f} N")
    print(f"模拟时长：{SIM_DURATION:.1f} s（关闭窗口即可提前结束）")
    dog_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "go2")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < SIM_DURATION:
            # get world coordinate
            dog_pos = data.xpos[dog_id]

            # camera setting
            viewer.cam.lookat[:] = dog_pos             
            viewer.cam.distance = 3.0                
            viewer.cam.elevation = -10         
            viewer.cam.azimuth = 180  
            data.ctrl[:] = 0.0
            for act_id in thruster_ids:
                data.ctrl[act_id] = THRUST_FORCE
            mujoco.mj_step(model, data)
            # for qpos_sl, dof_sl, ref in frozen:
            #     data.qpos[qpos_sl] = ref
            #     data.qvel[dof_sl] = 0.0
            # if frozen:
            mujoco.mj_forward(model, data)
            viewer.sync()
    print(f"模拟结束，最终时间：{data.time:.2f} s")
    print(f"最终底座位置：{data.qpos[:3]}")
if __name__ == "__main__":
    main()