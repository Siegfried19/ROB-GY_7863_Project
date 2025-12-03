import mujoco

model = mujoco.MjModel.from_xml_path("unitree_go2/go2.xml")

for j in range(model.njnt):
    print(j,
          model.joint(j).name,
          "qpos adr =", model.jnt_qposadr[j],
          "dof adr =", model.jnt_dofadr[j])