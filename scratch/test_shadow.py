import mujoco
import numpy as np

def test():
    model = mujoco.MjModel.from_xml_path("shadow_hand_test/dual_hands.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    lh_forearm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lh_forearm")
    rh_forearm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_forearm")
    
    print("lh_forearm pos:", data.xpos[lh_forearm_id])
    print("lh_forearm quat:", data.xquat[lh_forearm_id])
    
    print("rh_forearm pos:", data.xpos[rh_forearm_id])
    print("rh_forearm quat:", data.xquat[rh_forearm_id])

if __name__ == "__main__":
    test()
