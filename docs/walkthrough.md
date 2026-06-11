# Walkthrough: Real-Time Hand Teleoperated Robot Sim (Phase 2 Complete)

We have successfully extended the teleoperation pipeline to support high-fidelity manipulation using a 7-DOF **Franka Emika Panda** robot arm, proportional gripper finger controls mapped to your hand pinching distance, and stable contact dynamics to pick and place a cube.

---

## 1. Summary of Changes

### Robot Model: Franka Emika Panda
*   **[franka_panda.xml](file:///d:/VKS/VKSLearn/HandRobotSimMujoco/franka_panda.xml)**: Represents a meshless 7-DOF Franka Panda arm with parallel sliding jaws (`finger_joint1`, `finger_joint2`). It sits on a table workspace ($Z = 0.0\text{ m}$) containing a pick-and-place target cube ($4 \times 4 \times 4\text{ cm}$) with a free joint.
*   **Stable Contact Friction**: We configured the cube and fingers with rolling and torsional friction (`friction="1.5 0.01 0.0005"`) and contact dimensions of 6 (`condim="6"`) to enable stable friction force constraints, preventing the cube from sliding or rotating out of the jaws.

### Perception Engine
*   **[hand_tracker.py](file:///d:/VKS/VKSLearn/HandRobotSimMujoco/hand_tracker.py)**: Now extracts the 3D position of **Thumb Tip (Landmark 4)** and **Index Finger Tip (Landmark 8)**. It calculates the normalized Euclidean distance $d$ and draws a colored connection line: it blends from **Red** (fully open) to **Green** (pinched closed) to give visual feedback.

### Plug-and-Play System Integrator
*   **[main.py](file:///d:/VKS/VKSLearn/HandRobotSimMujoco/main.py)**:
    1. Established a `ROBOT_CONFIGS` dictionary holding XML paths, workspace centers, safety bounds, and effector names for both `simple_arm` and `franka_panda`.
    2. Added `ROBOT_NAME = "franka_panda"` at the top as a dynamic plug-and-play selector.
    3. Implemented proportional gripper mapping:
       $$\text{stroke} = \text{clip}\left( \frac{d - 0.05}{0.10} \times 0.04, 0.0, 0.04 \right)$$
       This writes to `data.ctrl` to control the finger sliding joints.

---

## 2. Validation & Testing Results

1. **Franka Panda Kinematics**: Running `python main.py` successfully loaded the 7-DOF XML model, resolved the two finger actuators, and aligned the robot's end-effector hand to the mocap target.
2. **Proportional Pinch Control**:
   *   Fingers wide apart ($d \ge 0.15$): Actuator stroke maps to $0.04\text{ m}$ (gripper fully open).
   *   Fingers touching ($d \le 0.05$): Actuator stroke maps to $0.00\text{ m}$ (gripper fully closed).
   *   OpenCV feedback: The connection line turns solid green when pinched and red when open, indicating the exact gripper status.
3. **Workspace Bounds**: Lowering the minimum height $Z_{\text{mj}}$ safety bound to $0.015\text{ m}$ allows the gripper jaws to descend directly onto the table surface to pick up the target cube without colliding with the floor.

---

## 3. How to Run the Teleoperation
To execute the fully integrated physical AI pipeline:
```bash
python main.py
```
*   To switch back to the simple arm, open `main.py` and set `ROBOT_NAME = "simple_arm"` at the top.
*   Pinch your thumb and index finger together to close the gripper on the cube, lift it, and move it.
*   Press **'q'** in the camera window to quit.

---

## 4. Bug Fixes
*   **Perception Return Mismatch**: Resolved `ValueError: not enough values to unpack (expected 4, got 3)` by ensuring `process_frame()` in [hand_tracker.py](file:///d:/VKS/VKSLearn/HandRobotSimMujoco/hand_tracker.py) correctly returns `d_palm` alongside the filtered/raw coordinates and pinch distance.
*   **Variable Scope / Initialization**: Resolved a potential `NameError: name 'prev_loop_time' is not defined` by initializing `prev_loop_time` to `start_time` in [main.py](file:///d:/VKS/VKSLearn/HandRobotSimMujoco/main.py) before entering the teleoperation loop.

---

## 5. Industrial Model Integration (MuJoCo Menagerie)
We integrated the official high-fidelity **Franka Emika Panda** model from the DeepMind `mujoco_menagerie` repository as a selectable, plug-and-play configuration option (`franka_panda_industrial`).

### Highlights
1. **Auto-Downloader**: Built automatic asset fetching into the XML generator. If the official model folder is not found locally, the script downloads the MJCF XML and all 67 OBJ/STL visual/collision meshes from GitHub raw storage.
2. **Recursive XML Prefixing**: Developed a recursive parser that duplicates the robot base (`link0`) tree, appending suffixes `_left` and `_right` to all joint, body, and geom names to avoid XML compilation naming collisions.
3. **Home Joint Pose Initialization**: Sets `sim.data.qpos` to the official home keyframe angles on load:
   `home_val = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]`
   This pre-configures both arms in a natural, stable bent configuration on startup.
4. **Target Alignment**: Symmetrically places the default mocap targets and the spatial transformer center at the exact end-effector home coordinate:
   * Left: `[0.554, 0.25, 0.625]`
   * Right: `[0.554, -0.25, 0.625]`
   This eliminates violent startup snaps.

