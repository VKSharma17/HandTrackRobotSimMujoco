import cv2
import numpy as np
import time
import sys
import random
import mujoco
import mujoco.viewer
from hand_tracker import HandTracker
from spatial_transformer import SpatialTransformer
from robot_sim import RobotSimulator

# Define the plug-and-play robot model selector
# Options: "simple_arm", "franka_panda"
ROBOT_NAME = "franka_panda"

# Variable control for number of pick-up objects in the scene
NUM_CUBES = 5

# Robot configuration mapping
ROBOT_CONFIGS = {
    "simple_arm": {
        "model_path": "simple_arm.xml",
        "mocap_name": "mocap_target",
        "ee_name": "end_effector",
        "workspace_center": (0.45, 0.0, 0.40),
        "scaling_factors": (0.8, 0.8, 0.8),
        "ref_mediapipe": (0.5, 0.5, -0.1),
        "bounds_x": (0.25, 0.65),
        "bounds_y": (-0.35, 0.35),
        "bounds_z": (0.15, 0.65),
        "gripper_actuators": []
    },
    "franka_panda": {
        "model_path": "franka_panda.xml",
        "mocap_name": "mocap_target",
        "ee_name": "hand",
        "workspace_center": (0.45, 0.0, 0.25),
        "scaling_factors": (0.8, 0.8, 3.0),
        "ref_mediapipe": (0.5, 0.5, -0.1),
        "bounds_x": (0.25, 0.65),
        "bounds_y": (-0.3, 0.3),
        "bounds_z": (0.015, 0.5),
        "gripper_actuators": ["finger_actuator1", "finger_actuator2"],
        "mocap_quat": (0.0, 1.0, 0.0, 0.0)
    }
}

def generate_franka_panda_xml(num_cubes: int) -> None:
    """Generates franka_panda.xml dynamically with a variable number of randomized cubes."""
    cube_colors = [
        "0.9 0.45 0.1 1.0",   # Orange
        "0.1 0.6 0.9 1.0",    # Blue
        "0.8 0.1 0.8 1.0",    # Magenta
        "0.9 0.9 0.1 1.0",    # Yellow
        "0.1 0.8 0.5 1.0",    # Teal
        "0.8 0.2 0.2 1.0"     # Red
    ]
    
    positions = []
    attempts = 0
    # Generate non-overlapping random positions on the table surface (Z = 0.02)
    # Table space bounds: X in [0.35, 0.55], Y in [-0.20, 0.20]
    while len(positions) < num_cubes and attempts < 200:
        attempts += 1
        x = random.uniform(0.35, 0.55)
        y = random.uniform(-0.18, 0.18)
        
        too_close = False
        for px, py in positions:
            if np.linalg.norm([x - px, y - py]) < 0.065:
                too_close = True
                break
        if not too_close:
            positions.append((x, y))
            
    # Fallback to structured layout if random allocation timed out
    while len(positions) < num_cubes:
        i = len(positions)
        x = 0.35 + 0.05 * (i % 4)
        y = -0.15 + 0.08 * (i // 4)
        positions.append((x, y))

    cube_bodies_xml = ""
    for i, (x, y) in enumerate(positions):
        rgba = cube_colors[i % len(cube_colors)]
        cube_bodies_xml += f"""
        <!-- Manipulation Cube Object {i} -->
        <body name="cube_{i}" pos="{x:.3f} {y:.3f} 0.02">
            <joint name="cube_joint_{i}" type="free"/>
            <geom name="cube_geom_{i}" type="box" size="0.02 0.02 0.02" rgba="{rgba}" mass="0.05" condim="6" friction="1.0 0.005 0.0001"/>
        </body>"""

    xml_content = f"""<mujoco model="franka_panda_teleop">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="0.002" integrator="RK4">
        <flag energy="enable"/>
    </option>
    
    <default>
        <joint damping="0.1" armature="0.01" limited="true"/>
        <geom condim="3" material="panda_material"/>
    </default>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.15 0.15 0.15" rgb2="0.2 0.2 0.2" width="512" height="512" mark="edge" markrgb="0.3 0.3 0.3"/>
        <material name="grid_material" texture="grid" texrepeat="2 2" texuniform="true"/>
        <material name="panda_material" rgba="0.85 0.85 0.85 1.0"/>
        <material name="joint_material" rgba="0.18 0.18 0.18 1.0"/>
        <material name="gripper_material" rgba="0.3 0.3 0.3 1.0"/>
    </asset>

    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="true"/>
        <geom name="floor" type="plane" size="2 2 0.1" material="grid_material"/>

        <!-- Manipulation Workspace Table (Surface at Z = 0) -->
        <body name="table" pos="0.45 0.0 0.0">
            <geom name="table_top" type="box" size="0.25 0.35 0.01" pos="0 0 -0.01" rgba="0.25 0.25 0.25 1.0"/>
        </body>

        {cube_bodies_xml}

        <!-- 7-DOF Franka Emika Panda Robot Arm -->
        <body name="base" pos="0 0 0.0">
            <geom name="base_geom" type="cylinder" size="0.075 0.05" pos="0 0 0.025" material="joint_material"/>
            
            <body name="link1" pos="0 0 0.05">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-166 166"/>
                <geom name="link1_geom" type="cylinder" size="0.06 0.14" pos="0 0 0.14"/>
                
                <body name="link2" pos="0 0 0.28">
                    <joint name="joint2" type="hinge" axis="0 1 0" range="-101 101"/>
                    <geom name="link2_geom" type="box" size="0.05 0.05 0.1" pos="0 0 0.1"/>
                    
                    <body name="link3" pos="0 0 0.2">
                        <joint name="joint3" type="hinge" axis="0 0 1" range="-166 166"/>
                        <geom name="link3_geom" type="cylinder" size="0.045 0.14" pos="0 0 0.14"/>
                        
                        <body name="link4" pos="0 0 0.28">
                            <joint name="joint4" type="hinge" axis="0 -1 0" range="-176 0"/>
                            <geom name="link4_geom" type="box" size="0.04 0.04 0.1" pos="0 0 0.1"/>
                            
                            <body name="link5" pos="0 0 0.2">
                                <joint name="joint5" type="hinge" axis="0 0 1" range="-166 166"/>
                                <geom name="link5_geom" type="cylinder" size="0.04 0.15" pos="0 0 0.15"/>
                                
                                <body name="link6" pos="0 0 0.3">
                                    <joint name="joint6" type="hinge" axis="0 1 0" range="-5 215"/>
                                    <geom name="link6_geom" type="box" size="0.035 0.035 0.06" pos="0 0 0.06"/>
                                    
                                    <body name="link7" pos="0 0 0.12">
                                        <joint name="joint7" type="hinge" axis="0 0 1" range="-166 166"/>
                                        <geom name="link7_geom" type="cylinder" size="0.035 0.04" pos="0 0 0.04" material="joint_material"/>
                                        
                                        <body name="hand" pos="0 0 0.08">
                                            <geom name="hand_geom" type="box" size="0.04 0.06 0.03" pos="0 0 0.015" material="gripper_material"/>
                                            
                                            <body name="finger_left" pos="0 0.02 0.03">
                                                <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
                                                <geom name="finger_left_geom" type="box" size="0.01 0.01 0.035" pos="0 0 0.035" material="panda_material" condim="4" friction="1.0 0.005 0.0001"/>
                                            </body>
                                            
                                            <body name="finger_right" pos="0 -0.02 0.03">
                                                <joint name="finger_joint2" type="slide" axis="0 -1 0" range="0 0.04"/>
                                                <geom name="finger_right_geom" type="box" size="0.01 0.01 0.035" pos="0 0 0.035" material="panda_material" condim="4" friction="1.0 0.005 0.0001"/>
                                            </body>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>

        <!-- Mocap Target (Initial position Z = 1.51m) -->
        <body name="mocap_target" mocap="true" pos="0.0 0.0 1.51">
            <geom name="mocap_geom" type="sphere" size="0.02" rgba="0 1 0 0.4" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0.06 0 0" rgba="1 0 0 0.8" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0 0.06 0" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0 0 0.06" rgba="0 0 1 0.8" contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <!-- Weld constraint to drag hand with mocap_target -->
    <equality>
        <weld name="teleop_weld" body1="hand" body2="mocap_target"/>
    </equality>

    <!-- Gripper position controllers -->
    <actuator>
        <position name="finger_actuator1" joint="finger_joint1" ctrlrange="0 0.04" kp="100"/>
        <position name="finger_actuator2" joint="finger_joint2" ctrlrange="0 0.04" kp="100"/>
    </actuator>
</mujoco>"""
    with open("franka_panda.xml", "w") as f:
        f.write(xml_content)
    print(f"[INFO] Generated franka_panda.xml with {num_cubes} randomized cubes.")

def main() -> None:
    """
    Module 4: System Integrator
    Orchestrates the real-time teleoperation pipeline.
    Loads plug-and-play robot configurations, processes hand tracking & pinch distance,
    transforms coordinates, applies proportional gripper control, and runs the MuJoCo simulation.
    """
    print(f"[INFO] Initializing Teleoperation System. Selected robot: {ROBOT_NAME}")

    if ROBOT_NAME not in ROBOT_CONFIGS:
        print(f"[CRITICAL] Robot '{ROBOT_NAME}' config not found.")
        sys.exit(1)
        
    config = ROBOT_CONFIGS[ROBOT_NAME]

    # Generate the randomized workspace if using the Franka Panda model
    if ROBOT_NAME == "franka_panda":
        try:
            generate_franka_panda_xml(NUM_CUBES)
        except Exception as e:
            print(f"[ERROR] Failed to dynamically generate Panda XML: {e}")
            sys.exit(1)

    # 1. Initialize Modules
    try:
        tracker = HandTracker(alpha=0.15)
        transformer = SpatialTransformer(
            workspace_center=config["workspace_center"],
            scaling_factors=config["scaling_factors"],
            ref_mediapipe=config["ref_mediapipe"],
            bounds_x=config["bounds_x"],
            bounds_y=config["bounds_y"],
            bounds_z=config["bounds_z"]
        )
        sim = RobotSimulator(
            model_path=config["model_path"],
            mocap_name=config["mocap_name"],
            ee_name=config["ee_name"]
        )
        if "mocap_quat" in config:
            sim.data.mocap_quat[sim.mocap_id] = config["mocap_quat"]
    except Exception as e:
        print(f"[CRITICAL] Initialization failed: {e}")
        sys.exit(1)

    # Resolve gripper actuator IDs in MuJoCo if applicable
    actuator_ids = []
    if config["gripper_actuators"]:
        for act_name in config["gripper_actuators"]:
            act_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id == -1:
                print(f"[WARNING] Actuator '{act_name}' not found in model.")
            else:
                actuator_ids.append(act_id)

    # 2. Initialize Camera Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CRITICAL] Failed to open webcam. Ensure a camera is connected and index 0 is valid.")
        tracker.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Time tracking variables
    dt = sim.model.opt.timestep
    start_time = time.time()
    sim_start_time = sim.data.time
    # Teleoperation state variables for bumpless initialization transfer
    hand_was_lost = True
    start_ee_pos = None
    blend_weight = 1.0
    prev_loop_time = start_time

    print("[INFO] System running. Opening passive simulation viewer on GPU...")
    print("[INFO] Teleoperation loop active. Press 'q' in the camera feed, or close the viewer to exit.")

    # 3. Synchronous Teleoperation Loop
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while cap.isOpened() and viewer.is_running():
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Frame capture dropped.")
                    continue

                # Mirror frame for intuitive human-in-the-loop movement
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Step 1: Run CPU perception to obtain landmarks, thumb-index pinch, and palm scale
                filtered_mp, raw_mp, pinch_dist, d_palm = tracker.process_frame(frame)

                if filtered_mp is not None:
                    if hand_was_lost:
                        # Capture the current physical position of the arm before tracking starts
                        start_ee_pos = sim.get_ee_position()
                        blend_weight = 1.0
                        hand_was_lost = False

                    # Step 2: Transform camera coordinates to MuJoCo workspace coordinates (meters)
                    target_pos = transformer.transform(filtered_mp)

                    # Apply smooth initialization blending (bumpless transfer) to avoid initial snap
                    if blend_weight > 0.0:
                        target_pos = (1.0 - blend_weight) * target_pos + blend_weight * start_ee_pos
                        blend_weight = max(0.0, blend_weight - 0.05)  # Decrements over ~20 frames (~0.6s)

                    # Step 3: Write mapped target directly to the simulator's mocap position
                    sim.set_mocap_position(target_pos)

                    # Step 4: Proportional Gripper Control
                    if actuator_ids:
                        # Map normalized pinch distance [0.05, 0.15] to actuator stroke [0.0 (closed), 0.04 (open)]
                        # Proportional control enables picking with fine displacement
                        gripper_ctrl = np.clip((pinch_dist - 0.05) / 0.10 * 0.04, 0.0, 0.04)
                        for act_id in actuator_ids:
                            sim.data.ctrl[act_id] = gripper_ctrl

                    # Draw text feedback of coordinates in the OpenCV window
                    ee_pos = sim.get_ee_position()
                    tracking_err = np.linalg.norm(target_pos - ee_pos) * 1000.0
                    
                    cv2.putText(
                        frame, f"Target Pos: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}] m", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, f"EE Pos:     [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}] m", 
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                    )
                    cv2.putText(
                        frame, f"Weld Lag:   {tracking_err:.1f} mm", 
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                    )
                    
                    if actuator_ids:
                        pinch_status = "CLOSED" if pinch_dist < 0.06 else "OPEN" if pinch_dist > 0.12 else f"PROP ({pinch_dist:.3f})"
                        cv2.putText(
                            frame, f"Gripper:     {pinch_status}", 
                            (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                        )
                        
                    # Calculate and display hand-to-camera distance (cm) and gripper height (cm)
                    # Palm scale d_palm ranges from 0.09 (~100cm) to 0.22 (~41cm)
                    hand_dist_cm = 9.0 / d_palm * 100.0 if d_palm > 0 else 100.0
                    gripper_height_cm = ee_pos[2] * 100.0
                    
                    cv2.putText(
                        frame, f"Hand Dist:   {hand_dist_cm:.1f} cm", 
                        (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    cv2.putText(
                        frame, f"Gripper Z:   {gripper_height_cm:.1f} cm", 
                        (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                else:
                    hand_was_lost = True
                    cv2.putText(
                        frame, "Hand lost - holding position", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                # Step 5: Step MuJoCo physics simulation to sync with real elapsed time
                elapsed = time.time() - start_time
                steps_needed = int((elapsed - (sim.data.time - sim_start_time)) / dt)
                if steps_needed > 0:
                    sim.step(steps=steps_needed)

                # Compute loop FPS (camera frame processing speed)
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_loop_time)
                prev_loop_time = curr_time

                cv2.putText(
                    frame, f"Loop FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

                # Render camera feed overlay
                cv2.imshow("Physical AI - Teleop Control Center", frame)

                # Step 6: Sync viewer window drawing
                viewer.sync()

                # Break loop on OpenCV window 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"[ERROR] Exception occurred in teleop loop: {e}")
    finally:
        print("[INFO] Shutting down teleoperation pipeline...")
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("[INFO] System shutdown complete.")

if __name__ == "__main__":
    main()
