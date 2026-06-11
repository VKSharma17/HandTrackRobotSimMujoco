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
ROBOT_NAME = "franka_panda_industrial"

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
    "franka_panda_geometric": {
        "model_path": "franka_panda.xml",
        "mocap_name_left": "mocap_target_left",
        "mocap_name_right": "mocap_target_right",
        "ee_name_left": "hand_left",
        "ee_name_right": "hand_right",
        "workspace_center_left": (0.45, 0.25, 0.25),
        "workspace_center_right": (0.45, -0.25, 0.25),
        "scaling_factors": (0.8, 0.8, 3.0),
        "ref_mediapipe": (0.5, 0.5, -0.1),
        "bounds_x": (0.25, 0.65),
        "bounds_y_left": (-0.05, 0.55),
        "bounds_y_right": (-0.55, 0.05),
        "bounds_z": (0.015, 0.5),
        "gripper_actuators_left": ["finger_actuator1_left", "finger_actuator2_left"],
        "gripper_actuators_right": ["finger_actuator1_right", "finger_actuator2_right"],
        "mocap_quat": (0.0, 1.0, 0.0, 0.0)
    },
    "franka_panda_industrial": {
        "model_path": "franka_panda_industrial.xml",
        "mocap_name_left": "mocap_target_left",
        "mocap_name_right": "mocap_target_right",
        "ee_name_left": "hand_left",
        "ee_name_right": "hand_right",
        "workspace_center_left": (0.554, 0.25, 0.25),
        "workspace_center_right": (0.554, -0.25, 0.25),
        "scaling_factors": (0.8, 0.8, 3.0),
        "ref_mediapipe": (0.5, 0.5, -0.1),
        "bounds_x": (0.25, 0.65),
        "bounds_y_left": (-0.05, 0.55),
        "bounds_y_right": (-0.55, 0.05),
        "bounds_z": (0.015, 0.65),
        "gripper_actuators_left": ["finger_actuator1_left", "finger_actuator2_left"],
        "gripper_actuators_right": ["finger_actuator1_right", "finger_actuator2_right"],
        "mocap_quat": (0.0, 1.0, 0.0, 0.0)
    }
}

def generate_franka_panda_xml(num_cubes: int) -> None:
    """Generates franka_panda.xml dynamically with a variable number of randomized cubes and two arms."""
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
    # Table space bounds: X in [0.35, 0.55], Y in [-0.35, 0.35]
    while len(positions) < num_cubes and attempts < 200:
        attempts += 1
        x = random.uniform(0.35, 0.55)
        y = random.uniform(-0.35, 0.35)
        
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
        y = -0.25 + 0.1 * (i // 4)
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

    def get_panda_arm_xml(suffix: str, base_pos: str) -> str:
        return f"""
        <!-- 7-DOF Franka Emika Panda Robot Arm ({suffix}) -->
        <body name="base{suffix}" pos="{base_pos}">
            <geom name="base_geom{suffix}" type="cylinder" size="0.075 0.05" pos="0 0 0.025" material="joint_material"/>
            
            <body name="link1{suffix}" pos="0 0 0.05">
                <joint name="joint1{suffix}" type="hinge" axis="0 0 1" range="-166 166"/>
                <geom name="link1_geom{suffix}" type="cylinder" size="0.06 0.14" pos="0 0 0.14"/>
                
                <body name="link2{suffix}" pos="0 0 0.28">
                    <joint name="joint2{suffix}" type="hinge" axis="0 1 0" range="-101 101"/>
                    <geom name="link2_geom{suffix}" type="box" size="0.05 0.05 0.1" pos="0 0 0.1"/>
                    
                    <body name="link3{suffix}" pos="0 0 0.2">
                        <joint name="joint3{suffix}" type="hinge" axis="0 0 1" range="-166 166"/>
                        <geom name="link3_geom{suffix}" type="cylinder" size="0.045 0.14" pos="0 0 0.14"/>
                        
                        <body name="link4{suffix}" pos="0 0 0.28">
                            <joint name="joint4{suffix}" type="hinge" axis="0 -1 0" range="-176 0"/>
                            <geom name="link4_geom{suffix}" type="box" size="0.04 0.04 0.1" pos="0 0 0.1"/>
                            
                            <body name="link5{suffix}" pos="0 0 0.2">
                                <joint name="joint5{suffix}" type="hinge" axis="0 0 1" range="-166 166"/>
                                <geom name="link5_geom{suffix}" type="cylinder" size="0.04 0.15" pos="0 0 0.15"/>
                                
                                <body name="link6{suffix}" pos="0 0 0.3">
                                    <joint name="joint6{suffix}" type="hinge" axis="0 1 0" range="-5 215"/>
                                    <geom name="link6_geom{suffix}" type="box" size="0.035 0.035 0.06" pos="0 0 0.06"/>
                                    
                                    <body name="link7{suffix}" pos="0 0 0.12">
                                        <joint name="joint7{suffix}" type="hinge" axis="0 0 1" range="-166 166"/>
                                        <geom name="link7_geom{suffix}" type="cylinder" size="0.035 0.04" pos="0 0 0.04" material="joint_material"/>
                                        
                                        <body name="hand{suffix}" pos="0 0 0.08">
                                            <geom name="hand_geom{suffix}" type="box" size="0.04 0.06 0.03" pos="0 0 0.015" material="gripper_material"/>
                                            
                                            <body name="finger_left{suffix}" pos="0 0.02 0.03">
                                                <joint name="finger_joint1{suffix}" type="slide" axis="0 1 0" range="0 0.04"/>
                                                <geom name="finger_left_geom{suffix}" type="box" size="0.01 0.01 0.035" pos="0 0 0.035" material="panda_material" condim="4" friction="1.0 0.005 0.0001"/>
                                            </body>
                                            
                                            <body name="finger_right{suffix}" pos="0 -0.02 0.03">
                                                <joint name="finger_joint2{suffix}" type="slide" axis="0 -1 0" range="0 0.04"/>
                                                <geom name="finger_right_geom{suffix}" type="box" size="0.01 0.01 0.035" pos="0 0 0.035" material="panda_material" condim="4" friction="1.0 0.005 0.0001"/>
                                            </body>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>"""

    def get_mocap_xml(suffix: str, init_pos: str) -> str:
        return f"""
        <!-- Mocap Target for {suffix} -->
        <body name="mocap_target{suffix}" mocap="true" pos="{init_pos}">
            <geom name="mocap_geom{suffix}" type="sphere" size="0.02" rgba="0 1 0 0.4" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0.06 0 0" rgba="1 0 0 0.8" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0 0.06 0" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0 0 0.06" rgba="0 0 1 0.8" contype="0" conaffinity="0"/>
        </body>"""

    left_arm_xml = get_panda_arm_xml("_left", "0 0.25 0.0")
    right_arm_xml = get_panda_arm_xml("_right", "0 -0.25 0.0")
    
    left_mocap_xml = get_mocap_xml("_left", "0.0 0.25 1.51")
    right_mocap_xml = get_mocap_xml("_right", "0.0 -0.25 1.51")

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
            <geom name="table_top" type="box" size="0.25 0.50 0.01" pos="0 0 -0.01" rgba="0.25 0.25 0.25 1.0"/>
        </body>

        {cube_bodies_xml}

        {left_arm_xml}
        {right_arm_xml}

        {left_mocap_xml}
        {right_mocap_xml}
    </worldbody>

    <!-- Weld constraints to drag hands with mocaps -->
    <equality>
        <weld name="teleop_weld_left" body1="hand_left" body2="mocap_target_left"/>
        <weld name="teleop_weld_right" body1="hand_right" body2="mocap_target_right"/>
    </equality>

    <!-- Gripper position controllers -->
    <actuator>
        <position name="finger_actuator1_left" joint="finger_joint1_left" ctrlrange="0 0.04" kp="100"/>
        <position name="finger_actuator2_left" joint="finger_joint2_left" ctrlrange="0 0.04" kp="100"/>
        <position name="finger_actuator1_right" joint="finger_joint1_right" ctrlrange="0 0.04" kp="100"/>
        <position name="finger_actuator2_right" joint="finger_joint2_right" ctrlrange="0 0.04" kp="100"/>
    </actuator>
</mujoco>"""
    with open("franka_panda.xml", "w") as f:
        f.write(xml_content)
    print(f"[INFO] Generated franka_panda.xml with {num_cubes} randomized cubes.")

def generate_franka_panda_industrial_xml(num_cubes: int) -> None:
    """
    Checks if franka_emika_panda assets are downloaded; if not, fetches them.
    Parses panda.xml to dynamically construct a dual industrial arm simulation setup.
    """
    import urllib.request
    import xml.etree.ElementTree as ET
    import os
    import re
    
    # 1. Download official files if missing
    base_dir = "franka_emika_panda"
    xml_path = os.path.join(base_dir, "panda.xml")
    assets_dir = os.path.join(base_dir, "assets")
    
    if not os.path.exists(xml_path):
        print("[INFO] MuJoCo Menagerie Franka Panda model not found locally. Starting automatic download...")
        os.makedirs(assets_dir, exist_ok=True)
        base_url = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/franka_emika_panda/"
        
        try:
            # Download panda.xml
            print(f"[INFO] Downloading {base_url}panda.xml -> {xml_path}")
            urllib.request.urlretrieve(base_url + "panda.xml", xml_path)
            
            with open(xml_path, "r", encoding="utf-8") as f:
                xml_content = f.read()
                
            mesh_files = re.findall(r'file="([^"]+)"', xml_content)
            mesh_files = sorted(list(set(mesh_files)))
            
            print(f"[INFO] Downloading {len(mesh_files)} mesh files from Menagerie...")
            for idx, mesh in enumerate(mesh_files):
                mesh_url = base_url + "assets/" + mesh
                dest_path = os.path.join(assets_dir, mesh)
                print(f"[{idx+1}/{len(mesh_files)}] Downloading: {mesh}")
                urllib.request.urlretrieve(mesh_url, dest_path)
            print("[INFO] Model assets downloaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to download Franka Panda assets: {e}")
            raise e

    # 2. Parse and generate the dual model
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        link0_el = root.find(".//body[@name='link0']")
        if link0_el is None:
            raise ValueError("[ERROR] link0 body element not found in panda.xml")
            
        def copy_and_suffix(el, suffix):
            new_el = ET.Element(el.tag, el.attrib.copy())
            if "name" in new_el.attrib:
                new_el.attrib["name"] = new_el.attrib["name"] + suffix
            if "joint" in new_el.attrib:
                new_el.attrib["joint"] = new_el.attrib["joint"] + suffix
                
            for child in el:
                if child.tag == "body":
                    new_el.append(copy_and_suffix(child, suffix))
                elif child.tag in ["joint", "geom", "site"]:
                    new_child = ET.Element(child.tag, child.attrib.copy())
                    if "name" in new_child.attrib:
                        new_child.attrib["name"] = new_child.attrib["name"] + suffix
                    for gc in child:
                        new_child.append(copy_and_suffix(gc, suffix))
                    new_el.append(new_child)
                else:
                    new_child = ET.Element(child.tag, child.attrib.copy())
                    for gc in child:
                        new_child.append(copy_and_suffix(gc, suffix))
                    new_el.append(new_child)
            return new_el

        # Generate subtrees
        left_arm = copy_and_suffix(link0_el, "_left")
        left_arm.attrib["pos"] = "0 0.25 0.0"
        
        right_arm = copy_and_suffix(link0_el, "_right")
        right_arm.attrib["pos"] = "0 -0.25 0.0"
        
        default_el = root.find("default")
        if default_el is not None:
            finger_joint = default_el.find(".//default[@class='finger']/joint")
            if finger_joint is not None:
                finger_joint.attrib["limited"] = "true"
                finger_joint.attrib["damping"] = "0.05"
                finger_joint.attrib["armature"] = "0.001"
                finger_joint.attrib["solimplimit"] = "0.99 0.999 0.001"
                finger_joint.attrib["solreflimit"] = "0.001 1"
        default_str = ET.tostring(default_el, encoding="utf-8").decode("utf-8") if default_el is not None else ""
        
        asset_el = root.find("asset")
        asset_str = ET.tostring(asset_el, encoding="utf-8").decode("utf-8") if asset_el is not None else ""
        
        left_arm_str = ET.tostring(left_arm, encoding="utf-8").decode("utf-8")
        right_arm_str = ET.tostring(right_arm, encoding="utf-8").decode("utf-8")
        
        # Spawn cubes
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
        while len(positions) < num_cubes and attempts < 200:
            attempts += 1
            x = random.uniform(0.35, 0.55)
            y = random.uniform(-0.35, 0.35)
            too_close = False
            for px, py in positions:
                if np.linalg.norm([x - px, y - py]) < 0.065:
                    too_close = True
                    break
            if not too_close:
                positions.append((x, y))
                
        while len(positions) < num_cubes:
            i = len(positions)
            x = 0.35 + 0.05 * (i % 4)
            y = -0.25 + 0.1 * (i // 4)
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
        
        # Build the final XML
        xml_content = f"""<mujoco model="franka_panda_industrial_teleop">
    <compiler angle="radian" meshdir="franka_emika_panda/assets" autolimits="true"/>
    <option integrator="implicitfast" timestep="0.002"/>
    
    {default_str}
    {asset_str}
    
    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="true"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.15 0.15 0.15 1.0"/>

        <!-- Manipulation Workspace Table (Surface at Z = 0) -->
        <body name="table" pos="0.45 0.0 0.0">
            <geom name="table_top" type="box" size="0.25 0.50 0.01" pos="0 0 -0.01" rgba="0.25 0.25 0.25 1.0"/>
        </body>

        {cube_bodies_xml}

        {left_arm_str}
        {right_arm_str}

        <!-- Left Mocap Target -->
        <body name="mocap_target_left" mocap="true" pos="0.554 0.25 0.625">
            <geom name="mocap_geom_left" type="sphere" size="0.02" rgba="0 1 0 0.4" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0.06 0 0" rgba="1 0 0 0.8" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0 0.06 0" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0 0 0.06" rgba="0 0 1 0.8" contype="0" conaffinity="0"/>
        </body>

        <!-- Right Mocap Target -->
        <body name="mocap_target_right" mocap="true" pos="0.554 -0.25 0.625">
            <geom name="mocap_geom_right" type="sphere" size="0.02" rgba="0 1 0 0.4" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0.06 0 0" rgba="1 0 0 0.8" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0 0.06 0" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.002" fromto="0 0 0 0 0 0.06" rgba="0 0 1 0.8" contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <!-- Weld constraints to drag hands with mocaps -->
    <equality>
        <weld name="teleop_weld_left" body1="hand_left" body2="mocap_target_left" relpose="0 0 0 1 0 0 0" solimp="0.9 0.95 0.001" solref="0.02 1"/>
        <weld name="teleop_weld_right" body1="hand_right" body2="mocap_target_right" relpose="0 0 0 1 0 0 0" solimp="0.9 0.95 0.001" solref="0.02 1"/>
    </equality>

    <!-- Gripper position controllers mapping straight to sliding joints -->
    <actuator>
        <position name="finger_actuator1_left" joint="finger_joint1_left" ctrlrange="0 0.04" kp="100"/>
        <position name="finger_actuator2_left" joint="finger_joint2_left" ctrlrange="0 0.04" kp="100"/>
        <position name="finger_actuator1_right" joint="finger_joint1_right" ctrlrange="0 0.04" kp="100"/>
        <position name="finger_actuator2_right" joint="finger_joint2_right" ctrlrange="0 0.04" kp="100"/>
    </actuator>
</mujoco>"""
        
        with open("franka_panda_industrial.xml", "w", encoding="utf-8") as f:
            f.write(xml_content)
        print("[INFO] Generated franka_panda_industrial.xml with dual arms and cubes.")
    except Exception as e:
        print(f"[ERROR] Failed to compile XML tree for industrial model: {e}")
        raise e

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
    if ROBOT_NAME == "franka_panda_geometric":
        try:
            generate_franka_panda_xml(NUM_CUBES)
        except Exception as e:
            print(f"[ERROR] Failed to dynamically generate geometric Panda XML: {e}")
            sys.exit(1)
    elif ROBOT_NAME == "franka_panda_industrial":
        try:
            generate_franka_panda_industrial_xml(NUM_CUBES)
        except Exception as e:
            print(f"[ERROR] Failed to dynamically generate industrial Panda XML: {e}")
            sys.exit(1)

    # 1. Initialize Modules
    try:
        tracker = HandTracker(alpha=0.15)
        is_dual = "mocap_name_left" in config
        
        if is_dual:
            # Dual spatial transformers
            transformers = {
                "Left": SpatialTransformer(
                    workspace_center=config["workspace_center_left"],
                    scaling_factors=config["scaling_factors"],
                    ref_mediapipe=config["ref_mediapipe"],
                    bounds_x=config["bounds_x"],
                    bounds_y=config["bounds_y_left"],
                    bounds_z=config["bounds_z"]
                ),
                "Right": SpatialTransformer(
                    workspace_center=config["workspace_center_right"],
                    scaling_factors=config["scaling_factors"],
                    ref_mediapipe=config["ref_mediapipe"],
                    bounds_x=config["bounds_x"],
                    bounds_y=config["bounds_y_right"],
                    bounds_z=config["bounds_z"]
                )
            }
            sim = RobotSimulator(model_path=config["model_path"])
            if "mocap_quat" in config:
                mocap_left_id = sim._resolve_mocap(config["mocap_name_left"])
                mocap_right_id = sim._resolve_mocap(config["mocap_name_right"])
                sim.data.mocap_quat[mocap_left_id] = config["mocap_quat"]
                sim.data.mocap_quat[mocap_right_id] = config["mocap_quat"]
                
            actuator_ids = {"Left": [], "Right": []}
            for side in ["Left", "Right"]:
                key = f"gripper_actuators_{side.lower()}"
                if key in config and config[key]:
                    for act_name in config[key]:
                        act_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                        if act_id == -1:
                            print(f"[WARNING] Actuator '{act_name}' not found in model.")
                        else:
                            actuator_ids[side].append(act_id)
        else:
            # Single spatial transformer
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
                legacy_mocap_id = sim._resolve_mocap(config["mocap_name"])
                sim.data.mocap_quat[legacy_mocap_id] = config["mocap_quat"]
                
            actuator_ids = []
            if config["gripper_actuators"]:
                for act_name in config["gripper_actuators"]:
                    act_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                    if act_id == -1:
                        print(f"[WARNING] Actuator '{act_name}' not found in model.")
                    else:
                        actuator_ids.append(act_id)

        # Initialize joint states for the industrial Panda arms if selected
        if ROBOT_NAME == "franka_panda_industrial":
            home_val = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]
            left_joints = [
                "joint1_left", "joint2_left", "joint3_left", "joint4_left", 
                "joint5_left", "joint6_left", "joint7_left", "finger_joint1_left", "finger_joint2_left"
            ]
            right_joints = [
                "joint1_right", "joint2_right", "joint3_right", "joint4_right", 
                "joint5_right", "joint6_right", "joint7_right", "finger_joint1_right", "finger_joint2_right"
            ]
            for i, jname in enumerate(left_joints):
                jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid != -1:
                    qadr = sim.model.jnt_qposadr[jid]
                    sim.data.qpos[qadr] = home_val[i]
            for i, jname in enumerate(right_joints):
                jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid != -1:
                    qadr = sim.model.jnt_qposadr[jid]
                    sim.data.qpos[qadr] = home_val[i]
            mujoco.mj_forward(sim.model, sim.data)
                        
    except Exception as e:
        print(f"[CRITICAL] Initialization failed: {e}")
        sys.exit(1)

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
    if is_dual:
        hand_was_lost = {"Left": True, "Right": True}
        start_ee_pos = {"Left": None, "Right": None}
        blend_weight = {"Left": 1.0, "Right": 1.0}
    else:
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

                # Step 1: Run CPU perception to obtain hand landmark data dictionaries
                hands_data = tracker.process_frame(frame)

                if is_dual:
                    # Bimanual Control Path
                    for side in ["Left", "Right"]:
                        if side in hands_data:
                            hand_info = hands_data[side]
                            filtered_mp = hand_info["filtered_pos"]
                            pinch_dist = hand_info["pinch_dist"]
                            d_palm = hand_info["d_palm"]
                            
                            if hand_was_lost[side]:
                                ee_name = config["ee_name_left"] if side == "Left" else config["ee_name_right"]
                                start_ee_pos[side] = sim.get_ee_position(ee_name)
                                blend_weight[side] = 1.0
                                hand_was_lost[side] = False
                                
                            # Step 2: Transform coordinates
                            target_pos = transformers[side].transform(filtered_mp)
                            
                            # Apply bumpless transfer
                            if blend_weight[side] > 0.0:
                                target_pos = (1.0 - blend_weight[side]) * target_pos + blend_weight[side] * start_ee_pos[side]
                                blend_weight[side] = max(0.0, blend_weight[side] - 0.05)
                                
                            # Step 3: Write mapped target directly to mocap
                            mocap_name = config["mocap_name_left"] if side == "Left" else config["mocap_name_right"]
                            sim.set_mocap_position(mocap_name, target_pos)
                            
                            # Step 4: Proportional Gripper Control
                            if actuator_ids[side]:
                                gripper_ctrl = np.clip((pinch_dist - 0.05) / 0.10 * 0.04, 0.0, 0.04)
                                for act_id in actuator_ids[side]:
                                    sim.data.ctrl[act_id] = gripper_ctrl
                                    
                            ee_name = config["ee_name_left"] if side == "Left" else config["ee_name_right"]
                            ee_pos = sim.get_ee_position(ee_name)
                            hand_dist_cm = 9.0 / d_palm * 100.0 if d_palm > 0 else 100.0
                            
                            # Visual overlays per side
                            y_start = 80 if side == "Left" else 150
                            cv2.putText(
                                frame, f"{side} Tar: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}] m", 
                                (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            )
                            cv2.putText(
                                frame, f"{side} EE:  [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}] m", 
                                (20, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2
                            )
                            pinch_status = "CLOSED" if pinch_dist < 0.15 else "OPEN" if pinch_dist > 0.18 else f"PROP ({pinch_dist:.2f})"
                            cv2.putText(
                                frame, f"{side} Grip: {pinch_status} | Dist: {hand_dist_cm:.1f} cm", 
                                (20, y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                            )
                        else:
                            hand_was_lost[side] = True
                            y_start = 80 if side == "Left" else 150
                            cv2.putText(
                                frame, f"{side} hand lost - holding position", 
                                (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                            )
                else:
                    # Single-Arm Control Path
                    # Fetch whichever hand is currently visible
                    hand_info = None
                    for side in ["Right", "Left"]:
                        if side in hands_data:
                            hand_info = hands_data[side]
                            break

                    if hand_info is not None:
                        filtered_mp = hand_info["filtered_pos"]
                        pinch_dist = hand_info["pinch_dist"]
                        d_palm = hand_info["d_palm"]
                        
                        if hand_was_lost:
                            start_ee_pos = sim.get_ee_position(config["ee_name"])
                            blend_weight = 1.0
                            hand_was_lost = False

                        # Step 2: Transform coordinates
                        target_pos = transformer.transform(filtered_mp)

                        # Apply smooth initialization blending
                        if blend_weight > 0.0:
                            target_pos = (1.0 - blend_weight) * target_pos + blend_weight * start_ee_pos
                            blend_weight = max(0.0, blend_weight - 0.05)

                        # Step 3: Write mapped target directly to mocap
                        sim.set_mocap_position(config["mocap_name"], target_pos)

                        # Step 4: Proportional Gripper Control
                        if actuator_ids:
                            gripper_ctrl = np.clip((pinch_dist - 0.05) / 0.10 * 0.04, 0.0, 0.04)
                            for act_id in actuator_ids:
                                sim.data.ctrl[act_id] = gripper_ctrl

                        ee_pos = sim.get_ee_position(config["ee_name"])
                        hand_dist_cm = 9.0 / d_palm * 100.0 if d_palm > 0 else 100.0

                        cv2.putText(
                            frame, f"Target Pos: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}] m", 
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )
                        cv2.putText(
                            frame, f"EE Pos:     [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}] m", 
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                        )
                    else:
                        hand_was_lost = True
                        cv2.putText(
                            frame, "Hand lost - holding position", 
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                        )

                # Step 5: Step physics simulation to sync with real elapsed time
                elapsed = time.time() - start_time
                steps_needed = int((elapsed - (sim.data.time - sim_start_time)) / dt)
                if steps_needed > 0:
                    sim.step(steps=steps_needed)

                # Compute loop FPS
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
