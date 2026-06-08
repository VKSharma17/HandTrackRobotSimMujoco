import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import Optional

class RobotSimulator:
    """
    Module 3: Physics Simulator
    Loads the robot MJCF model, manages the simulation state (mjData), 
    and handles the passive rendering window.
    """
    def __init__(self, model_path: str = "simple_arm.xml", mocap_name: str = "mocap_target", ee_name: str = "end_effector") -> None:
        """
        Initializes the MuJoCo simulation environment.

        Args:
            model_path: Path to the robot's MJCF XML model file.
            mocap_name: Name of the mocap body in the XML.
            ee_name: Name of the end-effector body in the XML.
        """
        print(f"[INFO] Loading MuJoCo model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.ee_name = ee_name

        # Retrieve the body ID and mocap index for the mocap body
        self.mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
        if self.mocap_body_id == -1:
            raise ValueError(f"[ERROR] Mocap target body '{mocap_name}' not found in XML.")
        
        self.mocap_id = self.model.body_mocapid[self.mocap_body_id]
        if self.mocap_id == -1:
            raise ValueError(f"[ERROR] '{mocap_name}' is not configured as a mocap body (mocap=true).")
            
        print(f"[INFO] MuJoCo model loaded successfully. Mocap ID: {self.mocap_id}")

    def get_mocap_position(self) -> np.ndarray:
        """Returns the current 3D position vector of the mocap target."""
        return self.data.mocap_pos[self.mocap_id].copy()

    def set_mocap_position(self, position: np.ndarray) -> None:
        """
        Updates the 3D position of the mocap target body in the simulation space.

        Args:
            position: 1D np.ndarray([x, y, z]) target coordinate in meters.
        """
        self.data.mocap_pos[self.mocap_id] = position

    def get_ee_position(self) -> np.ndarray:
        """Returns the current 3D position vector of the end-effector sphere."""
        ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_name)
        if ee_body_id == -1:
            raise ValueError(f"[ERROR] End-effector body '{self.ee_name}' not found in model.")
        return self.data.xpos[ee_body_id].copy()

    def step(self, steps: int = 1) -> None:
        """
        Steps the physics simulation forward.

        Args:
            steps: Number of integration steps to take.
        """
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)


if __name__ == "__main__":
    print("[INFO] Starting Module 3 Unit Test...")
    sim = RobotSimulator(model_path="simple_arm.xml")

    # Time parameters for trajectory generation
    dt = sim.model.opt.timestep  # Simulation time step (e.g., 0.002 seconds)
    print(f"[INFO] Simulation timestep configured to: {dt} seconds")

    # Launch passive viewer for real-time visualization on GPU
    print("[INFO] Launching passive viewer...")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Check if viewer is open
        if not viewer.is_running():
            print("[ERROR] Viewer failed to initialize.")
            exit(1)

        start_time = time.time()
        sim_start_time = sim.data.time
        print("[INFO] Trajectory active. Press Esc in the viewer window to close.")

        # Real-time synchronization loop
        while viewer.is_running():
            step_start = time.time()
            elapsed = step_start - start_time
            
            # Generate autonomous 3D sine-wave circle trajectory in workspace
            # Center: (0.45, 0.0, 0.40)
            # R_x = 0.12, R_y = 0.20, R_z = 0.10
            x_target = 0.45 + 0.12 * np.cos(1.5 * elapsed)
            y_target = 0.20 * np.sin(1.5 * elapsed)
            z_target = 0.40 + 0.10 * np.sin(3.0 * elapsed)
            target_pos = np.array([x_target, y_target, z_target], dtype=np.float64)

            # Update mocap body position
            sim.set_mocap_position(target_pos)

            # Step simulation physics (run enough steps to match real elapsed time)
            # Standard: sync frequency (60 Hz) means stepping multiple times per visual frame
            target_sim_time = elapsed
            steps_needed = int((target_sim_time - (sim.data.time - sim_start_time)) / dt)
            if steps_needed > 0:
                sim.step(steps=steps_needed)

            # Log tracking error between mocap target and end-effector positions
            ee_pos = sim.get_ee_position()
            tracking_error = np.linalg.norm(target_pos - ee_pos)
            print(
                f"Sim Time: {sim.data.time:.2f}s | "
                f"Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] | "
                f"EE Pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] | "
                f"Weld Error: {tracking_error*1000.0:.2f} mm",
                end="\r"
            )

            # Update the viewer window rendering
            viewer.sync()

            # Enforce 60 FPS loop rate (16.6 ms) for the Python control thread
            time_spent = time.time() - step_start
            time_to_sleep = (1.0 / 60.0) - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    print("\n[INFO] Simulator unit test completed successfully.")
