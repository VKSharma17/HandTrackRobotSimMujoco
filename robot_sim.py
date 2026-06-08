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
    def __init__(self, model_path: str = "simple_arm.xml", mocap_name: Optional[str] = None, ee_name: Optional[str] = None) -> None:
        """
        Initializes the MuJoCo simulation environment.

        Args:
            model_path: Path to the robot's MJCF XML model file.
            mocap_name: Optional legacy parameter for backward compatibility.
            ee_name: Optional legacy parameter for backward compatibility.
        """
        print(f"[INFO] Loading MuJoCo model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Cache dictionaries to store resolved body and mocap IDs on the fly
        self._body_ids = {}
        self._mocap_ids = {}

        # Resolve single-arm variables if legacy names are provided
        if mocap_name:
            self._resolve_mocap(mocap_name)
        if ee_name:
            self._resolve_body(ee_name)

    def _resolve_body(self, body_name: str) -> int:
        """Resolves body name to its internal MuJoCo body ID and caches it."""
        if body_name not in self._body_ids:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                raise ValueError(f"[ERROR] Body '{body_name}' not found in XML model.")
            self._body_ids[body_name] = body_id
        return self._body_ids[body_name]

    def _resolve_mocap(self, mocap_name: str) -> int:
        """Resolves mocap body name to its internal MuJoCo mocap ID and caches it."""
        if mocap_name not in self._mocap_ids:
            body_id = self._resolve_body(mocap_name)
            mocap_id = self.model.body_mocapid[body_id]
            if mocap_id == -1:
                raise ValueError(f"[ERROR] '{mocap_name}' is not configured as a mocap body (mocap=true).")
            self._mocap_ids[mocap_name] = mocap_id
        return self._mocap_ids[mocap_name]

    def get_mocap_position(self, mocap_name: str = "mocap_target") -> np.ndarray:
        """Returns the current 3D position vector of the specified mocap target."""
        mocap_id = self._resolve_mocap(mocap_name)
        return self.data.mocap_pos[mocap_id].copy()

    def set_mocap_position(self, mocap_name_or_pos, position: Optional[np.ndarray] = None) -> None:
        """
        Updates the 3D position of the specified mocap target body in the simulation space.

        Args:
            mocap_name_or_pos: Name of the mocap target (str), or position vector (legacy single-arm).
            position: 1D np.ndarray([x, y, z]) target coordinate in meters (when named).
        """
        if position is None:
            # Legacy single-arm call: set_mocap_position(position)
            mocap_name = "mocap_target"
            pos = mocap_name_or_pos
        else:
            mocap_name = mocap_name_or_pos
            pos = position

        mocap_id = self._resolve_mocap(mocap_name)
        self.data.mocap_pos[mocap_id] = pos

    def get_ee_position(self, ee_name: str = "end_effector") -> np.ndarray:
        """Returns the current 3D position vector of the specified end-effector body."""
        body_id = self._resolve_body(ee_name)
        return self.data.xpos[body_id].copy()

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
