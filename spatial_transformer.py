import numpy as np
from typing import Tuple

class SpatialTransformer:
    """
    Module 2: Kinematic Spatial Bridge
    Converts 3D landmarks from MediaPipe camera coordinates to MuJoCo world coordinates.
    Performs axis swapping, scaling, translation, and workspace boundary clipping.
    """
    def __init__(
        self,
        workspace_center: Tuple[float, float, float] = (0.45, 0.0, 0.40),
        scaling_factors: Tuple[float, float, float] = (0.8, 0.8, 0.8),
        ref_mediapipe: Tuple[float, float, float] = (0.5, 0.5, -0.1),
        bounds_x: Tuple[float, float] = (0.25, 0.65),
        bounds_y: Tuple[float, float] = (-0.35, 0.35),
        bounds_z: Tuple[float, float] = (0.15, 0.65)
    ) -> None:
        """
        Initializes the spatial bridge with workspace parameters and safety bounds.

        Args:
            workspace_center: Target (X, Y, Z) center of the robot workspace in MuJoCo (meters).
            scaling_factors: (S_x, S_y, S_z) scaling multipliers for each axis.
            ref_mediapipe: (X_ref, Y_ref, Z_ref) hand position corresponding to workspace center.
            bounds_x: (Min, Max) safety limits for MuJoCo X (forward/backward).
            bounds_y: (Min, Max) safety limits for MuJoCo Y (left/right).
            bounds_z: (Min, Max) safety limits for MuJoCo Z (up/down).
        """
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.bounds_z = bounds_z

        # Destructure parameters for matrix construction
        x_off, y_off, z_off = workspace_center
        s_x, s_y, s_z = scaling_factors
        x_ref, y_ref, z_ref = ref_mediapipe

        # Construct the 4x4 Homogeneous Transformation Matrix T
        # Maps [X_mp, Y_mp, Z_mp, 1]^T -> [X_mj, Y_mj, Z_mj, 1]^T
        # Row 1 (MuJoCo X / Forward-Backward): Maps from MediaPipe -Y (vertical)
        # Row 2 (MuJoCo Y / Left-Right): Maps from MediaPipe -X (horizontal)
        # Row 3 (MuJoCo Z / Up-Down): Maps from MediaPipe -Z (depth)
        self.T = np.array([
            [0.0, -s_x,  0.0, x_off + s_x * y_ref],
            [-s_y, 0.0,  0.0, y_off + s_y * x_ref],
            [0.0,  0.0, -s_z, z_off + s_z * z_ref],
            [0.0,  0.0,  0.0, 1.0]
        ], dtype=np.float64)

    def transform(self, mp_point: np.ndarray) -> np.ndarray:
        """
        Transforms a MediaPipe normalized 3D point to a MuJoCo 3D position vector in meters.

        Args:
            mp_point: A 1D array-like [x, y, z] from MediaPipe.

        Returns:
            A clipped 1D np.ndarray([x, y, z]) in MuJoCo coordinate space (meters).
        """
        # Create homogeneous coordinate vector [x, y, z, 1]^T
        p_hom = np.array([mp_point[0], mp_point[1], mp_point[2], 1.0], dtype=np.float64)
        
        # Apply transformation matrix multiplication
        p_transformed = self.T @ p_hom
        
        # Extract cartesian coordinates [X_mj, Y_mj, Z_mj]
        x_mj = p_transformed[0]
        y_mj = p_transformed[1]
        z_mj = p_transformed[2]

        # Apply safety bounding box clipping to prevent singularities and physical collisions
        x_clipped = np.clip(x_mj, self.bounds_x[0], self.bounds_x[1])
        y_clipped = np.clip(y_mj, self.bounds_y[0], self.bounds_y[1])
        z_clipped = np.clip(z_mj, self.bounds_z[0], self.bounds_z[1])

        return np.array([x_clipped, y_clipped, z_clipped], dtype=np.float64)


if __name__ == "__main__":
    print("[INFO] Starting Module 2 Unit Test...")
    transformer = SpatialTransformer()

    # Define test cases (MediaPipe X_mp, Y_mp, Z_mp)
    # MediaPipe: X (left->right 0->1), Y (top->bottom 0->1), Z (depth, closer is more negative, default ref is -0.1)
    test_cases = {
        "Center (Reference Point)": np.array([0.5, 0.5, -0.1]),
        "Move Hand Left": np.array([0.2, 0.5, -0.1]),
        "Move Hand Right": np.array([0.8, 0.5, -0.1]),
        "Move Hand Up (Forward)": np.array([0.5, 0.2, -0.1]),
        "Move Hand Down (Backward)": np.array([0.5, 0.8, -0.1]),
        "Move Hand Closer (Up)": np.array([0.5, 0.5, -0.3]),
        "Move Hand Further (Down)": np.array([0.5, 0.5, 0.1]),
        "Outside Safety Bound (Left Extreme)": np.array([-1.0, 0.5, -0.1]),
        "Outside Safety Bound (Below Ground)": np.array([0.5, 0.5, 0.5])
    }

    print("\nTransformation Verification:\n" + "="*60)
    for name, mp_coords in test_cases.items():
        mj_coords = transformer.transform(mp_coords)
        print(f"{name: <38} | MP: {mp_coords} -> MuJoCo: {mj_coords}")
    print("="*60)
    
    # Mathematical assertion checks
    # Center should transform exactly to workspace center (0.45, 0.0, 0.40)
    center_transform = transformer.transform(np.array([0.5, 0.5, -0.1]))
    assert np.allclose(center_transform, np.array([0.45, 0.0, 0.4])), "Center alignment failed!"
    print("[SUCCESS] Homogeneous transformation matrix assertions passed.")
