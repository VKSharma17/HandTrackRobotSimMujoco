# Task Progress - Dual-Arm Bimanual Teleoperation

- `[x]` Module 1: Dual-Hand Perception Engine (`hand_tracker.py`)
  - `[x]` Configure tracker for up to two hands (`num_hands=2`)
  - `[x]` Implement separate filter states for left/right hands
  - `[x]` Implement mirrored handedness logic (MediaPipe -> Physical mapping)
  - `[x]` Reformat return signature of `process_frame()` to dictionary mapping
- `[x]` Module 3: Generic Physics Simulator (`robot_sim.py`)
  - `[x]` Support tracking multiple mocaps and end-effectors via list/dict caches
- `[x]` Module 4: Dual-Arm System Integrator (`main.py`)
  - `[x]` Update XML generator to construct dual Franka Panda arms
  - `[x]` Configure dual spatial transformers with independent offsets
  - `[x]` Set up separate bumpless transfer blending logic for both arms
  - `[x]` Implement synchronous control loops for left and right arm target tracking
  - `[x]` Update OpenCV text overlays to display metrics for both arms
- `[/]` Verification
  - `[ ]` Verify successful dual-arm tracking and independent hand gestures
