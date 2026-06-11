# Task Progress - Industrial Franka Panda Integration

- `[x]` Reorganize configurations in `main.py`
  - `[x]` Rename current `franka_panda` configuration to `franka_panda_geometric`
  - `[x]` Add new `franka_panda_industrial` configuration in `ROBOT_CONFIGS`
- `[x]` Implement Industrial Model Auto-Downloader & Generator
  - `[x]` Add `generate_franka_panda_industrial_xml()` in `main.py`
  - `[x]` Implement robust checks and automatic fetching of model XML & OBJ/STL meshes
  - `[x]` Parse and recursively duplicate `link0` base body with Left/Right suffixes
  - `[x]` Generate Left/Right mocap targets, weld constraints, table, cubes, and actuators
- `[x]` Configure Initial State & Blending
  - `[x]` Set model initial joint variables to home keyframe pose on load
  - `[x]` Align starting mocap coordinates with home pose end-effector values
- `[x]` Verification
  - `[x]` Verify execution of `franka_panda_geometric`
  - `[x]` Verify auto-downloader, compile, and runtime execution of `franka_panda_industrial`
