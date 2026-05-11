Robot SDF
=========

Minimal pipeline to fit a Bernstein-polynomial SDF per link from a URDF and query/visualize the full robot SDF.

Requirements
------------
- Python with: `torch`, `pytorch_kinematics`, `trimesh`, `mesh_to_sdf`, `scikit-image`, `pycollada`

Generating Robot SDF for a New URDF
------------------------------------

To generate the robot SDF for a new URDF file:

1. **Prepare your URDF**: Place your URDF file (e.g., `my_robot.urdf`) and associated mesh files in a directory. The directory structure should match standard URDF conventions, with meshes in subdirectories like `meshes/visual/` or `meshes/collision/`.

2. **Run the training command**:
   ```bash
   python -m robot_sdf.run --urdf /path/to/your/urdf/directory --mesh-type visual --n-func 4 --n-seg 3
   ```
   - `--urdf`: Path to the URDF file or directory containing the URDF.
   - `--mesh-type`: Choose `visual` for visual meshes or `collision` for collision meshes.
   - `--n-func`: Number of Bernstein basis functions (higher = more accurate but slower).
   - `--n-seg`: Number of segments for the basis (higher = more accurate).
   - `robot_sdf.run`: Is used for general robots, meanwhile `robotArms_sdf.run` is a specific script for robotics arms.

   This command will:
   - Parse the URDF and load link meshes.
   - Sample SDF points per link (cached in `data/<robot_name>/`).
   - Fit Bernstein SDFs per link and save models in `models/<robot_name>/`.
   - Visualize the robot SDF at a midpoint joint configuration.

3. **Precompute cache only** (optional):
   If you want to generate only the SDF point samples without training:
   ```bash
   python -m robot_sdf.sample_sdf_points --urdf /path/to/your/urdf --mesh-type visual
   ```

Quick Start (CLI)
-----------------
From the repo root:

```bash
python -m robot_sdf.run --urdf urdf/panda_urdf --mesh-type visual --n-func 4 --n-seg 3
```

What this does:
- Parses the URDF and loads the link meshes (`visual` or `collision`).
- Samples SDF points per link (cached under `<urdf_dir>/data/sdf_points`).
- Fits Bernstein SDFs per link and saves the model under `<urdf_dir>/models`.
- Visualizes the full robot SDF as a mesh.

Generate Cache Only
-------------------
If you want to precompute only the cached SDF samples:

```bash
python -m robot_sdf.sample_sdf_points --urdf urdf/panda_urdf --mesh-type visual
```

Using the Robot SDF
--------------------

After training, you can use the SDF in your code:

```python
from robot_sdf import RobotSDF
import torch

# Load or train the model
robot = RobotSDF("urdf/panda_urdf", mesh_type="visual", n_func=4, n_seg=3)
robot.train()  # or robot.load("path/to/model.pt")

# Query SDF values and gradients
theta = torch.zeros(robot.dof)  # joint angles
points = torch.rand(1000, 3) * 2.0 - 1.0  # 3D points in world frame
sdf, grad = robot.query(points, theta)

# sdf: signed distance (negative inside, positive outside)
# grad: gradient of SDF (for collision avoidance, etc.)

# Visualize at a specific configuration
robot.visualize(theta)
```

Python API
----------
```python
from robot_sdf import RobotSDF
import torch

robot = RobotSDF("urdf/panda_urdf", mesh_type="visual", n_func=4, n_seg=3)
robot.train()

theta = torch.zeros(robot.dof)
points = torch.rand(1000, 3) * 2.0 - 1.0
sdf, grad = robot.query(points, theta)

robot.visualize(theta)
```

Notes
-----
- `--urdf` accepts either a URDF file or a directory containing a single URDF.
- Cache files: `<repo_root>/data/<robot_name>/<mesh_type>_<link_name>.npy`
- Model files: `<repo_root>/models/<robot_name>/<urdf_name>_bf_<n_func>_seg<n_seg>_<mesh_type>.pt`
- For other robots, just pass a different URDF path.
