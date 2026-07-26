from robot_sdf import RobotSDF
import torch
import os

# Set CUDA device to CPU for testing
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load or train the model
robot = RobotSDF("/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/urdf/InspireHand/inspire_hand_r.urdf", mesh_type="visual", n_func=4, n_seg=3)
#robot.train()  # or robot.load("path/to/model.pt")
robot.load("/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/models/InspireHand/inspire_hand_r_bf_4_seg3_visual.pt")

# Query SDF values and gradients
theta = torch.zeros(robot.dof)  # joint angles
points = torch.rand(1000, 3) * 2.0 - 1.0  # 3D points in world frame
sdf, grad = robot.query(points, theta)
print("Signed Distance:", sdf)
print("Gradient:", grad)

# sdf: signed distance (negative inside, positive outside)
# grad: gradient of SDF (for collision avoidance, etc.)

# Visualize at a specific configuration
#robot.visualize(theta,mesh_points=20000, nb_data=250, domain_min=(-0.5, -0.5, -0.5), domain_max=(0.5, 0.5, 0.5))

# 4. Visualización 3D
print("Generando visualización 3D...")
robot.visualize(theta,mesh_points=40000, nb_data=300, domain_min=(-0.5, -0.5, -0.5), domain_max=(0.5, 0.5, 0.5))
robot.visualize_lightweight(theta, num_points_per_link=2000)

# 5. Visualización 2D (Corte YZ)
print("Generando visualización 2D (Corte YZ)...")
robot.visualize_level_curves(
    theta=theta,
    mode="2d",
    plane="xz",
    intercept=0.0,
    bounds=(-0.20, 0.15, -0.1, 0.31),                             
    levels=[0.0, 0.02, 0.04, 0.06, 0.08], 
    num_points_per_link=200,              
    show_frames=False,                     
    show=True                             
)