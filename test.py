from robot_sdf import RobotSDF
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load or train the model
robot = RobotSDF("urdf/kuka_iiwa", mesh_type="visual", n_func=4, n_seg=3)
#robot.train()  # or robot.load("path/to/model.pt")
robot.load("/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/models/iiwa7/iiwa7_bf_4_seg3_visual.pt")

# Query SDF values and gradients
theta = torch.zeros(robot.dof)  # joint angles
points = torch.rand(1000, 3) * 2.0 - 1.0  # 3D points in world frame
sdf, grad = robot.query(points, theta)
print("Signed Distance:", sdf)
print("Gradient:", grad)

# sdf: signed distance (negative inside, positive outside)
# grad: gradient of SDF (for collision avoidance, etc.)

# Visualize at a specific configuration
robot.visualize(theta)