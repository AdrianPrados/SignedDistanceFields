""" from robot_sdf import RobotSDF
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load or train the model
robot = RobotSDF("/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/urdf/ADAM/models/robot/rb1_base_description/robots/robotDummyOriginal.urdf", mesh_type="visual", n_func=4, n_seg=3)
#robot.train()  # or robot.load("path/to/model.pt")
robot.load("/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/models/ADAM/robotDummyOriginal_bf_4_seg3_visual.pt")

# Query SDF values and gradients
theta = torch.zeros(robot.dof)  # joint angles
points = torch.rand(1000, 3) * 2.0 - 1.0  # 3D points in world frame
sdf, grad = robot.query(points, theta)
print("Signed Distance:", sdf)
print("Gradient:", grad)

# sdf: signed distance (negative inside, positive outside)
# grad: gradient of SDF (for collision avoidance, etc.)

# Visualize at a specific configuration
#robot.visualize(theta,mesh_points=40000, nb_data=150)
robot.visualize_lightweight(theta,num_points_per_link=700)
robot.visualize_level_curves(
    theta=theta,
    mode="2d",
    plane="yz",                             
    levels=[0.0, 0.02, 0.04, 0.06, 0.08], 
    num_points_per_link=200,              
    show_frames=False,                     
    show=True                             
) """
from robot_sdf import RobotSDF
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load the model
urdf_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/urdf/ADAM/models/robot/rb1_base_description/robots/robotDummyOriginal.urdf"
model_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/models/ADAM/robotDummyOriginal_bf_4_seg3_visual.pt"

robot = RobotSDF(urdf_path, mesh_type="visual", n_func=4, n_seg=3)
robot.load(model_path)

# Inicializamos todos los joints a cero
theta = torch.zeros(robot.dof)

# --- BRAZO DERECHO (UR3) ---
# Índices del 5 al 10
theta[5] = 0.0      # rb1_right_arm_shoulder_pan_joint
theta[6] = -1.57    # rb1_right_arm_shoulder_lift_joint 
theta[7] = 1.57     # rb1_right_arm_elbow_joint 
theta[8] = -1.57    # rb1_right_arm_wrist_1_joint 
theta[9] = -1.57    # rb1_right_arm_wrist_2_joint
theta[10] = 0.0     # rb1_right_arm_wrist_3_joint

# --- BRAZO IZQUIERDO (UR3) ---
# Índices del 23 al 28
# Añadimos 3.1416 radianes (180 grados) para compensar el montaje y que apunte igual que el derecho
theta[23] = 3.1416  # rb1_left_arm_shoulder_pan_joint 
theta[24] = -1.57   # rb1_left_arm_shoulder_lift_joint 
theta[25] = -1.57    # rb1_left_arm_elbow_joint 
theta[26] = -3.1416   # rb1_left_arm_wrist_1_joint 
theta[27] = -1.57   # rb1_left_arm_wrist_2_joint
theta[28] = 0.0     # rb1_left_arm_wrist_3_joint

# (Opcional) Consultar valores SDF en la nueva postura
points = torch.rand(1000, 3) * 2.0 - 1.0  
sdf, grad = robot.query(points, theta)

# 4. Visualización 3D
print("Generando visualización 3D...")
robot.visualize(theta, mesh_points=50000, nb_data=150)
robot.visualize_lightweight(theta, num_points_per_link=700)

# 5. Visualización 2D (Corte YZ)
print("Generando visualización 2D (Corte YZ)...")
robot.visualize_level_curves(
    theta=theta,
    mode="2d",
    plane="yz",                             
    levels=[0.0, 0.02, 0.04, 0.06, 0.08], 
    num_points_per_link=200,              
    show_frames=False,                     
    show=True                             
)