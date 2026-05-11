import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import torch
import os

# Desactivar GPU si es necesario
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from robot_sdf import RobotSDF

def main():
    # 1. INICIALIZAR PYBULLET
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    urdf_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/urdf/ADAM/models/robot/rb1_base_description/robots/robotDummyOriginal.urdf"
    model_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/models/ADAM/robotDummyOriginal_bf_4_seg3_visual.pt"

    # --- ROBOT REAL ---
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.05], useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    
    revolute_joint_indices = []
    joint_names = []
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        if info[2] == p.JOINT_REVOLUTE:
            revolute_joint_indices.append(i)
            joint_names.append(info[1].decode('utf-8'))

    dof = len(revolute_joint_indices)
    left_arm_indices = [i for i, name in enumerate(joint_names) if "left_arm" in name]
    
    # --- POSE OBJETIVO ---
    pose_objetivo = np.zeros(dof)
    for idx in left_arm_indices:
        name = joint_names[idx]
        if "shoulder_pan" in name: pose_objetivo[idx] = 0.7
        if "shoulder_lift" in name: pose_objetivo[idx] = -1.1
        if "elbow" in name: pose_objetivo[idx] = 1.6
    
    # 2. CARGAR MODELO SDF
    robot_sdf = RobotSDF(urdf_path, mesh_type="visual", n_func=4, n_seg=3)
    robot_sdf.load(model_path)

    # 3. OBSTÁCULO MÓVIL
    obst_radius = 0.12
    base_obst_pos = np.array([0.45, 0.3, 1.1]) # Centro de la oscilación
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=obst_radius),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=obst_radius, rgbaColor=[1, 0, 0, 0.7]),
        basePosition=base_obst_pos
    )

    # 4. INDICADORES VISUALES
    # Objetivo (Azul)
    goal_visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[0, 0, 1, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_visual_id, basePosition=[0.5, 0.5, 1.2])

    # Punto de Evasión (Amarillo)
    evasion_point_body = p.createMultiBody(
        baseMass=0, 
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 1, 0, 1]), 
        basePosition=[0,0,0]
    )

    print("Simulación iniciada. La esfera ahora se mueve.")
    
    t = 0
    try:
        while True:
            # --- MOVIMIENTO DE LA ESFERA ---
            # Hacemos que la esfera oscile en el eje Y (de lado a lado)
            current_obst_pos = base_obst_pos.copy()
            current_obst_pos[1] = base_obst_pos[1] + math.sin(t * 1.5) * 0.25 # Oscilación de 25cm
            p.resetBasePositionAndOrientation(obstacle_id, current_obst_pos, [0,0,0,1])

            joint_states = p.getJointStates(robot_id, revolute_joint_indices)
            current_angles = np.array([state[0] for state in joint_states])
            
            # 1. Atracción al Goal
            v_att = 1.2 * (pose_objetivo - current_angles)

            # 2. Repulsión SDF
            theta_tensor = torch.tensor(current_angles, dtype=torch.float32, requires_grad=True)
            point_tensor = torch.tensor([current_obst_pos], dtype=torch.float32)
            
            sdf, _ = robot_sdf.query(point_tensor, theta_tensor)
            real_distance = sdf.item() - obst_radius
            
            v_rep = np.zeros(dof)
            if real_distance < 0.01: # Aumentamos un poco el margen para que se note el esquive
                sdf.backward() 
                # Subimos la ganancia a 0.4 para que el esquive sea más "valiente"
                v_rep = (0.4 / (real_distance + 0.01)) * theta_tensor.grad.numpy() 
                
                cp = p.getClosestPoints(robot_id, obstacle_id, distance=0.5)
                if len(cp) > 0:
                    p.resetBasePositionAndOrientation(evasion_point_body, cp[0][6], [0,0,0,1])
            else:
                p.resetBasePositionAndOrientation(evasion_point_body, [0,0,-1], [0,0,0,1])

            # 3. Mezcla y Control
            cmd_vel = np.clip(v_att + v_rep, -2.5, 2.5)
            p.setJointMotorControlArray(robot_id, revolute_joint_indices, p.VELOCITY_CONTROL, targetVelocities=cmd_vel.tolist())

            p.stepSimulation()
            time.sleep(1./240.)
            t += 1./240.

    except KeyboardInterrupt:
        p.disconnect()

if __name__ == '__main__':
    main()