import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from robot_sdf import RobotSDF

def main():
    # 1. INICIALIZAR PYBULLET
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf")

    urdf_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/urdf/kuka_iiwa/iiwa7.urdf"
    
    # --- ROBOT REAL ---
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    revolute_joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE]
    ee_link_index = num_joints - 1 # El último link es el efector final

    # --- ROBOT FANTASMA (Para la trayectoria planeada) ---
    ghost_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    # Hacer el fantasma transparente y quitarle las colisiones
    for j in range(-1, num_joints):
        p.changeVisualShape(ghost_id, j, rgbaColor=[0.5, 0.8, 0.5, 0.3]) # Verde transparente
        p.setCollisionFilterGroupMask(ghost_id, j, 0, 0) # No choca con nada

    # 2. CARGAR LA RED NEURONAL
    print("Cargando modelo SDF...")
    robot_sdf = RobotSDF("urdf/kuka_iiwa", mesh_type="visual", n_func=4, n_seg=3)
    model_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/models/iiwa7/iiwa7_bf_4_seg3_visual.pt"
    robot_sdf.load(model_path)

    # 3. CREAR OBSTÁCULO
    obst_radius = 0.12
    obst_pos = [0.4, 0.0, 0.5]
    
    col_sphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=obst_radius)
    vis_sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=obst_radius, rgbaColor=[1, 0, 0, 0.8])
    obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_sphereId, 
                                    baseVisualShapeIndex=vis_sphereId, basePosition=obst_pos)

    print("Iniciando evasión activa. Presiona Ctrl+C para salir.")
    
    t = 0
    prev_real_pos = None
    prev_ghost_pos = None

    try:
        while True:
            # --- A. LEER ESTADO DEL ROBOT REAL ---
            joint_states = p.getJointStates(robot_id, revolute_joint_indices)
            current_angles = np.array([state[0] for state in joint_states])
            
            # --- B. TRAYECTORIA PLANEADA (EL OBJETIVO) ---
            target_angles = np.zeros(7)
            target_angles[0] = math.sin(t) * 1.5
            target_angles[1] = 0.5
            target_angles[3] = -1.5
            
            # Actualizamos instantáneamente al robot fantasma a la pose ideal
            for i, idx in enumerate(revolute_joint_indices):
                p.resetJointState(ghost_id, idx, target_angles[i])

            Kp = 2.0
            v_att = Kp * (target_angles - current_angles)

            # --- C. FUERZA REPULSIVA (EL OBSTÁCULO) ---
            theta_tensor = torch.tensor(current_angles, dtype=torch.float32, requires_grad=True)
            point_tensor = torch.tensor([obst_pos], dtype=torch.float32)
            
            sdf, _ = robot_sdf.query(point_tensor, theta_tensor)
            real_distance = sdf.item() - obst_radius
            
            v_rep = np.zeros(7)
            margen_seguridad = 0.35 
            
            if real_distance < margen_seguridad:
                sdf.backward() 
                grad_theta = theta_tensor.grad.numpy() 
                
                fuerza = 0.05 / (real_distance + 0.01) 
                v_rep = fuerza * grad_theta
                
                # --- VISUALIZAR EL PUNTO DE EVASIÓN ---
                # Usamos PyBullet para encontrar el punto exacto más cercano entre el robot y la bola
                closest_points = p.getClosestPoints(robot_id, obstacle_id, distance=2.0)
                if len(closest_points) > 0:
                    punto_en_robot = closest_points[0][6] # Coordenada 3D en la carcasa del robot
                    punto_en_bola = closest_points[0][5]  # Coordenada 3D en la bola
                    
                    # Dibujar una línea roja gruesa (Vector de repulsión) que dura 0.1 segundos
                    p.addUserDebugLine(punto_en_robot, punto_en_bola, lineColorRGB=[1, 0, 0], lineWidth=5, lifeTime=0.1)

            # --- D. COMBINAR Y MOVER ---
            cmd_vel = v_att + v_rep
            cmd_vel = np.clip(cmd_vel, -2.0, 2.0)
            
            p.setJointMotorControlArray(robot_id, revolute_joint_indices, 
                                        p.VELOCITY_CONTROL, 
                                        targetVelocities=cmd_vel.tolist())

            # --- E. TRAZAR LAS LÍNEAS DE TRAYECTORIA EN EL AIRE ---
            real_ee_state = p.getLinkState(robot_id, ee_link_index)
            ghost_ee_state = p.getLinkState(ghost_id, ee_link_index)
            
            real_pos = real_ee_state[0]
            ghost_pos = ghost_ee_state[0]
            
            if prev_real_pos is not None:
                # Línea Azul = Trayectoria Real del Robot (duración 8 segundos para no saturar memoria)
                p.addUserDebugLine(prev_real_pos, real_pos, lineColorRGB=[0, 0.3, 1], lineWidth=3, lifeTime=8.0)
                # Línea Verde = Trayectoria Planeada Ideal (duración 8 segundos)
                p.addUserDebugLine(prev_ghost_pos, ghost_pos, lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=8.0)

            prev_real_pos = real_pos
            prev_ghost_pos = ghost_pos

            # --- AVANZAR SIMULACIÓN ---
            p.stepSimulation()
            time.sleep(1./240.)
            t += 1./240.

    except KeyboardInterrupt:
        print("Simulación terminada.")
        p.disconnect()

if __name__ == '__main__':
    main()