import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from robot_sdf import RobotSDF

def main():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf")

    urdf_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/urdf/kuka_iiwa/iiwa7.urdf"
    
    # 1. Cargamos los robots
    robot1_id = p.loadURDF(urdf_path, basePosition=[0, -0.6, 0], useFixedBase=True)
    orientacion_r2 = p.getQuaternionFromEuler([0, 0, 3.14159])
    robot2_id = p.loadURDF(urdf_path, basePosition=[0, 0.6, 0], baseOrientation=orientacion_r2, useFixedBase=True)
    
    num_joints = p.getNumJoints(robot1_id)
    revolute_joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot1_id, i)[2] == p.JOINT_REVOLUTE]
    ee_link_index = num_joints - 1 

    # --- AUMENTAR FRICCIÓN DE LAS "MANOS" ---
    # Es crucial para levantar objetos sin pinzas (agarrando por compresión)
    p.changeDynamics(robot1_id, ee_link_index, lateralFriction=2.0)
    p.changeDynamics(robot2_id, ee_link_index, lateralFriction=2.0)

    # 2. Cargar la red neuronal SDF
    print("Cargando modelo SDF...")
    robot_sdf = RobotSDF("urdf/kuka_iiwa", mesh_type="visual", n_func=4, n_seg=3)
    model_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/models/iiwa7/iiwa7_bf_4_seg3_visual.pt"
    robot_sdf.load(model_path)

    # 3. CREAR LA CAJA CON FÍSICA REAL
    box_half_extents = [0.15, 0.1, 0.1] 
    box_initial_pos = [0.0, 0.0, 0.1] 
    
    col_boxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
    vis_boxId = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half_extents, rgbaColor=[0.8, 0.6, 0.2, 1])
    
    # ¡Le damos masa (1.5 kg)! Ahora se caerá si no la sujetan bien.
    box_id = p.createMultiBody(baseMass=1.5, baseCollisionShapeIndex=col_boxId, 
                            baseVisualShapeIndex=vis_boxId, basePosition=box_initial_pos)
    # Alta fricción en la caja también
    p.changeDynamics(box_id, -1, lateralFriction=2.0)

    # Postura inicial neutral alta
    postura_inicial = [0.0, 0.5, 0.0, -1.5, 0.0, 0.5, 0.0]
    for i, idx in enumerate(revolute_joint_indices):
        p.resetJointState(robot1_id, idx, postura_inicial[i])
        p.resetJointState(robot2_id, idx, postura_inicial[i])

    print("Iniciando Misión de Elevación. Presiona Ctrl+C para salir.")
    
    def calcular_velocidades_hibridas(robot_id, current_angles, grasp_point, box_center):
        # IK (Atracción global a las manos)
        target_angles = p.calculateInverseKinematics(robot_id, ee_link_index, grasp_point)
        v_att = 2.5 * (np.array(target_angles[:7]) - current_angles) 

        # SDF (Repulsión local para los codos)
        theta_rep = torch.tensor(current_angles, dtype=torch.float32, requires_grad=True)
        pt_rep = torch.tensor([box_center], dtype=torch.float32)
        
        sdf_rep, _ = robot_sdf.query(pt_rep, theta_rep)
        dist_al_centro = sdf_rep.item()
        
        v_rep = np.zeros(7)
        margen = 0.25 # Margen de seguridad para no aplastar la caja
        
        if dist_al_centro < margen:
            sdf_rep.backward()
            grad_rep = theta_rep.grad.numpy()
            fuerza = 0.05 / (dist_al_centro + 0.001)
            v_rep = fuerza * grad_rep 

        cmd_vel = v_att + v_rep
        return np.clip(cmd_vel, -2.0, 2.0)

    # Variables para controlar el tiempo
    sim_time = 0
    dt = 1./240.

    try:
        while True:
            # --- MÁQUINA DE ESTADOS ---
            if sim_time < 3.0:
                # FASE 1: Acercamiento 
                target_z = 0.1 
                # La caja llega hasta 0.1. Nos quedamos a 0.20 (10 cm de margen de seguridad)
                target_y = 0.20 
            
            elif sim_time < 5.0:
                # FASE 2: Apriete (Compresión)
                target_z = 0.1
                # La caja está en 0.1. Le pedimos ir a 0.08. 
                # ¡Esos 2 centímetros de "penetración" son los que generan la fuerza para que no resbale!
                target_y = 0.08 
            
            else:
                # FASE 3: Elevación
                target_z = min(0.5, 0.1 + (sim_time - 5.0) * 0.1) 
                target_y = 0.08 # Mantenemos la fuerza de apriete mientras subimos

            # Asignamos los objetivos
            grasp_pt_1 = [0.0, -target_y, target_z] 
            grasp_pt_2 = [0.0,  target_y, target_z]

            # --- LECTURA DEL ENTORNO DINÁMICO ---
            # ¡La caja se mueve! Necesitamos saber dónde está ahora para el SDF
            caja_pos_actual, _ = p.getBasePositionAndOrientation(box_id)

            # Leer estado de los robots
            states_1 = p.getJointStates(robot1_id, revolute_joint_indices)
            angles_1 = np.array([s[0] for s in states_1])
            
            states_2 = p.getJointStates(robot2_id, revolute_joint_indices)
            angles_2 = np.array([s[0] for s in states_2])
            
            # Calcular velocidades
            cmd_1 = calcular_velocidades_hibridas(robot1_id, angles_1, grasp_pt_1, caja_pos_actual)
            cmd_2 = calcular_velocidades_hibridas(robot2_id, angles_2, grasp_pt_2, caja_pos_actual)
            
            # Aplicar motores
            p.setJointMotorControlArray(robot1_id, revolute_joint_indices, p.VELOCITY_CONTROL, targetVelocities=cmd_1.tolist())
            p.setJointMotorControlArray(robot2_id, revolute_joint_indices, p.VELOCITY_CONTROL, targetVelocities=cmd_2.tolist())

            p.stepSimulation()
            time.sleep(dt)
            sim_time += dt

    except KeyboardInterrupt:
        print("Simulación terminada.")
        p.disconnect()

if __name__ == '__main__':
    main()