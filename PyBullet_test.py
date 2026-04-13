import pybullet as p
import pybullet_data
import time
import torch
import os

# Forzamos CPU para evitar problemas de VRAM (como hicimos antes)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from robot_sdf import RobotSDF

def main():
    # ---------------------------------------------------------
    # 1. INICIALIZAR PYBULLET Y EL ENTORNO
    # ---------------------------------------------------------
    physicsClient = p.connect(p.GUI) # Abre la ventana 3D
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    """ p.setAdditionalSearchPath("/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2")
    p.setAdditionalSearchPath("/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/urdf") """
    p.setGravity(0, 0, -9.81)
    
    # Cargamos el suelo
    plane = p.loadURDF("plane.urdf")

    # Cargamos tu KUKA iiwa
    urdf_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/urdf/kuka_iiwa/iiwa7.urdf"
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)

    # ---------------------------------------------------------
    # 2. CARGAR LA RED NEURONAL (SDF)
    # ---------------------------------------------------------
    print("Cargando modelo SDF...")
    robot_sdf = RobotSDF("urdf/kuka_iiwa", mesh_type="visual", n_func=4, n_seg=3)
    model_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/models/iiwa7/iiwa7_bf_4_seg3_visual.pt"
    robot_sdf.load(model_path)

    # ---------------------------------------------------------
    # 3. CREAR UN OBSTÁCULO VIRTUAL EN PYBULLET
    # ---------------------------------------------------------
    # Vamos a poner una pequeña esfera roja que simule un obstáculo
    obst_radius = 0.1
    obst_pos = [0.4, 0.0, 0.5] # X, Y, Z respecto a la base del robot
    
    col_sphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=obst_radius)
    vis_sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=obst_radius, rgbaColor=[1, 0, 0, 1])
    obstacle_id = p.createMultiBody(baseMass=0, # Masa 0 = Estático (no cae)
                                    baseCollisionShapeIndex=col_sphereId, 
                                    baseVisualShapeIndex=vis_sphereId, 
                                    basePosition=obst_pos)

    # ---------------------------------------------------------
    # 4. BUCLE DE CONTROL Y SIMULACIÓN
    # ---------------------------------------------------------
    print("Iniciando simulación. Presiona Ctrl+C en la terminal para salir.")
    
    # Definimos qué articulaciones son realmente motores (ignorando fixed joints si las hay)
    revolute_joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE]

    try:
        while True:
            # A. Leer la posición actual de los motores desde PyBullet
            joint_states = p.getJointStates(robot_id, revolute_joint_indices)
            current_angles = [state[0] for state in joint_states]
            
            # B. Convertir los datos para PyTorch
            # El modelo espera un tensor de 1D con los 7 ángulos
            theta_tensor = torch.tensor(current_angles, dtype=torch.float32)
            
            # El obstáculo es un solo punto en el espacio, pero el modelo espera un array de N puntos (N, 3)
            # Pasamos la posición del centro de la esfera.
            point_tensor = torch.tensor([obst_pos], dtype=torch.float32)
            
            # C. CONSULTAR LA RED NEURONAL
            sdf, grad = robot_sdf.query(point_tensor, theta_tensor)
            
            distance = sdf.item() # Distancia desde el robot al centro del obstáculo
            
            # Como el obstáculo tiene radio, la distancia real a la superficie es:
            real_distance = distance - obst_radius

            # D. LÓGICA DE EVASIÓN (Obstacle Avoidance)
            margen_seguridad = 0.15 # 15 centímetros
            
            if real_distance < margen_seguridad:
                # ¡PELIGRO! El robot está muy cerca.
                # 'grad' contiene el vector 3D de repulsión.
                gradient_vector = grad[0].numpy() # Convertir tensor a numpy array
                
                print(f"ALERTA: Obstáculo a {real_distance:.3f}m! Vector de escape: {gradient_vector}")
                
                # AQUI IRÍA TU CONTROLADOR CINEMÁTICO:
                # Usarías el Jacobiano del robot para convertir ese 'gradient_vector' (que está en XYZ)
                # en velocidades articulares (q_dot) para mover los motores alejándolos del obstáculo.
                
                # Por ahora, solo frenamos los motores para que no siga avanzando.
                p.setJointMotorControlArray(robot_id, revolute_joint_indices, 
                                            p.VELOCITY_CONTROL, 
                                            targetVelocities=[0]*len(revolute_joint_indices))
            else:
                # Seguro. El robot puede seguir su trayectoria normal.
                pass 

            # Avanzar un paso en el simulador
            p.stepSimulation()
            time.sleep(1./240.) # PyBullet funciona a 240Hz por defecto

    except KeyboardInterrupt:
        print("Simulación terminada.")
        p.disconnect()

if __name__ == '__main__':
    main()