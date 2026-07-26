import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import os

# Desactivar GPU si solo quieres inferencia rápida en CPU para PyBullet
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from robot_sdf import RobotSDF

def main():
    # ==========================================
    # 1. INICIALIZAR PYBULLET Y ENTORNO
    # ==========================================
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Cargar suelo y mesa
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=[0.8, 0, 0])

    # ==========================================
    # 2. CREAR LAS LATAS SOBRE LA MESA
    # ==========================================
    z_mesa = 0.65

    # [radio, altura, [x, y], color_rgba]
    latas_info = [
        {"nombre": "Refresco Cola",  "radius": 0.033, "length": 0.115, "pos": [0.65,  0.10], "color": [0.8, 0.1, 0.1, 1.0]}, 
        {"nombre": "Bebida Energía", "radius": 0.028, "length": 0.160, "pos": [0.75, -0.15], "color": [0.1, 0.8, 0.1, 1.0]}, 
        {"nombre": "Lata Atún",      "radius": 0.045, "length": 0.040, "pos": [0.60, -0.30], "color": [0.1, 0.3, 0.8, 1.0]}, 
        {"nombre": "Lata Conserva",  "radius": 0.038, "length": 0.110, "pos": [0.80,  0.30], "color": [0.6, 0.6, 0.6, 1.0]}, 
        {"nombre": "Zumo Pequeño",   "radius": 0.025, "length": 0.090, "pos": [0.55,  0.25], "color": [0.9, 0.8, 0.1, 1.0]}  
    ]

    objetos_pos = []
    radios_objetos = []

    for lata in latas_info:
        r = lata["radius"]
        l = lata["length"]
        p_pos = [lata["pos"][0], lata["pos"][1], z_mesa + (l / 2)]
        c = lata["color"]

        col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=l)
        vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=l, rgbaColor=c)
        
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, 
                        baseVisualShapeIndex=vis_id, basePosition=p_pos)
        
        objetos_pos.append(p_pos)
        radios_objetos.append(r)

    # Convertimos las listas de objetos a arrays de numpy para la evaluación punto a punto
    objetos_np = np.array(objetos_pos)
    radios_np = np.array(radios_objetos)

    # ==========================================
    # 3. CARGAR EL ROBOT ADAM
    # ==========================================
    urdf_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/urdf/ADAM/models/robot/rb1_base_description/robots/robotDummyOriginal.urdf"
    robotId = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

    postura_agarre = {
        5: 0.0, 6: -0.5, 7: 1.5, 8: -1.0, 9: -1.57, 10: 0.0,     # Brazo Derecho
        23: 3.1416, 24: -0.5, 25: 1.5, 26: -1.0, 27: -1.57, 28: 0.0 # Brazo Izquierdo
    }

    for joint_idx, angle in postura_agarre.items():
        p.resetJointState(robotId, joint_idx, angle)

    # ==========================================
    # 4. INICIALIZAR EL MODELO SDF
    # ==========================================
    print("Cargando modelo SDF...")
    model_path = "/home/adrian/Escritorio/ImitationLearning/SignedDistanceFields/RDF_v2/models/ADAM/robotDummyOriginal_bf_4_seg3_visual.pt"
    robot_sdf = RobotSDF(urdf_path, mesh_type="visual", n_func=4, n_seg=3)
    robot_sdf.load(model_path)

    points_id = None
    margen_seguridad = 0.12 

    print("Iniciando simulación. Presiona Ctrl+C en la terminal para salir.")

    # ==========================================
    # 5. BUCLE PRINCIPAL DE SIMULACIÓN
    # ==========================================
    try:
        while True:
            # --- A. LEER ESTADO DEL ROBOT ---
            num_joints = p.getNumJoints(robotId)
            theta_np = np.zeros(robot_sdf.dof)
            
            mobi_joint_idx = 0
            for i in range(num_joints):
                info = p.getJointInfo(robotId, i)
                if info[2] != p.JOINT_FIXED:
                    state = p.getJointState(robotId, i)
                    if mobi_joint_idx < robot_sdf.dof:
                        theta_np[mobi_joint_idx] = state[0]
                    mobi_joint_idx += 1
                    
            theta_tensor = torch.tensor(theta_np, dtype=torch.float32)

            # --- B. CONSULTAR SDF GENERAL ---
            point_tensor = torch.tensor(objetos_pos, dtype=torch.float32)
            with torch.no_grad():
                sdf_vals, _ = robot_sdf.query(point_tensor, theta_tensor)

            # --- C. EXTRAER PUNTOS Y AISLAR BRAZOS ---
            with torch.no_grad():
                surface_points_raw = robot_sdf.get_surface_points(theta_tensor, num_points_per_link=60, safety_margin=0.03)
            
            puntos_raw_np = np.array(surface_points_raw)
            
            if len(puntos_raw_np) > 0:
                # FILTRO ESPACIAL: Aislar solo los brazos
                mask_brazos = (puntos_raw_np[:, 2] > 0.4) & (puntos_raw_np[:, 0] > 0.05)
                puntos_np = puntos_raw_np[mask_brazos]
            else:
                puntos_np = np.array([])

            # --- D. LÓGICA DE COLORES PUNTO A PUNTO ---
            if len(puntos_np) > 0:
                # 1. Calculamos distancia Euclídea masiva de cada punto a todas las latas (Broadcasting)
                diff = puntos_np[:, np.newaxis, :] - objetos_np[np.newaxis, :, :] # (Num_Puntos, 5_latas, 3_coord)
                distancias_centros = np.linalg.norm(diff, axis=2) 
                
                # 2. Restamos el radio a cada lata para tener la distancia a la superficie
                distancias_superficies = distancias_centros - radios_np
                
                # 3. Nos quedamos con la distancia mínima (la lata más cercana para cada punto)
                distancias_minimas_por_punto = np.min(distancias_superficies, axis=1)

                # 4. Asignamos colores
                # Creamos un array de colores rellenado de Azul cian por defecto
                colores_np = np.zeros((len(puntos_np), 3))
                colores_np[:] = [0.1, 0.6, 1.0] # Zona segura
                
                # Encontramos qué puntos exactos han violado el margen de seguridad
                mask_riesgo = distancias_minimas_por_punto < margen_seguridad
                
                # Solo a esos puntos les asignamos el color Naranja
                colores_np[mask_riesgo] = [1.0, 0.6, 0.0]

                # Convertimos de vuelta a lista para PyBullet
                surface_points = puntos_np.tolist()
                point_colors = colores_np.tolist()

                if points_id is None:
                    points_id = p.addUserDebugPoints(
                        pointPositions=surface_points,
                        pointColorsRGB=point_colors,
                        pointSize=4
                    )
                else:
                    points_id = p.addUserDebugPoints(
                        pointPositions=surface_points,
                        pointColorsRGB=point_colors,
                        pointSize=4,
                        replaceItemUniqueId=points_id
                    )

            # Avanzar simulación
            p.stepSimulation()
            time.sleep(1./240.)

    except KeyboardInterrupt:
        p.disconnect()
        print("Simulación terminada.")

if __name__ == '__main__':
    main()