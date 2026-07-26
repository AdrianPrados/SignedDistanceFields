# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# General porpose robot SDF module for PyTorch, using Bernstein basis functions and mesh sampling.
# It can be applied to robot with multiple end-effectors (humanoids, mobile manipulators, etc.) by simply providing the URDF and using the same query interface. 
# It can also be used for non-robotic articulated objects like hands, animals, or even plants, as long as they can be described by a URDF and have meshes. 
# The key is that the method learns a separate SDF for each link, so it can handle any structure that can be represented in that way.
# -----------------------------------------------------------------------------

import glob
import os
import shutil
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:
    import trimesh
except ModuleNotFoundError as _e:  # pragma: no cover - optional dependency
    trimesh = None
    _TRIMESH_IMPORT_ERROR = _e

try:
    import mesh_to_sdf
except ModuleNotFoundError as _e:  # pragma: no cover - optional dependency
    mesh_to_sdf = None
    _MESH_TO_SDF_IMPORT_ERROR = _e

try:
    from skimage import measure
except ModuleNotFoundError:
    measure = None

try:
    import pytorch_kinematics as pk
except ModuleNotFoundError as _e:  # pragma: no cover - optional dependency
    pk = None
    _PK_IMPORT_ERROR = _e

DEFAULT_SCAN_KWARGS = {
    "sign_method": "normal",
    "scan_count": 100,
    "scan_resolution": 400,
    "sample_point_count": 10_000_000,
    "normal_sample_count": 100,
    "min_size": 0.015,
}
DEFAULT_MESH_SDF_KWARGS = {
    "sign_method": "normal",
    "scan_count": 100,
    "scan_resolution": 400,
    "sample_point_count": 10_000_000,
    "normal_sample_count": 100,
}


class BernsteinBasis:
    def __init__(self, n_func: int, n_seg: int, domain_min: float, domain_max: float, device: str):
        self.n_func = n_func
        self.n_seg = n_seg
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device
        self.B = self._compute_B()
        self.C = self._compute_C()
        self.i = torch.arange(self.n_func, device=self.device)
        self.segment_size = 1.0 / float(self.n_seg)
        self.n_basis = int(self.C.size(-1))

    def _binomial(self, n: int, k: int) -> torch.Tensor:
        n = torch.tensor(n, dtype=torch.float64, device=self.device)
        k = torch.tensor(k, dtype=torch.float64, device=self.device)
        mask = (n >= 0) & (k >= 0)
        return torch.where(
            mask,
            torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)),
            torch.tensor(0.0, dtype=torch.float64, device=self.device),
        )

    def _compute_B(self) -> torch.Tensor:
        B0 = torch.zeros([self.n_func, self.n_func], dtype=torch.float32, device=self.device)
        for n in range(1, self.n_func + 1):
            for i in range(1, self.n_func + 1):
                B0[self.n_func - i, n - 1] = (-1) ** (self.n_func - i - n) * -self._binomial(
                    self.n_func - 1, i - 1
                ) * self._binomial((self.n_func - 1) - (i - 1), (self.n_func - 1) - (n - 1) - (i - 1))
        return torch.kron(torch.eye(self.n_seg, dtype=torch.float32, device=self.device), B0)

    def _compute_C(self) -> torch.Tensor:
        if self.n_func > 4:
            C0 = torch.zeros([4 + self.n_func - 4, 2 + self.n_func - 4], dtype=torch.float32)
            C0[: self.n_func - 4, : self.n_func - 4] = torch.eye(self.n_func - 4, dtype=torch.float32)
            C0[self.n_func - 4 :, self.n_func - 4 :] = torch.tensor(
                [[1, 0, 0, -1], [0, 1, 1, 2]], dtype=torch.float32
            ).T
        else:
            C0 = torch.tensor([[1, 0, 0, -1], [0, 1, 1, 2]], dtype=torch.float32).T

        C = torch.zeros(
            [2 + (self.n_seg - 1) * C0.shape[0] + self.n_func - 2, 2 + (self.n_seg - 1) * C0.shape[1] + self.n_func - 2],
            dtype=torch.float32,
            device=self.device,
        )
        C[:2, :2] = torch.eye(2, dtype=torch.float32)
        r = 2
        c = 2
        for _ in range(self.n_seg - 1):
            C[r : r + C0.shape[0], c : c + C0.shape[1]] = C0
            r += C0.shape[0]
            c += C0.shape[1]
        C[r:, c:] = torch.eye(self.n_func - 2, dtype=torch.float32)
        return C.to(self.device)

    def _compute_T_1d(
        self, t: torch.Tensor, use_derivative: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_pts = len(t)
        t = torch.clamp(t, min=1e-4, max=1 - 1e-4)
        T = torch.zeros((n_pts, self.n_func * self.n_seg), device=self.device)
        dT = torch.zeros_like(T) if use_derivative else None

        idx = torch.floor(t / self.segment_size).long()
        t = (t - idx * self.segment_size) / self.segment_size
        T_seg = t.unsqueeze(-1) ** self.i
        idx_expand = idx.unsqueeze(-1).expand(-1, self.n_func) * self.n_func + self.i.unsqueeze(0).expand(n_pts, -1)
        T.scatter_(1, idx_expand, T_seg)

        if use_derivative:
            dT_seg = torch.zeros_like(T_seg)
            dT_seg[:, 1:] = self.i[1:] * T_seg[:, : self.n_func - 1] * self.n_seg
            dT.scatter_(1, idx_expand, dT_seg)
        return T, dT

    def build_basis_function_from_points(
        self, p: torch.Tensor, use_derivative: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_pts = len(p)
        p = ((p - self.domain_min) / (self.domain_max - self.domain_min)).reshape(-1)
        T, dT = self._compute_T_1d(p, use_derivative=use_derivative)
        phi_1d = torch.matmul(torch.matmul(T, self.B), self.C)
        phi = phi_1d.reshape(n_pts, 3, self.n_basis)
        phi_x = phi[:, 0, :]
        phi_y = phi[:, 1, :]
        phi_z = phi[:, 2, :]
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).view(-1, self.n_basis**2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).view(-1, self.n_basis**3)
        if not use_derivative:
            return phi_xyz, None
        d_phi_1d = torch.matmul(torch.matmul(dT, self.B), self.C)
        d_phi = d_phi_1d.reshape(n_pts, 3, self.n_basis)
        d_phi_x_1d = d_phi[:, 0, :]
        d_phi_y_1d = d_phi[:, 1, :]
        d_phi_z_1d = d_phi[:, 2, :]
        d_phi_x = torch.einsum(
            "ij,ik->ijk", torch.einsum("ij,ik->ijk", d_phi_x_1d, phi_y).view(-1, self.n_basis**2), phi_z
        ).view(-1, self.n_basis**3)
        d_phi_y = torch.einsum(
            "ij,ik->ijk", torch.einsum("ij,ik->ijk", phi_x, d_phi_y_1d).view(-1, self.n_basis**2), phi_z
        ).view(-1, self.n_basis**3)
        d_phi_z = torch.einsum("ij,ik->ijk", phi_xy, d_phi_z_1d).view(-1, self.n_basis**3)
        d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1), d_phi_y.unsqueeze(-1), d_phi_z.unsqueeze(-1)), dim=-1)
        return phi_xyz, d_phi_xyz

    def bound_points_to_unit_sphere(
        self, points: torch.Tensor, eps: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        radius = float(min(abs(self.domain_min), abs(self.domain_max)))
        if radius <= 0.0:
            raise ValueError("Domain radius must be positive to bound points.")
        norm = torch.linalg.norm(points, dim=-1, keepdim=True)
        max_norm = radius - eps
        scale = max_norm / (norm + 1e-12)
        bounded = torch.where(norm > max_norm, points * scale, points)
        residual = points - bounded
        return bounded, residual


class RobotSDF:
    def __init__(
        self,
        urdf_path: str,
        mesh_type: str = "visual",
        device: Optional[str] = None,
        n_func: int = 8,
        n_seg: int = 1,
        cache_dir: Optional[str] = None,
    ) -> None:
        if pk is None:
            raise ModuleNotFoundError(
                "Missing dependency 'pytorch_kinematics'. Install it to use RobotSDF."
            ) from _PK_IMPORT_ERROR
        if trimesh is None:
            raise ModuleNotFoundError("Missing dependency 'trimesh'.") from _TRIMESH_IMPORT_ERROR
        if mesh_to_sdf is None:
            raise ModuleNotFoundError("Missing dependency 'mesh_to_sdf'.") from _MESH_TO_SDF_IMPORT_ERROR

        self.urdf_path = self._resolve_urdf_path(urdf_path)
        self.urdf_dir = os.path.dirname(os.path.abspath(self.urdf_path))
        self.robot_name = self._infer_robot_name()
        self.root_dir = self._infer_root_dir()
        self.mesh_type = mesh_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.basis = BernsteinBasis(n_func, n_seg, domain_min=-1.0, domain_max=1.0, device=self.device)
        self.cache_dir = cache_dir or os.path.join(self.root_dir, "data", self.robot_name)
        self.cache_prefix = mesh_type
        self._maybe_migrate_cache()
        self.model_dir = os.path.join(self.root_dir, "models", self.robot_name)
        self._maybe_migrate_models()

        self.link_meshes = self._load_link_meshes()
        self.joint_tree = self._build_joint_tree()
        self.link_names = sorted(self.link_meshes.keys())
        self.chain = pk.build_chain_from_urdf(open(self.urdf_path).read().encode()).to(dtype=torch.float32, device=self.device)
        
        #end_link = self._guess_end_link()
        """ self.chain = pk.build_serial_chain_from_urdf(
            open(self.urdf_path).read().encode(), end_link
        ).to(dtype=torch.float32, device=self.device) """
        self.dof = len(self.chain.get_joint_parameter_names())

        self.model: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    @staticmethod
    def _resolve_urdf_path(urdf_path: str) -> str:
        if os.path.isfile(urdf_path):
            return urdf_path
        if os.path.isdir(urdf_path):
            candidates = sorted(glob.glob(os.path.join(urdf_path, "*.urdf")))
            if not candidates:
                raise FileNotFoundError(f"No URDF files found in directory: {urdf_path}")
            if len(candidates) == 1:
                return candidates[0]
            base = os.path.basename(os.path.normpath(urdf_path))
            preferred = os.path.join(urdf_path, f"{base}.urdf")
            if preferred in candidates:
                return preferred
            raise ValueError(
                f"Multiple URDF files found in {urdf_path}. "
                f"Pass a specific file: {', '.join(os.path.basename(c) for c in candidates)}"
            )
        raise FileNotFoundError(f"URDF path not found: {urdf_path}")

    def _infer_robot_name(self) -> str:
        if os.path.isfile(self.urdf_path):
            base = os.path.splitext(os.path.basename(self.urdf_path))[0]
        else:
            base = os.path.basename(os.path.normpath(self.urdf_path))
        if base.endswith("_urdf"):
            base = base[: -len("_urdf")]
        return base

    def _infer_root_dir(self) -> str:
        parent = os.path.dirname(self.urdf_dir)
        if os.path.basename(parent) == "urdf":
            return os.path.dirname(parent)
        return parent

    def _maybe_migrate_cache(self) -> None:
        old_dir = os.path.join(self.urdf_dir, "data", "sdf_points")
        if os.path.exists(self.cache_dir) or not os.path.exists(old_dir):
            return
        os.makedirs(os.path.dirname(self.cache_dir), exist_ok=True)
        shutil.move(old_dir, self.cache_dir)
        print(f"Moved cache from {old_dir} to {self.cache_dir}")

    def _maybe_migrate_models(self) -> None:
        old_dir = os.path.join(self.urdf_dir, "models")
        if os.path.exists(self.model_dir) or not os.path.exists(old_dir):
            return
        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        shutil.move(old_dir, self.model_dir)
        print(f"Moved models from {old_dir} to {self.model_dir}")

    def _guess_end_link(self) -> str:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        parents = set()
        children = set()
        for joint in root.findall("joint"):
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            parents.add(parent)
            children.add(child)
        leaf = sorted(children - parents)
        if not leaf:
            raise ValueError("Failed to detect end link from URDF.")
        if len(leaf) > 1:
            print(f"Multiple leaf links detected, using {leaf[0]}")
        return leaf[0]

    def _resolve_mesh_path(self, filename: str, mesh_type: str) -> str:
        if filename.startswith("package://"):
            rel = filename.replace("package://", "")
            rel_parts = rel.split("/", 1)
            candidates = [os.path.join(self.urdf_dir, rel)]
            if len(rel_parts) == 2:
                candidates.append(os.path.join(self.urdf_dir, rel_parts[1]))
            if "meshes/" in rel:
                mesh_idx = rel.index("meshes/")
                candidates.append(os.path.join(self.urdf_dir, rel[mesh_idx:]))
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
        if os.path.isabs(filename):
            if os.path.exists(filename):
                return filename
        rel_path = os.path.join(self.urdf_dir, filename)
        if os.path.exists(rel_path):
            return rel_path
        basename = os.path.basename(filename)
        matches = sorted(glob.glob(os.path.join(self.urdf_dir, "**", basename), recursive=True))
        if matches:
            preferred = [m for m in matches if f"{os.sep}{mesh_type}{os.sep}" in m]
            if len(preferred) == 1:
                return preferred[0]
            if preferred:
                return preferred[0]
            return matches[0]
        raise FileNotFoundError(f"Mesh file not found: {filename}")

    @staticmethod
    def _rpy_xyz_to_matrix(rpy: List[float], xyz: List[float]) -> np.ndarray:
        roll, pitch, yaw = rpy
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        R = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float32,
        )
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = np.array(xyz, dtype=np.float32)
        return T

    def _parse_origin(self, elem: ET.Element) -> np.ndarray:
        origin = elem.find("origin")
        if origin is None:
            return np.eye(4, dtype=np.float32)
        xyz = [float(x) for x in origin.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(r) for r in origin.attrib.get("rpy", "0 0 0").split()]
        return self._rpy_xyz_to_matrix(rpy, xyz)
    
    def _build_joint_tree(self) -> Dict[str, Tuple[str, np.ndarray]]:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        joint_tree = {}
        for joint in root.findall("joint"):
            parent_elem = joint.find("parent")
            child_elem = joint.find("child")
            if parent_elem is not None and child_elem is not None:
                parent = parent_elem.attrib["link"]
                child = child_elem.attrib["link"]
                origin_mat = self._parse_origin(joint)
                joint_tree[child] = (parent, origin_mat)
        return joint_tree

    def _load_link_meshes(self) -> Dict[str, "trimesh.Trimesh"]:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        meshes: Dict[str, "trimesh.Trimesh"] = {}
        for link in root.findall("link"):
            link_name = link.attrib["name"]
            visuals = link.findall(self.mesh_type)
            if not visuals:
                continue
            mesh_list = []
            for visual in visuals:
                geom = visual.find("geometry")
                if geom is None:
                    continue
                origin_T = self._parse_origin(visual)
                mesh = None
                mesh_elem = geom.find("mesh")
                if mesh_elem is not None:
                    filename = mesh_elem.attrib.get("filename", "")
                    if not filename:
                        continue
                    mesh_path = self._resolve_mesh_path(filename, self.mesh_type)
                    mesh = trimesh.load(mesh_path, force="mesh")
                    scale_str = mesh_elem.attrib.get("scale")
                    if scale_str:
                        scale = [float(x) for x in scale_str.split()]
                        mesh.apply_scale(scale)
                else:
                    box = geom.find("box")
                    cylinder = geom.find("cylinder")
                    sphere = geom.find("sphere")
                    if box is not None:
                        size = [float(x) for x in box.attrib.get("size", "0 0 0").split()]
                        mesh = trimesh.creation.box(extents=size)
                    elif cylinder is not None:
                        radius = float(cylinder.attrib.get("radius", "0"))
                        length = float(cylinder.attrib.get("length", "0"))
                        mesh = trimesh.creation.cylinder(radius=radius, height=length)
                    elif sphere is not None:
                        radius = float(sphere.attrib.get("radius", "0"))
                        mesh = trimesh.creation.icosphere(radius=radius)
                if mesh is None:
                    continue
                mesh.apply_transform(origin_T)
                mesh_list.append(mesh)
            if mesh_list:
                meshes[link_name] = trimesh.util.concatenate(mesh_list)
        if not meshes:
            raise ValueError(f"No {self.mesh_type} meshes found in URDF.")
        return meshes

    @staticmethod
    def _mesh_offset_scale(mesh: "trimesh.Trimesh") -> Tuple[np.ndarray, float]:
        offset = mesh.bounding_box.centroid
        scale = np.max(np.linalg.norm(mesh.vertices - offset, axis=1))
        if scale < 1e-8:
            scale = 1e-8
        return offset, float(scale)

    def _cache_path(self, link_name: str) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{self.cache_prefix}_{link_name}.npy")

    def _load_or_sample(
        self, link_name: str, mesh: "trimesh.Trimesh", n_surface: int, n_random: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_path = self._cache_path(link_name)
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True).item()
            return (
                data["near_points"].astype(np.float32),
                data["near_sdf"].astype(np.float32),
                data["random_points"].astype(np.float32),
                data["random_sdf"].astype(np.float32),
            )
        mesh_unit = mesh_to_sdf.scale_to_unit_sphere(mesh.copy())
        random_points = np.random.uniform(-1.0, 1.0, size=(n_random, 3)).astype(np.float32)
        scan_kwargs = dict(DEFAULT_SCAN_KWARGS)
        mesh_sdf_kwargs = dict(DEFAULT_MESH_SDF_KWARGS)
        try:
            near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(
                mesh_unit,
                number_of_points=n_surface,
                surface_point_method="scan",
                return_gradients=False,
                **scan_kwargs,
            )
            random_sdf = mesh_to_sdf.mesh_to_sdf(
                mesh_unit,
                random_points,
                surface_point_method="scan",
                bounding_radius=None,
                **mesh_sdf_kwargs,
            )
        except Exception as exc:
            print(f"surface_method='scan' failed ({exc}); retrying with surface_method='sample'.")
            near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(
                mesh_unit,
                number_of_points=n_surface,
                surface_point_method="sample",
                return_gradients=False,
                **scan_kwargs,
            )
            random_sdf = mesh_to_sdf.mesh_to_sdf(
                mesh_unit,
                random_points,
                surface_point_method="sample",
                bounding_radius=None,
                **mesh_sdf_kwargs,
            )
        random_sdf = random_sdf.astype(np.float32)
        mask = random_sdf < -1.0
        random_sdf[mask] = -random_sdf[mask]
        data = {
            "near_points": near_points.astype(np.float32),
            "near_sdf": near_sdf.astype(np.float32),
            "random_points": random_points,
            "random_sdf": random_sdf,
        }
        np.save(cache_path, data)
        return data["near_points"], data["near_sdf"], data["random_points"], data["random_sdf"]

    def train(
        self,
        epochs: int = 500,
        n_surface: int = 500000,
        n_random: int = 500000,
        batch_near: int = 1024,
        batch_random: int = 256,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        model: Dict[str, Dict[str, torch.Tensor]] = {}
        for link_name in self.link_names:
            mesh = self.link_meshes[link_name].copy()
            offset, scale = self._mesh_offset_scale(mesh)
            near_points, near_sdf, random_points, random_sdf = self._load_or_sample(
                link_name, mesh, n_surface, n_random
            )
            wb = torch.zeros(self.basis.n_basis**3, device=self.device).float()
            B = (torch.eye(self.basis.n_basis**3, device=self.device) / 1e-4).float()
            for _ in range(epochs):
                near_count = min(batch_near, len(near_points))
                rand_count = min(batch_random, len(random_points))
                choice_near = np.random.choice(len(near_points), near_count, replace=False)
                choice_random = np.random.choice(len(random_points), rand_count, replace=False)
                p_near = torch.from_numpy(near_points[choice_near]).float().to(self.device)
                sdf_near = torch.from_numpy(near_sdf[choice_near]).float().to(self.device)
                p_random = torch.from_numpy(random_points[choice_random]).float().to(self.device)
                sdf_random = torch.from_numpy(random_sdf[choice_random]).float().to(self.device)
                p = torch.cat([p_near, p_random], dim=0)
                sdf = torch.cat([sdf_near, sdf_random], dim=0)
                phi_xyz, _ = self.basis.build_basis_function_from_points(p, use_derivative=False)
                K = B @ phi_xyz.T @ torch.linalg.inv(torch.eye(len(p), device=self.device) + phi_xyz @ B @ phi_xyz.T)
                B = B - K @ phi_xyz @ B
                wb = wb + K @ (sdf - phi_xyz @ wb).squeeze()
            model[link_name] = {
                "mesh_name": link_name,
                "weights": wb.detach().cpu(),
                "offset": torch.from_numpy(offset).float(), #! Lo cambio para que no de el erro de los floatde Numpy
                #"offset": torch.from_numpy(offset.copy()).float(),
                "scale": float(scale),
            }
            print(f"Fitted link {link_name}")
        self.model = model
        return model

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("No model to save. Call train() or load() first.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model, path)
        print(f"Saved model to {path}")

    def load(self, path: str) -> Dict[str, Dict[str, torch.Tensor]]:
        self.model = torch.load(path, map_location=self.device)
        print(f"Loaded model from {path}")
        return self.model

    def _get_link_transforms(
        self, theta: torch.Tensor, used_links: Iterable[str], base_pose: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta[None, :]
        trans = self.chain.forward_kinematics(theta)
        if base_pose is None:
            base_pose = torch.eye(4, device=theta.device, dtype=theta.dtype).view(1, 4, 4).expand(theta.shape[0], 4, 4)
        elif base_pose.ndim == 2:
            base_pose = base_pose.view(1, 4, 4).expand(theta.shape[0], 4, 4)
        if base_pose.device != theta.device:
            base_pose = base_pose.to(theta.device)
        transforms: Dict[str, torch.Tensor] = {}
        for k, v in trans.items():
            if k not in used_links:
                continue
            transforms[k] = torch.matmul(base_pose, v.get_matrix())
        return transforms

    @staticmethod
    def _transform_points(points: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        B = trans.shape[0]
        if points.ndim == 2:
            N = points.shape[0]
            pts_exp = points.unsqueeze(0).expand(B, N, 3)
        elif points.ndim == 3:
            if points.shape[0] != B:
                raise ValueError(f"Batch mismatch for point transform: points {points.shape}, trans {trans.shape}")
            N = points.shape[1]
            pts_exp = points
        else:
            raise ValueError(f"points must be (N,3) or (B,N,3); got {points.shape}")
        ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
        pts = torch.cat([pts_exp, ones], dim=-1)
        pts_t = torch.matmul(trans, pts.transpose(1, 2)).transpose(1, 2)
        return pts_t[:, :, :3]

    def query(
        self,
        points: torch.Tensor,
        theta: torch.Tensor,
        return_grad: bool = True,
        base_pose: Optional[torch.Tensor] = None,
        used_links: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.model is None:
            raise ValueError("No model loaded. Call train() or load() first.")
        if points.ndim == 2:
            points = points[None, ...]
        if theta.ndim == 1:
            theta = theta[None, ...]
        if points.shape[0] != theta.shape[0] and points.shape[0] != 1 and theta.shape[0] != 1:
            raise ValueError(f"Batch mismatch: points {points.shape}, theta {theta.shape}")
        B = max(points.shape[0], theta.shape[0])
        if points.shape[0] == 1 and B > 1:
            points = points.expand(B, -1, -1)
        if theta.shape[0] == 1 and B > 1:
            theta = theta.expand(B, -1)
        points = points.to(self.device).float()
        theta = theta.to(self.device).float()

        link_names = used_links if used_links is not None else [ln for ln in self.link_names if ln in self.model]
        if len(link_names) == 0:
            raise ValueError("No overlapping link names between requested links and the model.")
        transforms = self._get_link_transforms(theta, link_names, base_pose=base_pose)
        if len(transforms) == 0:
            raise ValueError("No overlapping link names between kinematic chain and requested links.")

        K = len(link_names)
        N = points.shape[1]

        #fk_trans = torch.stack([transforms[name] for name in link_names], dim=1)
        device = list(transforms.values())[0].device
        dtype = list(transforms.values())[0].dtype

        # Función recursiva que busca la posición de cualquier link
        def get_transform(link_name):
            # 1. Si la cinemática lo conoce (es móvil o raíz), devolvemos su posición
            if link_name in transforms:
                return transforms[link_name]
            
            # 2. Si el link no existe ni en el URDF, usamos el de la raíz por seguridad
            if link_name not in self.joint_tree:
                fallback = 'rb1_base_footprint' if 'rb1_base_footprint' in transforms else list(transforms.keys())[0]
                return transforms[fallback]
            
            # 3. Si es fijo, buscamos a su padre en el URDF y su distancia (matriz 4x4)
            parent_name, local_np = self.joint_tree[link_name]
            
            # Buscamos dónde está el padre en el mundo real
            parent_T = get_transform(parent_name) 
            
            # Convertimos la matriz local a tensor y multiplicamos (Global = Padre * Local)
            local_T = torch.tensor(local_np, dtype=dtype, device=device)
            return torch.matmul(parent_T, local_T)

        # Calculamos todos los links con una sola línea elegante
        fk_trans = torch.stack([get_transform(name) for name in link_names], dim=1)
        fk_flat = fk_trans.reshape(-1, 4, 4)
        fk_inv = torch.linalg.inv(fk_flat)

        pts_expanded = points.unsqueeze(1).expand(B, K, N, 3).reshape(B * K, N, 3)
        x_link = self._transform_points(pts_expanded, fk_inv)

        offset = torch.stack([self.model[name]["offset"] for name in link_names], dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B, K, 3).reshape(B * K, 3).float()
        scale = torch.tensor([self.model[name]["scale"] for name in link_names], device=self.device)
        scale = scale.unsqueeze(0).expand(B, K).reshape(B * K).float()

        x_scaled = x_link - offset.unsqueeze(1)
        x_scaled = x_scaled / scale.unsqueeze(-1).unsqueeze(-1)
        x_bounded, res_x = self.basis.bound_points_to_unit_sphere(x_scaled)

        if not return_grad:
            phi, _ = self.basis.build_basis_function_from_points(x_bounded.reshape(-1, 3), use_derivative=False)
            phi = phi.reshape(B, K, N, -1).transpose(0, 1).reshape(K, B * N, -1)
            weights = torch.stack([self.model[name]["weights"] for name in link_names], dim=0).to(self.device)
            sdf = torch.einsum("ijk,ik->ij", phi, weights).reshape(K, B, N).transpose(0, 1).reshape(B * K, N)
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
            sdf_value, _ = sdf.min(dim=1)
            return sdf_value, None

        phi, dphi = self.basis.build_basis_function_from_points(x_bounded.reshape(-1, 3), use_derivative=True)
        phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
        phi_cat = phi_cat.reshape(B, K, N, -1, 4).transpose(0, 1).reshape(K, B * N, -1, 4)
        weights = torch.stack([self.model[name]["weights"] for name in link_names], dim=0).to(self.device)
        output = torch.einsum("ijkl,ik->ijl", phi_cat, weights).reshape(K, B, N, 4).transpose(0, 1).reshape(B * K, N, 4)
        sdf = output[:, :, 0]
        gradient = output[:, :, 1:]
        sdf = sdf + res_x.norm(dim=-1)
        sdf = sdf.reshape(B, K, N)
        sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
        sdf_value, idx = sdf.min(dim=1)

        gradient = res_x + torch.nn.functional.normalize(gradient, dim=-1)
        gradient = torch.nn.functional.normalize(gradient, dim=-1).float()
        fk_rotation = fk_flat[:, :3, :3]
        gradient_base = torch.einsum("ijk,ikl->ijl", fk_rotation, gradient.transpose(1, 2)).transpose(1, 2).reshape(B, K, N, 3)
        idx_grad = idx.unsqueeze(1).unsqueeze(-1).expand(B, K, N, 3)
        gradient_value = torch.gather(gradient_base, 1, idx_grad)[:, 0, :, :]
        return sdf_value, gradient_value

    def visualize(
        self,
        theta: torch.Tensor,
        nb_data: int = 64,
        domain_min: Tuple[float, float, float] = (-2.5, -2.5, -2.5),
        domain_max: Tuple[float, float, float] = (2.5, 2.5, 2.5),
        show: bool = True,
        out_path: Optional[str] = None,
        mesh_points=200_000,
    ) -> "trimesh.Trimesh":
        if measure is None:
            raise ModuleNotFoundError("Missing dependency 'scikit-image' (skimage). Install it to run marching cubes.")
        if trimesh is None:
            raise ModuleNotFoundError("Missing dependency 'trimesh'. Install it to visualize meshes.") from _TRIMESH_IMPORT_ERROR
        xs = torch.linspace(domain_min[0], domain_max[0], nb_data, device=self.device)
        ys = torch.linspace(domain_min[1], domain_max[1], nb_data, device=self.device)
        zs = torch.linspace(domain_min[2], domain_max[2], nb_data, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        sdf_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for p in torch.split(points, mesh_points, dim=0):
                v, _g = self.query(p, theta, return_grad=False)
                sdf_chunks.append(v.reshape(-1).detach())
        sdf_flat = torch.cat(sdf_chunks, dim=0)
        sdf_volume = sdf_flat.reshape(nb_data, nb_data, nb_data).float().cpu().numpy()
        origin = np.array(domain_min, dtype=np.float32)
        spacing = (
            (domain_max[0] - domain_min[0]) / (nb_data - 1),
            (domain_max[1] - domain_min[1]) / (nb_data - 1),
            (domain_max[2] - domain_min[2]) / (nb_data - 1),
        )
        verts, faces, _normals, _values = measure.marching_cubes(sdf_volume, level=0.0, spacing=spacing)
        verts = verts + origin[None, :]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            mesh.export(out_path)
        if show:
            try:
                mesh.show()
            except BaseException as exc:
                print(f"Mesh viewer failed: {exc}")
        return mesh
    
    def visualize_lightweight(
        self,
        theta: torch.Tensor,
        num_points_per_link: int = 300,
        show_frames: bool = False,
        show_pointcloud: bool = True,
        show: bool = True,
    ) -> "trimesh.Scene":
        """
        Visualización interactiva y ligera del robot.
        - show_frames: Muestra los ejes de coordenadas simulando el "perfil" o esqueleto del robot.
        - show_pointcloud: Muestra una nube de puntos envolvente que representa el límite del SDF.
        """
        if trimesh is None:
            raise ModuleNotFoundError("Se requiere 'trimesh' para la visualización ligera.")
        
        # Ajuste de dimensiones para el tensor
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)

        # 1. Calcular cinemática directa (FK) para obtener las poses de los links
        link_names = [ln for ln in self.link_names if ln in self.model]
        if not link_names:
            print("No hay modelos cargados para visualizar.")
            return trimesh.Scene()
            
        transforms = self._get_link_transforms(theta, link_names)
        device = list(transforms.values())[0].device
        dtype = list(transforms.values())[0].dtype

        # Función recursiva de FK idéntica a la de query()
        def get_transform(link_name):
            if link_name in transforms:
                return transforms[link_name][0]  # Tomar el primer elemento del batch
            if link_name not in self.joint_tree:
                fallback = 'rb1_base_footprint' if 'rb1_base_footprint' in transforms else list(transforms.keys())[0]
                return transforms[fallback][0]
            parent_name, local_np = self.joint_tree[link_name]
            parent_T = get_transform(parent_name)
            local_T = torch.tensor(local_np, dtype=dtype, device=device)
            return torch.matmul(parent_T, local_T)

        scene = trimesh.Scene()
        all_surface_points = []
        all_colors = []

        # 2. Generar elementos visuales link por link
        for name in link_names:
            T_tensor = get_transform(name)
            T_matrix = T_tensor.detach().cpu().numpy()

            # PERFIL DEL ROBOT: Dibujar los sistemas de coordenadas (Frames)
            if show_frames:
                frame = trimesh.creation.axis(origin_size=0.015, transform=T_matrix)
                scene.add_geometry(frame)

            # NUBE DE PUNTOS: Superficie del robot actualizada según su pose
            if show_pointcloud and name in self.link_meshes:
                mesh = self.link_meshes[name]
                
                # Muestrear puntos de la superficie local del link
                # (Se podría cachear esto en el __init__ para ser aún más rápidos)
                points, _ = trimesh.sample.sample_surface(mesh, num_points_per_link)
                
                # Transformar puntos al espacio global (multiplicación matricial homogénea)
                points_hom = np.hstack((points, np.ones((len(points), 1))))
                points_world = (T_matrix @ points_hom.T).T[:, :3]
                
                all_surface_points.append(points_world)
                
                # Color distintivo por link para ver las descomposiciones (Rojo, Verde, Azul, etc.)
                # Generamos colores pastel aleatorios
                color = np.random.randint(80, 220, size=3).tolist() + [255]
                colors = np.tile(color, (len(points_world), 1))
                all_colors.append(colors)

        # 3. Ensamblar la nube de puntos final
        if show_pointcloud and all_surface_points:
            final_points = np.vstack(all_surface_points)
            final_colors = np.vstack(all_colors)
            point_cloud = trimesh.points.PointCloud(final_points, colors=final_colors)
            scene.add_geometry(point_cloud)

        # 4. Mostrar escena
        if show:
            try:
                scene.show()
            except BaseException as exc:
                print(f"El visor interactivo falló: {exc}")

        return scene
    
    def get_surface_points(self, theta: torch.Tensor, num_points_per_link: int = 150, safety_margin: float = 0.0) -> np.ndarray:
        """
        Calcula y devuelve los puntos de la superficie de todo el robot 
        transformados al espacio global según la configuración articular actual.
        
        Args:
            theta: Tensor con la configuración articular.
            num_points_per_link: Número de puntos a muestrear por link.
            safety_margin: Margen de seguridad en metros para "inflar" la superficie hacia afuera.
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)

        link_names = [ln for ln in self.link_names if ln in self.model]
        if not link_names:
            return np.empty((0, 3))
            
        transforms = self._get_link_transforms(theta, link_names)
        device = list(transforms.values())[0].device
        dtype = list(transforms.values())[0].dtype

        def get_transform(link_name):
            if link_name in transforms:
                return transforms[link_name][0]
            if link_name not in self.joint_tree:
                fallback = list(transforms.keys())[0]
                return transforms[fallback][0]
            parent_name, local_np = self.joint_tree[link_name]
            parent_T = get_transform(parent_name)
            local_T = torch.tensor(local_np, dtype=dtype, device=device)
            return torch.matmul(parent_T, local_T)

        all_surface_points = []

        for name in link_names:
            T_tensor = get_transform(name)
            T_matrix = T_tensor.detach().cpu().numpy()

            if name in self.link_meshes:
                mesh = self.link_meshes[name]
                
                # Muestreamos puntos locales y los índices de las caras correspondientes
                points, face_indices = trimesh.sample.sample_surface(mesh, num_points_per_link)
                
                # Si se define un margen de seguridad, desplazamos los puntos usando las normales
                if safety_margin > 0.0:
                    # Obtenemos las normales de las caras donde cayó cada punto muestreado
                    normals = mesh.face_normals[face_indices]
                    # Desplazamos los puntos en la dirección de la normal una distancia igual al margen
                    points = points + normals * safety_margin
                
                # Transformamos los puntos expandidos a coordenadas globales mundiales
                points_hom = np.hstack((points, np.ones((len(points), 1))))
                points_world = (T_matrix @ points_hom.T).T[:, :3]
                all_surface_points.append(points_world)

        if all_surface_points:
            return np.vstack(all_surface_points)
        return np.empty((0, 3))
    
    def get_level_curves(self, theta: torch.Tensor, levels: List[float] = [0.0, 0.05, 0.1, 0.2], num_points_per_link: int = 80) -> List[Tuple[np.ndarray, List[float]]]:
        """
        Genera múltiples iso-superficies discretas (curvas de nivel) inflando 
        el contorno del robot a diferentes rangos de seguridad.
        
        Args:
            theta: Tensor articular.
            levels: Distancias del SDF en metros que deseamos mapear visualmente.
            num_points_per_link: Densidad de puntos de la malla por nivel.
            
        Returns:
            Una lista de tuplas [(puntos_nivel_i, color_rgb_i), ...] lista para renderizar en paralelo.
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)

        link_names = [ln for ln in self.link_names if ln in self.model]
        if not link_names:
            return []
            
        transforms = self._get_link_transforms(theta, link_names)
        device = list(transforms.values())[0].device
        dtype = list(transforms.values())[0].dtype

        def get_transform(link_name):
            if link_name in transforms:
                return transforms[link_name][0]
            if link_name not in self.joint_tree:
                fallback = list(transforms.keys())[0]
                return transforms[fallback][0]
            parent_name, local_np = self.joint_tree[link_name]
            parent_T = get_transform(parent_name)
            local_T = torch.tensor(local_np, dtype=dtype, device=device)
            return torch.matmul(parent_T, local_T)

        # Paleta de degradado precalculada para las curvas de nivel (Estilo Jet / Mapa térmico)
        def get_level_color(idx, total_levels):
            if total_levels <= 1:
                return [0.1, 0.6, 1.0] # Azul por defecto
            ratio = idx / (total_levels - 1)
            # Interpolación simple de Rojo (superficie corta) a Verde (superficie externa segura)
            r = float(1.0 - ratio)
            g = float(ratio)
            b = float(0.2 + ratio * 0.4)
            return [r, g, b]

        curves_data = []

        # Estructuramos la iteración primero por nivel para empaquetar nubes consolidadas
        for l_idx, margin in enumerate(levels):
            level_points = []
            
            for name in link_names:
                T_matrix = get_transform(name).detach().cpu().numpy()

                if name in self.link_meshes:
                    mesh = self.link_meshes[name]
                    points, face_indices = trimesh.sample.sample_surface(mesh, num_points_per_link)
                    
                    if margin > 0.0:
                        normals = mesh.face_normals[face_indices]
                        points = points + normals * margin
                    
                    points_hom = np.hstack((points, np.ones((len(points), 1))))
                    points_world = (T_matrix @ points_hom.T).T[:, :3]
                    level_points.append(points_world)
            
            if level_points:
                stacked_points = np.vstack(level_points)
                color_rgb = get_level_color(l_idx, len(levels))
                curves_data.append((stacked_points, color_rgb))

        return curves_data
    
    def visualize_level_curves(
        self,
        theta: torch.Tensor,
        mode: str = "3d",  # "3d" o "2d"
        levels: List[float] = [0.0, 0.05, 0.1, 0.2],
        num_points_per_link: int = 150,
        show_frames: bool = False,
        show: bool = True,
        # Parámetros específicos para la sección de corte 2D
        plane: str = "xy",
        intercept: float = 0.0,
        bounds: Tuple[float, float, float, float] = (-1.0, 1.0, -0.5, 2.0,),
        res: int = 100,
    ) -> Optional["trimesh.Scene"]:
        """
        Visualiza las curvas de nivel (iso-superficies) del SDF.
        Permite alternar entre una vista volumétrica 3D (Trimesh) y un corte planar 2D (Matplotlib).
        """
        # ==========================================
        # MODALIDAD 2D: CORTE DE PLANO (MATPLOTLIB)
        # ==========================================
        if mode == "2d":
            import matplotlib.pyplot as plt
            
            # Generar la rejilla de evaluación planar
            u = torch.linspace(bounds[0], bounds[1], res, device=self.device)
            v = torch.linspace(bounds[2], bounds[3], res, device=self.device)
            grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")
            
            flat_u = grid_u.reshape(-1)
            flat_v = grid_v.reshape(-1)
            flat_fixed = torch.ones_like(flat_u) * intercept

            # Mapear los ejes correspondientes según el plano solicitado
            if plane == "xy":
                points_3d = torch.stack([flat_u, flat_v, flat_fixed], dim=-1)
                xlabel, ylabel = "X (metros)", "Y (metros)"
            elif plane == "xz":
                points_3d = torch.stack([flat_u, flat_fixed, flat_v], dim=-1)
                xlabel, ylabel = "X (metros)", "Z (metros)"
            elif plane == "yz":
                points_3d = torch.stack([flat_fixed, flat_u, flat_v], dim=-1)
                xlabel, ylabel = "Y (metros)", "Z (metros)"
            else:
                raise ValueError("Plano no válido. Elige entre 'xy', 'xz' o 'yz'.")

            # Evaluar por fragmentos el campo escalar tridimensional del SDF
            sdf_list = []
            with torch.no_grad():
                for chunk in torch.split(points_3d, 10000, dim=0):
                    sdf_chunk, _ = self.query(chunk, theta, return_grad=False)
                    sdf_list.append(sdf_chunk.reshape(-1))
            
            sdf_flat = torch.cat(sdf_list, dim=0)
            sdf_grid = sdf_flat.reshape(res, res).cpu().numpy()
            grid_u_np = grid_u.cpu().numpy()
            grid_v_np = grid_v.cpu().numpy()

            if show:
                plt.figure(figsize=(8, 7))
                # Fondo térmico continuo para denotar el gradiente del campo
                contour_f = plt.contourf(grid_u_np, grid_v_np, sdf_grid, levels=50, cmap="RdYlGn")
                plt.colorbar(contour_f, label="SDF value (meters)")
                
                # Resaltar la superficie metálica (0.0) sobre los márgenes flotantes
                linewidths = [2.5 if l == 0.0 else 1.2 for l in levels]
                linestyles = ["-" if l >= 0.0 else "--" for l in levels]
                
                contours = plt.contour(
                    grid_u_np, grid_v_np, sdf_grid, 
                    levels=levels, 
                    colors="black", 
                    linewidths=linewidths,
                    linestyles=linestyles
                )
                plt.clabel(contours, inline=True, fmt="%.2f", fontsize=10)
                plt.title(f"SDF Cross-Section in the {plane.upper()} plane (Orthogonal axes = {intercept}m)")
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.grid(True, linestyle=":", alpha=0.6)
                plt.axis("equal")
                plt.show()

            return grid_u_np, grid_v_np, sdf_grid

        # ==========================================
        # MODALIDAD 3D: NUBES VOLUMÉTRICAS (TRIMESH)
        # ==========================================
        elif mode == "3d":
            if trimesh is None:
                raise ModuleNotFoundError("Se requiere 'trimesh' para la visualización de curvas de nivel 3D.")
            
            curves_data = self.get_level_curves(theta, levels=levels, num_points_per_link=num_points_per_link)
            scene = trimesh.Scene()
            
            if show_frames:
                link_names = [ln for ln in self.link_names if ln in self.model]
                transforms = self._get_link_transforms(theta, link_names)
                device = list(transforms.values())[0].device
                dtype = list(transforms.values())[0].dtype

                def get_transform(link_name):
                    if link_name in transforms:
                        return transforms[link_name][0]
                    if link_name not in self.joint_tree:
                        fallback = list(transforms.keys())[0]
                        return transforms[fallback][0]
                    parent_name, local_np = self.joint_tree[link_name]
                    parent_T = get_transform(parent_name)
                    local_T = torch.tensor(local_np, dtype=dtype, device=device)
                    return torch.matmul(parent_T, local_T)

                for name in link_names:
                    T_matrix = get_transform(name).detach().cpu().numpy()
                    frame = trimesh.creation.axis(origin_size=0.015, transform=T_matrix)
                    scene.add_geometry(frame)

            for i, (points, color_float) in enumerate(curves_data):
                color_255 = [int(c * 255) for c in color_float] + [255]
                colors_array = np.tile(color_255, (len(points), 1))
                point_cloud = trimesh.points.PointCloud(points, colors=colors_array)
                scene.add_geometry(point_cloud, node_name=f"sdf_level_{levels[i]}")

            if show:
                try:
                    scene.show()
                except BaseException as exc:
                    print(f"El visor interactivo de Trimesh falló: {exc}")

            return scene
        else:
            raise ValueError("Modo no reconocido. Use '3d' o '2d'.")
