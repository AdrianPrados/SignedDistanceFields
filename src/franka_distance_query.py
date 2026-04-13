import glob
import os
import re
import sys
from typing import Optional, Tuple

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robot_sdf import RobotSDF  # noqa: E402


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
        names = ", ".join(os.path.basename(c) for c in candidates)
        raise ValueError(f"Multiple URDF files found in {urdf_path}: {names}")
    raise FileNotFoundError(f"URDF path not found: {urdf_path}")


def _infer_robot_name(urdf_path: str) -> str:
    base = os.path.splitext(os.path.basename(urdf_path))[0]
    return base[: -len("_urdf")] if base.endswith("_urdf") else base


def _infer_root_dir(urdf_dir: str) -> str:
    parent = os.path.dirname(urdf_dir)
    return os.path.dirname(parent) if os.path.basename(parent) == "urdf" else parent


def _default_model_path(model_dir: str, urdf_base: str, n_func: int, n_seg: int, mesh_type: str) -> str:
    suffix = f"{n_func}_seg{n_seg}" if n_seg != 1 else f"{n_func}"
    return os.path.join(model_dir, f"{urdf_base}_bf_{suffix}_{mesh_type}.pt")


def _find_existing_model(model_dir: str, urdf_base: str, mesh_type: str) -> Optional[Tuple[str, int, int]]:
    if not os.path.isdir(model_dir):
        return None
    pattern = os.path.join(model_dir, f"{urdf_base}_bf_*_{mesh_type}.pt")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for path in candidates:
        match = re.search(
            rf"{re.escape(urdf_base)}_bf_(\d+)(?:_seg(\d+))?_{mesh_type}\.pt$",
            os.path.basename(path),
        )
        if match:
            n_func = int(match.group(1))
            n_seg = int(match.group(2) or 1)
            return path, n_func, n_seg
    return None


def main() -> None:
    urdf_path = os.path.join(REPO_ROOT, "urdf", "panda_urdf")
    mesh_type = "visual"

    urdf_path = _resolve_urdf_path(urdf_path)
    urdf_base = os.path.splitext(os.path.basename(urdf_path))[0]
    root_dir = _infer_root_dir(os.path.dirname(urdf_path))
    model_dir = os.path.join(root_dir, "models", _infer_robot_name(urdf_path))

    model_info = _find_existing_model(model_dir, urdf_base, mesh_type)
    if model_info:
        model_path, n_func, n_seg = model_info
        print(f"Using existing model: {model_path}")
    else:
        n_func, n_seg = 4, 3
        model_path = _default_model_path(model_dir, urdf_base, n_func, n_seg, mesh_type)
        print("No existing model found; training a new one.")

    robot = RobotSDF(urdf_path, mesh_type=mesh_type, n_func=n_func, n_seg=n_seg)

    if os.path.exists(model_path):
        robot.load(model_path)
    else:
        robot.train()
        robot.save(model_path)

    # theta = torch.tensor([0, -0.3, 0, -2.2, 0, 2.0, torch.pi/4]).float().to(robot.device).reshape(-1,7)
    theta = torch.zeros(2,7,device=robot.device)
    points = torch.tensor([[0.0,0.5,0.3],[0.3,0.8,1.1]], device=robot.device)
    sdf, grad = robot.query(points, theta, return_grad=True)
    print("Distance:", sdf.detach().cpu().numpy(),grad)


if __name__ == "__main__":
    main()
