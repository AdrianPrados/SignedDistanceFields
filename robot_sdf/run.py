# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

import argparse
import os
import xml.etree.ElementTree as ET

import torch

from .robot_sdf import RobotSDF


def _default_model_path(robot: RobotSDF, n_func: int, n_seg: int, mesh_type: str) -> str:
    base = os.path.splitext(os.path.basename(robot.urdf_path))[0]
    suffix = f"{n_func}_seg{n_seg}" if n_seg != 1 else f"{n_func}"
    return os.path.join(robot.model_dir, f"{base}_bf_{suffix}_{mesh_type}.pt")


def _parse_joint_limits(urdf_path: str) -> dict:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limits = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        joint_type = joint.attrib.get("type")
        if joint_type == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = limit.attrib.get("lower")
        upper = limit.attrib.get("upper")
        if lower is None or upper is None:
            continue
        try:
            limits[name] = (float(lower), float(upper))
        except ValueError:
            continue
    return limits


def _midpoint_theta(robot: RobotSDF) -> torch.Tensor:
    limits = _parse_joint_limits(robot.urdf_path)
    joint_names = robot.chain.get_joint_parameter_names()
    theta = []
    for name in joint_names:
        if name in limits:
            lower, upper = limits[name]
            theta.append(0.3 * (lower + upper))
        else:
            theta.append(0.0)
    return torch.tensor(theta, dtype=torch.float32, device=robot.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and visualize robot SDF from a URDF.")
    parser.add_argument("--urdf", required=True, type=str, help="path to the URDF file")
    parser.add_argument("--mesh-type", default="visual", choices=["visual", "collision"])
    parser.add_argument("--n-func", default=4, type=int)
    parser.add_argument("--n-seg", default=3, type=int)
    args = parser.parse_args()

    robot = RobotSDF(
        urdf_path=args.urdf,
        mesh_type=args.mesh_type,
        n_func=args.n_func,
        n_seg=args.n_seg,
    )
    model_path = _default_model_path(robot, args.n_func, args.n_seg, args.mesh_type)
    if os.path.exists(model_path):
        robot.load(model_path)
    else:
        robot.train()
        robot.save(model_path)

    theta = _midpoint_theta(robot)
    robot.visualize(theta, show=True, out_path=None)


if __name__ == "__main__":
    main()
