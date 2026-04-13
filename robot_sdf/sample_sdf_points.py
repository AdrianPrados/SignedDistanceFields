# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

import argparse

from .robot_sdf import RobotSDF


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cached SDF samples per link from a URDF.")
    parser.add_argument("--urdf", required=True, type=str)
    parser.add_argument("--mesh-type", default="visual", choices=["visual", "collision"])
    parser.add_argument("--n-surface", default=200000, type=int)
    parser.add_argument("--n-random", default=200000, type=int)
    args = parser.parse_args()

    robot = RobotSDF(args.urdf, mesh_type=args.mesh_type)
    for link_name in robot.link_names:
        mesh = robot.link_meshes[link_name]
        robot._load_or_sample(link_name, mesh, args.n_surface, args.n_random)
        print(f"cached {link_name}")


if __name__ == "__main__":
    main()
