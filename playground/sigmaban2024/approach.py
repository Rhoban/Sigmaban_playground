import numpy as np
import argparse
import mujoco

from playground.sigmaban2024.mujoco_infer import MjInfer
from playground.sigmaban2024.footsteps_mapper import FootstepsMapper
from playground.sigmaban2024.footsteps_net import FootstepsNet
import meshcat.transformations as tf


class ApproachSimulator:
    def __init__(
        self, mjinfer: MjInfer, mapper: FootstepsMapper, footsteps_net: FootstepsNet
    ):
        self.mjinfer = mjinfer
        self.mapper = mapper
        self.footsteps_net = footsteps_net

    def draw_footstep(self, T_world_footstep: np.ndarray):
        # Drawing target footstep
        foot_length = 0.15
        foot_width = 0.08
        points = [
            [-foot_length / 2, foot_width / 2],
            [-foot_length / 2, -foot_width / 2],
            [foot_length / 2, -foot_width / 2],
            [foot_length / 2, foot_width / 2],
        ]
        for k in range(4):
            point = T_world_footstep @ [*points[k], 0, 1]
            next_point = T_world_footstep @ [*points[(k + 1) % len(points)], 0, 1]

            mujoco.mjv_connector(
                self.mjinfer.viewer.user_scn.geoms[k],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                width=25,
                from_=point[:3],
                to=next_point[:3],
            )
            self.mjinfer.viewer.user_scn.geoms[k].rgba = [1, 0.5, 0.5, 1]
        self.mjinfer.viewer.user_scn.ngeom = 4

    def walk_approach(self):
        self.mjinfer.enable_viewer(False)

        while True:
            self.mjinfer.reset()

            x = np.random.uniform(-0.75, 0.75)
            y = np.random.uniform(-0.75, 0.75)
            yaw = np.random.uniform(-np.pi, np.pi)
            T_world_target = tf.translation_matrix((x, y, 0)) @ tf.rotation_matrix(
                yaw, (0, 0, 1)
            )
            self.draw_footstep(T_world_target)
            arrived = False

            while not arrived:
                T_world_left = self.mjinfer.get_T_world_site("left_foot")
                T_world_right = self.mjinfer.get_T_world_site("right_foot")

                T_right_target = np.linalg.inv(T_world_right) @ T_world_target
                error_pos = np.linalg.norm(T_right_target[:2, 3])
                error_yaw = abs(np.arctan2(T_right_target[1, 0], T_right_target[0, 0]))
                arrived = error_pos < 1.5e-2 and error_yaw < np.deg2rad(5)

                dx, dy, dtheta = self.footsteps_net.infer(
                    T_world_left,
                    T_world_right,
                    T_world_target,
                    self.mjinfer.support,
                    "right",
                )

                vx, vy, vtheta = self.mapper.remap(dx, dy, dtheta)

                self.mjinfer.set_command(vx, vy, vtheta)
                self.mjinfer.walk_one_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--walk_onnx", type=str, required=True)
    parser.add_argument("-f", "--footstepsnet_onnx", type=str, required=True)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/sigmaban2024/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/sigmaban2024/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument(
        "--footsteps_mapping",
        type=str,
        default="footsteps_mapping.json",
    )
    args = parser.parse_args()

    mjinfer = MjInfer(args.model_path, args.reference_data, args.walk_onnx)
    footsteps_net = FootstepsNet(args.footstepsnet_onnx)
    mapper = FootstepsMapper()
    mapper.load(args.footsteps_mapping)

    approach = ApproachSimulator(mjinfer, mapper, footsteps_net)
    approach.walk_approach()
