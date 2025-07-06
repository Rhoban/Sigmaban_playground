import numpy as np
import placo
import argparse
import mujoco

from playground.sigmaban2024.mujoco_infer import MjInfer, USE_FOOTSTEP_REWARD
from playground.sigmaban2024.footsteps_mapper import FootstepsMapper
from playground.sigmaban2024.footsteps_net import FootstepsNet
import meshcat.transformations as tf


class Trajectory:
    def __init__(self, footsteps_net: FootstepsNet):
        self.footsteps_net = footsteps_net
        self.left_foot_color = [0, 0, 0, 0.5]
        self.right_foot_color = [1, 0, 0, 0.5]

    def draw_trajectory(self, scene, trajectory):
        for step in trajectory:
            T_world_left = step["T_world_left"]
            T_world_right = step["T_world_right"]
            support_foot = step["support"]

            if support_foot == "left":
                T_world_footstep = T_world_left
            else:
                T_world_footstep = T_world_right

            pos = T_world_footstep[:3, 3]
            theta = np.arctan2(T_world_footstep[1, 0], T_world_footstep[0, 0])
            render_footstep(
                scene,
                [pos[0], pos[1], theta],
                [0.14, 0.08],
                color=(
                    self.left_foot_color
                    if support_foot == "left"
                    else self.right_foot_color
                ),
            )

    def sample_trajectory(
        self,
        humanoid_parameters: placo.HumanoidParameters,
        T_world_left,
        T_world_right,
        starting_support_foot,
        T_world_target,
        target_support_foot,
    ):
        support_foot = starting_support_foot

        arrived = False

        self.trajectory = []
        nb_steps = 0
        while not arrived and nb_steps < 100:
            dx, dy, dtheta = self.footsteps_net.infer(
                T_world_left,
                T_world_right,
                T_world_target,
                support_foot,
                target_support_foot,
            )

            dx, dy, dtheta = humanoid_parameters.ellipsoid_overlap_clip(
                (
                    placo.HumanoidRobot_Side.left
                    if support_foot == "left"
                    else placo.HumanoidRobot_Side.right
                ),
                np.array([dx, dy, dtheta]),
            )

            self.trajectory.append(
                {
                    "support": support_foot,
                    "dx": dx,
                    "dy": dy,
                    "dtheta": dtheta,
                    "T_world_left": T_world_left,
                    "T_world_right": T_world_right,
                }
            )

            if support_foot == "left":
                T_world_right = (
                    T_world_left
                    @ tf.translation_matrix(
                        (dx, dy - self.footsteps_net.feet_spacing, 0)
                    )
                    @ tf.rotation_matrix(dtheta, (0, 0, 1))
                )
            else:
                T_world_left = (
                    T_world_right
                    @ tf.translation_matrix(
                        (dx, dy + self.footsteps_net.feet_spacing, 0)
                    )
                    @ tf.rotation_matrix(dtheta, (0, 0, 1))
                )

            T_right_target = np.linalg.inv(T_world_right) @ T_world_target
            error_pos = np.linalg.norm(T_right_target[:2, 3])
            error_yaw = abs(np.arctan2(T_right_target[1, 0], T_right_target[0, 0]))
            arrived = error_pos < 1e-2 and error_yaw < np.deg2rad(2)

            support_foot = "left" if support_foot == "right" else "right"
            nb_steps += 1
        return self.trajectory


def render_plane(scene, center, rot_mat, size, rgba):
    if scene.ngeom >= len(scene.geoms):
        return

    sx, sy = size

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_PLANE,
        size=np.array(
            [sx / 2, sy / 2, 0.0], dtype=np.float64
        ),  # Planes have no thickness
        pos=np.array(center, dtype=np.float64),
        mat=rot_mat.flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )

    scene.ngeom += 1


def render_footstep(scene, pos, foot_size, color=[1, 0, 0, 0.5]):
    """
    pos : [x, y, theta] (m, m, rad)
    """
    x, y = pos[:2]
    center = [x, y, 0.001]
    theta = pos[2]
    size = foot_size

    rot_mat = np.eye(3)
    rot_mat[0, 0] = np.cos(theta)
    rot_mat[0, 1] = -np.sin(theta)
    rot_mat[1, 0] = np.sin(theta)
    rot_mat[1, 1] = np.cos(theta)

    tip_size = [size[0] / 6, size[1]]
    tip_color = [0, 0, 0, color[3]]

    # rotate and translate the tip
    tip_rot_mat = np.eye(3)
    tip_rot_mat[0, 0] = np.cos(theta)
    tip_rot_mat[0, 1] = -np.sin(theta)
    tip_rot_mat[1, 0] = np.sin(theta)
    tip_rot_mat[1, 1] = np.cos(theta)
    tip_center = [
        x + size[0] / 2 * np.cos(theta),
        y + size[0] / 2 * np.sin(theta),
        0.001,
    ]
    tip_center = np.array(tip_center, dtype=np.float64)
    tip_rot_mat = np.array(tip_rot_mat, dtype=np.float64)

    render_plane(scene, center=center, size=size, rot_mat=rot_mat, rgba=color)
    render_plane(
        scene,
        center=tip_center,
        size=tip_size,
        rot_mat=tip_rot_mat,
        rgba=tip_color,
    )


class ApproachSimulator:
    def __init__(
        self, mjinfer: MjInfer, mapper: FootstepsMapper, footsteps_net: FootstepsNet
    ):
        self.mjinfer = mjinfer
        self.mapper = mapper
        self.footsteps_net = footsteps_net
        self.traj = Trajectory(footsteps_net)

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
            arrived = False

            params = self.mjinfer.make_humanoid_parameters()
            params.walk_max_dx_forward = 0.14
            params.walk_max_dx_backward = 0.05
            params.walk_max_dy = 0.1
            params.walk_max_dtheta = np.deg2rad(55)

            while not arrived:
                self.mjinfer.viewer.user_scn.ngeom = (
                    0  # Clear previous custom geometries
                )
                self.draw_footstep(T_world_target)
                T_world_left = self.mjinfer.get_T_world_site("left_foot")
                T_world_right = self.mjinfer.get_T_world_site("right_foot")

                T_right_target = np.linalg.inv(T_world_right) @ T_world_target
                error_pos = np.linalg.norm(T_right_target[:2, 3])
                error_yaw = abs(np.arctan2(T_right_target[1, 0], T_right_target[0, 0]))
                arrived = error_pos < 5e-2 and error_yaw < np.deg2rad(5)
                if arrived:
                    break

                trajectory = self.traj.sample_trajectory(
                    params,
                    T_world_left,
                    T_world_right,
                    self.mjinfer.support,
                    T_world_target,
                    "right",
                )
                self.traj.draw_trajectory(self.mjinfer.viewer.user_scn, trajectory)
                step = trajectory[0]
                dx, dy, dtheta = step["dx"], step["dy"], step["dtheta"]
                if USE_FOOTSTEP_REWARD:
                    vx, vy, vtheta = dx, dy, dtheta
                else:
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
