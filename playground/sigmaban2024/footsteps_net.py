import numpy as np
import argparse

from playground.sigmaban2024.mujoco_infer import MjInfer
from playground.common.onnx_infer import OnnxInfer
import meshcat.transformations as tf


class FootstepsNet:
    def __init__(self, onnx: str, feet_spacing: float = 0.14):
        """
        Footsteps Net inference agent, using ONNX

        :param onnx: ONNX footsteps net to load
        :param feet_spacing: spacing between feet, constant used during training, defaults to 0.15
        """
        self.footstepsnet_infer = OnnxInfer(onnx, "input.1")
        self.feet_spacing: float = feet_spacing

        # Obstacle settings
        self.obstacle_position = [0.0, 0.0]
        self.obstacle_radius = 0.0

    def flatten_on_floor(self, T_world_frame: np.ndarray) -> np.ndarray:
        """
        Flattening a frame on the floor

        :param T_world_frame: _description_
        :return: _description_
        """
        x, y = T_world_frame[:2, 3]
        yaw = np.arctan2(T_world_frame[1, 0], T_world_frame[0, 0])

        return tf.translation_matrix((x, y, 0)) @ tf.rotation_matrix(yaw, (0, 0, 1))

    def infer(
        self,
        T_world_left: np.ndarray,
        T_world_right: np.ndarray,
        T_world_target: np.ndarray,
        support_side: str,
        target_side: str,
    ) -> np.ndarray:
        """
        Infers the next footstep (dx, dy, dtheta)

        :param T_world_left: Left foot pose
        :param T_world_right: Right foot
        :param T_world_target: Target to reach
        :param support_side: Current support side
        :param target_side: Side we want to reach the target
        :return: Next (dx, dy, dtheta) footstep
        """

        sym_sign = -1 if support_side == "left" else 1.0

        # Computing T_support_target transformation
        T_world_support = self.flatten_on_floor(
            T_world_left if support_side == "left" else T_world_right
        )
        T_support_target = np.linalg.inv(T_world_support) @ T_world_target

        observation = [
            T_support_target[0, 3],  # x
            T_support_target[1, 3] * sym_sign,  # y
            T_support_target[0, 0],  # cos(alpha)
            T_support_target[1, 0] * sym_sign,  # sin(alpha)
            1.0 if support_side == target_side else 0.0,
            self.obstacle_position[0],
            self.obstacle_position[1] * sym_sign,
            self.obstacle_radius,
        ]

        action = self.footstepsnet_infer.infer(np.array([observation]))[0]
        action[1:] *= sym_sign

        return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--footstepsnet_onnx", type=str, required=True)
    args = parser.parse_args()

    footsteps_net = FootstepsNet(args.footstepsnet_onnx)

    import meshcat.transformations as tf

    T_world_left = tf.translation_matrix((0.0, 0.0, 0.0))
    T_world_right = tf.translation_matrix((0.0, 0.15, 0.0))
    T_world_target = tf.translation_matrix((0.0, 1.0, 0.0))
    action = footsteps_net.infer(
        T_world_left, T_world_right, T_world_target, "left", "left"
    )

    print(action)
