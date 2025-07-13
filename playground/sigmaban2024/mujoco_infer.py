import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.sigmaban2024.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = False
MASK_HEAD_AND_ARMS = True
USE_FOOTSTEP_REWARD = False


class MjInfer(MJInferBase):
    def __init__(
        self,
        model_path: str,
        reference_data: str,
        onnx_model_path: str,
        save_obs: bool = False,
    ):
        super().__init__(model_path)

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 1.0
        self.action_scale = 1.0

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        self.PRM = PolyReferenceMotion(
            reference_data, convert_to_speeds=not USE_FOOTSTEP_REWARD
        )

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.COMMANDS_RANGE_X = [np.min(self.PRM.dxs), np.max(self.PRM.dxs)]
        self.COMMANDS_RANGE_Y = [np.min(self.PRM.dys), np.max(self.PRM.dys)]
        self.COMMANDS_RANGE_THETA = [np.min(self.PRM.dthetas), np.max(self.PRM.dthetas)]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0]

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])

        self.save_obs = save_obs
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        self.viewer = None

        self.random_head_and_arm_position = (np.random.random(8) - 0.5) * 2
        # self.random_head_and_arm_position = np.zeros(8)
        # print(self.random_head_and_arm_position)
        # exit()

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def make_humanoid_parameters(self):
        import placo

        humanoid_parameters = placo.HumanoidParameters()

        humanoid_parameters.foot_length = self.PRM.placo_parameters["foot_length"]
        humanoid_parameters.foot_width = self.PRM.placo_parameters["foot_width"]
        humanoid_parameters.feet_spacing = self.PRM.placo_parameters["feet_spacing"]
        humanoid_parameters.walk_max_dx_forward = self.PRM.placo_parameters[
            "walk_max_dx_forward"
        ]
        humanoid_parameters.walk_max_dx_backward = self.PRM.placo_parameters[
            "walk_max_dx_backward"
        ]
        humanoid_parameters.walk_max_dy = self.PRM.placo_parameters["walk_max_dy"]
        humanoid_parameters.walk_max_dtheta = self.PRM.placo_parameters[
            "walk_max_dtheta"
        ]

        return humanoid_parameters

    def get_feet_contacts(self, data):
        left_foot_cleat_back_left = self.check_contact(
            data, "left_foot_cleat_back_left", "floor"
        )
        left_foot_cleat_back_right = self.check_contact(
            data, "left_foot_cleat_back_right", "floor"
        )
        left_foot_cleat_front_left = self.check_contact(
            data, "left_foot_cleat_front_left", "floor"
        )
        left_foot_cleat_front_right = self.check_contact(
            data, "left_foot_cleat_front_right", "floor"
        )
        right_foot_cleat_back_left = self.check_contact(
            data, "right_foot_cleat_back_left", "floor"
        )
        right_foot_cleat_back_right = self.check_contact(
            data, "right_foot_cleat_back_right", "floor"
        )
        right_foot_cleat_front_left = self.check_contact(
            data, "right_foot_cleat_front_left", "floor"
        )
        right_foot_cleat_front_right = self.check_contact(
            data, "right_foot_cleat_front_right", "floor"
        )
        left_contact = (
            left_foot_cleat_back_left
            or left_foot_cleat_back_right
            or left_foot_cleat_front_left
            or left_foot_cleat_front_right
        )
        right_contact = (
            right_foot_cleat_back_left
            or right_foot_cleat_back_right
            or right_foot_cleat_front_left
            or right_foot_cleat_front_right
        )

        # left_contact = self.check_contact(data, "left_foot___list_t0v6opd9rekumc_default", "floor")
        # right_contact = self.check_contact(data, "right_foot_", "floor")
        return left_contact, right_contact

    def set_command(self, vx: float, vy: float, vtheta: float):
        self.commands = [vx, vy, vtheta]

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3

        # BEFORE
        # gravity = np.array(data.site_xmat[self.get_body_id_from_name("trunk")]).T.reshape((3, 3)) @ np.array(
        #     [0, 0, -1]
        # )

        # AFTER
        gravity = np.array(data.site_xmat[self.get_site_id_from_name("trunk")]).reshape(
            (3, 3)
        ).T @ np.array([0, 0, -1])

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        if MASK_HEAD_AND_ARMS:
            joint_angles = joint_angles[8:]
            joint_vel = joint_vel[8:]

        # add noise to joint vel
        # joint_vel += np.random.random(20)*1.5

        contacts = self.get_feet_contacts(data)
        # contacts = [1., 1.]

        linvel = self.get_linvel(data)

        home_offset = self.default_actuator
        if MASK_HEAD_AND_ARMS:
            home_offset = home_offset[8:]
        obs = np.concatenate(
            [
                # linvel,
                gyro,
                # accelerometer,
                gravity,
                command,
                joint_angles - home_offset,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                (
                    self.motor_targets
                    if not MASK_HEAD_AND_ARMS
                    else self.motor_targets[8:]
                ),
                contacts,
                self.imitation_phase,
            ]
        )

        return obs

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        self.random_head_and_arm_position = (np.random.random(8) - 0.5) * 2
        if keycode == 265:  # arrow up
            lin_vel_x = self.COMMANDS_RANGE_X[1]
        if keycode == 264:  # arrow down
            lin_vel_x = self.COMMANDS_RANGE_X[0]
        if keycode == 263:  # arrow left
            lin_vel_y = self.COMMANDS_RANGE_Y[1]
        if keycode == 262:  # arrow right
            lin_vel_y = self.COMMANDS_RANGE_Y[0]
        if keycode == 81:  # a
            ang_vel = self.COMMANDS_RANGE_THETA[1]
        if keycode == 69:  # e
            ang_vel = self.COMMANDS_RANGE_THETA[0]
        if keycode == 80:  # p
            self.data.qvel[:2] = [1.0, 0]
            # self.phase_frequency_factor += 0.1
        if keycode == 59:  # m
            self.data.qvel[:2] = [-1.0, 0]
            # self.phase_frequency_factor -= 0.1
            # self.random_head_and_arm_position = (np.random.random(8)-0.5)*2
        if keycode == 82:  # r
            self.reset()

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel

    def get_T_world_site(self, site_name: str) -> np.ndarray:
        """
        Gets the transformation from world to site frame.

        Args:
            site_name (str): site name
        """
        T = np.eye(4)
        site = self.data.site(site_name)
        T[:3, :3] = site.xmat.reshape(3, 3)
        T[:3, 3] = site.xpos

        return T

    def walk_one_step(self):
        """
        Walks until the next support is reached
        """
        current_support = self.support

        while self.support == current_support:
            self.step()

    def reset(self):
        self.counter = 0
        self.t = 0
        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_actuator
        self.random_head_and_arm_position = (np.random.random(8) - 0.5) * 2

        self.support = "left"

    def step(self):
        step_start = time.time()

        mujoco.mj_step(self.model, self.data)

        # TODO: Move this somewhere else (calibrating neutral feet spacing)
        # T_left_right = np.linalg.inv(
        #     self.get_T_world_site("left_foot")
        # ) @ self.get_T_world_site("right_foot")
        # print(f"Right position inl left frame: {T_left_right[:3, 3]}")

        self.counter += 1
        self.t += self.model.opt.timestep

        if self.counter % self.decimation == 0:
            # if np.linalg.norm(self.commands) > 0.0:
            self.imitation_i += 1.0 * self.phase_frequency_factor

            self.support = (
                "left"
                if self.imitation_i < self.PRM.nb_steps_in_period / 2
                else "right"
            )

            self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period

            # else:
            #     self.imitation_i = 0.0
            # print(self.PRM.nb_steps_in_period)
            # exit()
            self.imitation_phase = np.array(
                [
                    np.cos(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                    np.sin(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                ]
            )
            obs = self.get_obs(
                self.data,
                self.commands,
            )

            self.saved_obs.append(obs)
            action = self.policy.infer(obs)

            # self.action_filter.push(action)
            # action = self.action_filter.get_filtered_action()

            self.last_last_last_action = self.last_last_action.copy()
            self.last_last_action = self.last_action.copy()
            self.last_action = action.copy()

            self.motor_targets = self.default_actuator + action * self.action_scale

            if USE_MOTOR_SPEED_LIMITS:
                self.motor_targets = np.clip(
                    self.motor_targets,
                    self.prev_motor_targets
                    - self.max_motor_velocity * (self.sim_dt * self.decimation),
                    self.prev_motor_targets
                    + self.max_motor_velocity * (self.sim_dt * self.decimation),
                )

                self.prev_motor_targets = self.motor_targets.copy()

            # head_targets = self.commands[3:]
            if MASK_HEAD_AND_ARMS:
                self.motor_targets[:8] = self.random_head_and_arm_position
            self.data.ctrl = self.motor_targets.copy()
            # self.data.ctrl = np.zeros(20)

        if self.viewer is not None:
            self.viewer.sync()

            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def enable_viewer(self, key: bool = True):
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback if key else None,
        )

    def run(self):
        try:
            self.reset()
            self.enable_viewer()

            while True:
                self.step()
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
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
    parser.add_argument("--save-obs", action="store_true", default=False)

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path,
        args.reference_data,
        args.onnx_model_path,
        args.save_obs,
    )
    mjinfer.run()
