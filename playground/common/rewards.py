"""
Set of commonly used rewards
For examples on how to use some rewards, look at https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py
"""

import jax
import jax.numpy as jp


# Tracking rewards.
def reward_tracking_lin_vel(
    commands: jax.Array,
    local_vel: jax.Array,
    tracking_sigma: float,
) -> jax.Array:
    # lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    # return jp.nan_to_num(jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma))
    y_tol = 0.1
    error_x = jp.square(commands[0] - local_vel[0])
    error_y = jp.clip(jp.abs(local_vel[1] - commands[1]) - y_tol, 0.0, None)
    lin_vel_error = error_x + jp.square(error_y)
    return jp.nan_to_num(jp.exp(-lin_vel_error / tracking_sigma))


def reward_tracking_footsteps(
    command: jax.Array,
    T_world_left: jax.Array,
    T_world_right: jax.Array,
    support_was_left: jax.Array,
    support_changed: jax.Array,
    feet_spacing,
) -> jax.Array:

    # Target footstep, expressed in support foot
    y_offset = jp.where(support_was_left, -feet_spacing, feet_spacing)
    T_support_target = jp.array(
        [
            [jp.cos(command[2]), -jp.sin(command[2]), 0, command[0]],
            [jp.sin(command[2]), jp.cos(command[2]), 0, command[1] + y_offset],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Computing placement error
    T_world_support = jp.where(support_was_left, T_world_left, T_world_right)
    T_world_flying = jp.where(support_was_left, T_world_right, T_world_left)
    T_target_flying = jp.linalg.inv(T_world_flying) @ T_world_support @ T_support_target

    pos_error = jp.linalg.norm(T_target_flying[:2, 3])
    yaw_error = jp.abs(jp.arctan2(T_target_flying[1, 0], T_target_flying[0, 0]))
    error = 5 * pos_error + yaw_error

    placement_reward = jp.exp(-error * 5)

    return jp.where(support_changed, placement_reward, 0)


def reward_tracking_ang_vel(
    commands: jax.Array,
    ang_vel: jax.Array,
    tracking_sigma: float,
) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.nan_to_num(jp.exp(-ang_vel_error / tracking_sigma))


# Base-related rewards.


def cost_lin_vel_z(global_linvel) -> jax.Array:
    return jp.nan_to_num(jp.square(global_linvel[2]))


def cost_ang_vel_xy(global_angvel) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(global_angvel[:2])))


def cost_orientation(torso_zaxis: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(torso_zaxis[:2])))


def cost_base_height(base_height: jax.Array, base_height_target: float) -> jax.Array:
    return jp.nan_to_num(jp.square(base_height - base_height_target))


def reward_base_y_swing(
    base_y_speed: jax.Array,
    freq: float,
    amplitude: float,
    t: float,
    tracking_sigma: float,
) -> jax.Array:
    target_y_speed = amplitude * jp.sin(2 * jp.pi * freq * t)
    y_speed_error = jp.square(target_y_speed - base_y_speed)
    return jp.nan_to_num(jp.exp(-y_speed_error / tracking_sigma))


# Energy related rewards.


def cost_torques(torques: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(torques)))
    # return jp.nan_to_num(jp.sum(jp.abs(torques)))


def cost_energy(qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator)))


def cost_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
    c1 = jp.nan_to_num(jp.sum(jp.square(act - last_act)))
    return c1


# Other rewards.


def cost_joint_pos_limits(
    qpos: jax.Array, soft_lowers: float, soft_uppers: float
) -> jax.Array:
    out_of_limits = -jp.clip(qpos - soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - soft_uppers, 0.0, None)
    return jp.nan_to_num(jp.sum(out_of_limits))


def cost_stand_still(
    commands: jax.Array,
    qpos: jax.Array,
    qvel: jax.Array,
    default_pose: jax.Array,
    ignore_head: bool = False,
) -> jax.Array:
    # TODO no hard coded slices
    cmd_norm = jp.linalg.norm(commands[:3])
    if not ignore_head:
        pose_cost = jp.sum(jp.abs(qpos - default_pose))
        vel_cost = jp.sum(jp.abs(qvel))
    else:
        left_leg_pos = qpos[:5]
        right_leg_pos = qpos[9:]
        left_leg_vel = qvel[:5]
        right_leg_vel = qvel[9:]
        left_leg_default = default_pose[:5]
        right_leg_default = default_pose[9:]
        pose_cost = jp.sum(jp.abs(left_leg_pos - left_leg_default)) + jp.sum(
            jp.abs(right_leg_pos - right_leg_default)
        )
        vel_cost = jp.sum(jp.abs(left_leg_vel)) + jp.sum(jp.abs(right_leg_vel))

    return jp.nan_to_num(pose_cost + vel_cost) * (cmd_norm < 0.01)


def cost_termination(done: jax.Array) -> jax.Array:
    return done


def reward_alive() -> jax.Array:
    return jp.array(1.0)


# Pose-related rewards.


def cost_head_pos(
    joints_qpos: jax.Array,
    joints_qvel: jax.Array,
    cmd: jax.Array,
) -> jax.Array:
    move_cmd_norm = jp.linalg.norm(cmd[:3])
    head_cmd = cmd[3:]
    head_pos = joints_qpos[5:9]
    # head_vel = joints_qvel[5:9]

    # target_head_qvel = jp.zeros_like(head_cmd)

    head_pos_error = jp.sum(jp.square(head_pos - head_cmd))

    # head_vel_error = jp.sum(jp.square(head_vel - target_head_qvel))

    return jp.nan_to_num(head_pos_error) * (move_cmd_norm > 0.01)
    # return jp.nan_to_num(head_pos_error + head_vel_error)


# FIXME
def cost_joint_deviation_hip(
    qpos: jax.Array, cmd: jax.Array, hip_indices: jax.Array, default_pose: jax.Array
) -> jax.Array:
    cost = jp.sum(jp.abs(qpos[hip_indices] - default_pose[hip_indices]))
    cost *= jp.abs(cmd[1]) > 0.1
    return jp.nan_to_num(cost)


# FIXME
def cost_joint_deviation_knee(
    qpos: jax.Array, knee_indices: jax.Array, default_pose: jax.Array
) -> jax.Array:
    return jp.nan_to_num(
        jp.sum(jp.abs(qpos[knee_indices] - default_pose[knee_indices]))
    )


# FIXME
def cost_pose(
    qpos: jax.Array, default_pose: jax.Array, weights: jax.Array
) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(qpos - default_pose) * weights))


# Feet related rewards.


# FIXME
def cost_feet_slip(contact: jax.Array, global_linvel: jax.Array) -> jax.Array:
    body_vel = global_linvel[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return jp.nan_to_num(reward)


# FIXME
def cost_feet_clearance(
    feet_vel: jax.Array,
    foot_pos: jax.Array,
    max_foot_height: float,
) -> jax.Array:
    # feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    # foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - max_foot_height)
    return jp.nan_to_num(jp.sum(delta * vel_norm))


# FIXME
def cost_feet_height(
    swing_peak: jax.Array,
    first_contact: jax.Array,
    max_foot_height: float,
) -> jax.Array:
    error = swing_peak / max_foot_height - 1.0
    return jp.nan_to_num(jp.sum(jp.square(error) * first_contact))


# FIXME
def reward_feet_air_time(
    air_time: jax.Array,
    first_contact: jax.Array,
    commands: jax.Array,
    threshold_min: float = 0.1,  # 0.2
    threshold_max: float = 0.5,
) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands[:3])
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return jp.nan_to_num(reward)


# FIXME
def reward_feet_phase(
    foot_pos: jax.Array,
    rz: jax.Array,
) -> jax.Array:
    # Reward for tracking the desired foot height.
    # foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    # rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    # TODO(kevin): Ensure no movement at 0 command.
    # cmd_norm = jp.linalg.norm(commands)
    # reward *= cmd_norm > 0.1  # No reward for zero commands.
    return jp.nan_to_num(reward)
