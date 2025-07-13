import jax.numpy as jp
import json

# contains only one reference motion


class EpisodicReferenceMotion:
    def __init__(self, ref_motion_path: str):
        self.ref_motion = json.load(open(ref_motion_path, "r"))
        self.frames = jp.array(self.ref_motion["Frames"])
        self.nb_steps = len(self.ref_motion["Frames"])

    # ref_motion["Frames"][i]:
    # root_position
    # + root_orientation_quat
    # + joints_positions
    # + left_toe_pos
    # + right_toe_pos
    # + world_linear_vel
    # + world_angular_vel
    # + joints_vel
    # + left_toe_vel
    # + right_toe_vel
    # + foot_contacts
    def get_frame(self, i):
        # outputs [joints_pos, joints_vel, foot_contacts, world_linear_vel, world_angular_vel],

        joints_pos = self.frames[i][7 : 7 + 20]  # joints pos
        joints_vel = self.frames[i][39 : 39 + 20]  # joints vel
        foot_contacts = self.frames[i][-2:]  # foot contacts
        world_lin_vel_ang_vel = self.frames[i][
            33 : 33 + 6
        ]  # world linear vel + world angular vel
        frame = jp.concatenate(
            [
                joints_pos,
                joints_vel,
                foot_contacts,
                world_lin_vel_ang_vel,
            ]
        )
        return frame


if __name__ == "__main__":
    ERM = EpisodicReferenceMotion(
        "playground/open_duck_mini_v2/data/animation_data_leg_flexing.json"
    )
    for i in range(ERM.nb_steps):
        frame = ERM.get_frame(i)
        print(frame)
        exit()
