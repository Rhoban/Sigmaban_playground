import json

# contains only one reference motion


class EpisodicReferenceMotion:
    def __init__(self, ref_motion_path: str):
        self.ref_motion = json.load(open(ref_motion_path, "r"))
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
        # outputs [joints_pos, joints_vel, foot_contacts, base_linear_vel, base_angular_vel],

        frame = []
        frame += self.ref_motion["Frames"][i][7 : 7 + 20]  # joints pos
        frame += self.ref_motion["Frames"][i][39 : 39 + 20]  # joints vel
        frame += self.ref_motion["Frames"][i][-2:]  # foot contacts
        frame += self.ref_motion["Frames"][i][
            33 : 33 + 6
        ]  # base linear vel + base angular vel
        return frame


if __name__ == "__main__":
    ERM = EpisodicReferenceMotion("/home/antoine/MISC/Open_Duck_reference_motion_generator/parkour.json")
    for i in range(ERM.nb_steps):
        frame = ERM.get_frame(i)
        print(frame)
        # print(type(frame[0]))
        # print(type(frame))
        # print(frame.dtype())
        # exit()
