import json

# contains only one reference motion


class EpisodicReferenceMotion:
    def __init__(self, ref_motion_path: str):
        self.ref_motion = json.load(open(ref_motion_path, "r"))
        self.nb_steps = self.ref_motion["nb_steps"]
        self.frames = self.ref_motion["Frames"]
        self.frames = [
            list(self.frames[i].values()) for i in range(len(self.frames))
        ]

    def get_frame(self, i):
        return self.frames[i]


if __name__ == "__main__":
    ERM = EpisodicReferenceMotion("sampled_splines.json")
    for i in range(ERM.nb_steps):
        frame = ERM.get_frame(i)
        print(frame)
        # print(type(frame[0]))
        # print(type(frame))
        # print(frame.dtype())
        # exit()
