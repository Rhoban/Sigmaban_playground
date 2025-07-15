import jax.numpy as jp
import json

# contains only one reference motion


class EpisodicReferenceMotion:
    def __init__(self, ref_motion_path: str):
        self.ref_motion = json.load(open(ref_motion_path, "r"))
        # self.frames = jp.array(self.ref_motion["Frames"])
        self.nb_steps = self.ref_motion["nb_steps"]
        frames = self.ref_motion["Frames"]
        self.frames = jp.array([list(frames[i].values()) for i in range(len(frames))])

    def get_frame(self, i):
        return self.frames[i]
