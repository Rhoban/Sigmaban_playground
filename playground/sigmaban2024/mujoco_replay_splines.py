import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse

from playground.sigmaban2024.mujoco_infer_base import MJInferBase
from playground.sigmaban2024.read_spline import ReadSplines

USE_MOTOR_SPEED_LIMITS = False
MASK_HEAD_AND_ARMS = True


class MjInfer(MJInferBase):
    def __init__(
        self,
        model_path: str,
        support="left",
    ):
        super().__init__(model_path)
        self.read_splines = ReadSplines(
            "/home/antoine/Rhoban/Sigmaban_playground/kick.json"
        )
        self.viewer = None
        self.counter = 0
        self.support = support
        self.shoot = "right" if self.support == "left" else "left"
        self.actuators_used = [
            name
            for name in list(self.read_splines.splines.keys())
            if "contact" not in name
        ]

        self.actuators_used_ids = []
        for name in self.actuators_used:
            nname = name.replace("shoot", self.shoot).replace("support", self.support)
            self.actuators_used_ids.append(self.get_actuator_id_from_name(nname))

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        if keycode == 82:  # r
            self.reset()

    def reset(self):
        print("reset")
        self.counter = 0
        self.t = -1
        self.data.qpos[:] = self.model.keyframe("home").qpos
        # zero = np.zeros_like(self.data.qpos)
        # zero[2] = 0.4
        # self.data.qpos[:] = zero
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_actuator

    def step(self):
        step_start = time.time()

        mujoco.mj_step(self.model, self.data)

        self.counter += 1
        remap_factor = self.read_splines.get_remap_factor(self.t)
        self.t += self.model.opt.timestep * remap_factor

        # self.data.qpos[2] = 0.5

        if self.counter % self.decimation == 0:
            all_q = self.read_splines.get_all_splines_at(self.actuators_used, self.t)
            all_q = list(all_q.values())
            left_foot_contact = self.read_splines.get_spline_at("left_foot_contact", self.t) > 1e-5
            right_foot_contact = self.read_splines.get_spline_at("right_foot_contact", self.t) > 1e-5
            print(left_foot_contact, right_foot_contact)
            self.data.ctrl = self.model.keyframe("home").ctrl
            self.data.ctrl[self.actuators_used_ids] += all_q

            if self.t > self.read_splines.max_ts["support_hip_roll"]:
                self.reset()

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
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/sigmaban2024/xmls/scene_flat_terrain_backlash_shoot.xml",
    )

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path,
    )
    mjinfer.run()
