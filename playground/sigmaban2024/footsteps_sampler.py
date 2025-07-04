import numpy as np
import json
import tqdm
import argparse

from playground.sigmaban2024.mujoco_infer import MjInfer


class FootstepsSamples:
    """
    Footstep samples dataset obtained by the sampler
    """

    def __init__(self):
        # List containing footsteps
        self.footsteps: list[dict] = []

    def add(self, support: str, command: tuple, footstep: tuple):
        """
        Adds a sample

        :param support: support ("left" or "right")
        :param command: command (vx, vy, vtheta)
        :param footstep: footstep (dx, dy, dtheta)
        """
        self.footsteps.append(
            {"support": support, "command": command, "footstep": footstep}
        )

    def save(self, filename: str):
        """
        Saves the samples to JSON file

        :param filename: json filename
        """
        with open(filename, "w") as f:
            json.dump(self.footsteps, f)

    def load(self, filename):
        """
        Loads samples from json

        :param filename: json filename
        """
        if filename:
            with open(filename, "r") as f:
                self.footsteps = json.load(f)

    def get_data(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve data for a given side

        :param side: side ("left" or "right")
        :return: a list of (commands, footsteps), each being respectively (vx, vy, vtheta) or (dx, dy, dtheta)
        """

        return (
            np.array(
                [
                    entry["command"]
                    for entry in self.footsteps
                    if entry["support"] == side
                ]
            ).copy(),
            np.array(
                [
                    entry["footstep"]
                    for entry in self.footsteps
                    if entry["support"] == side
                ]
            ).copy(),
        )

    def prepare(self) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Pre-process dataset. Computes the feet spacing and offset footstep dy.

        :return: a tuple of (feet_spacing, commands, footsteps)
        """
        commands_right, footsteps_right = self.get_data("right")
        commands_left, footsteps_left = self.get_data("left")

        feet_spacing = (
            np.mean(footsteps_right[:, 1]) - np.mean(footsteps_left[:, 1])
        ) / 2

        footsteps_right[:, 1] -= feet_spacing
        footsteps_left[:, 1] += feet_spacing

        commands = np.vstack((commands_left, commands_right))
        footsteps = np.vstack((footsteps_left, footsteps_right))

        return feet_spacing, commands, footsteps


class FootstepsSampler:
    """
    Gets the robot walking and sample footsteps (dx, dy, dtheta) vs commands (vx, vy, vtheta)
    """

    def __init__(self, mjinfer: MjInfer):
        # Mjinfer simulation
        self.mjinfer: MjInfer = mjinfer

        # Number of footsteps taken at the begining
        self.warmup_footsteps: int = 3

        # Data generated
        self.samples = FootstepsSamples()

    def reset(self):
        """
        Resets the simulation and sample a random linear velocity
        """
        self.mjinfer.reset()
        self.sample_command()

    def sample_command(self):
        """
        Sample a random command
        """

        command_x = float(np.random.uniform(*self.mjinfer.COMMANDS_RANGE_X))
        command_y = float(np.random.uniform(*self.mjinfer.COMMANDS_RANGE_Y))
        command_theta = float(np.random.uniform(*self.mjinfer.COMMANDS_RANGE_THETA))

        self.mjinfer.commands = [
            command_x,
            command_y,
            command_theta,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def compute_footstep(
        self, support_foot: str, landing_foot: str
    ) -> tuple[float, float, float]:
        """
        Computes the footstep (dx, dy, dtheta) in MuJoCo from given foot frame to the other

        :param support_foot: support foot frame
        :param landing_foot: landing foot frame
        :return: a tuple (dx, dy, dtheta)
        """
        T_support_landing = np.linalg.inv(
            self.mjinfer.get_T_world_site(support_foot)
        ) @ self.mjinfer.get_T_world_site(landing_foot)

        dx = T_support_landing[0, 3]
        dy = T_support_landing[1, 3]
        dtheta = np.arctan2(T_support_landing[1, 0], T_support_landing[0, 0])

        return float(dx), float(dy), float(dtheta)

    def sample(self, samples: int = 1):
        """
        Sample a footstep. The robot walks during a warmup period, and the left+right footsteps are then
        measured in the simulator

        :param samples: number of consecutive samples to take, defaults to 1
        """
        # Letting simulation stabilize
        for _ in range(self.warmup_footsteps):
            self.mjinfer.walk_one_step()

        # Waiting for the left foot to reach the ground
        while self.mjinfer.support != "left":
            self.mjinfer.walk_one_step()

        for _ in range(samples):
            dx, dy, dtheta = self.compute_footstep("right_foot", "left_foot")
            self.samples.add("right", self.mjinfer.commands[:3], [dx, dy, dtheta])
            self.mjinfer.walk_one_step()

            dx, dy, dtheta = self.compute_footstep("left_foot", "right_foot")
            self.samples.add("left", self.mjinfer.commands[:3], [dx, dy, dtheta])
            self.mjinfer.walk_one_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/sigmaban2024/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/sigmaban2024/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="footsteps.json",
    )
    parser.add_argument("--view", action="store_true", default=False)
    args = parser.parse_args()

    mjinfer = MjInfer(args.model_path, args.reference_data, args.onnx_model_path)
    if args.view:
        mjinfer.enable_viewer(key=False)

    sampler = FootstepsSampler(mjinfer)

    for n in tqdm.tqdm(range(args.n_samples)):
        sampler.reset()
        sampler.sample()

    print(f"Writing data to footsteps.json")
    sampler.samples.save(args.output)
