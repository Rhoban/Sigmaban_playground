import numpy as np
import json
import tqdm
import argparse

from playground.sigmaban2024.mujoco_infer import MjInfer


class FootstepsSampler:
    def __init__(self, mjinfer: MjInfer):
        # Mjinfer simulation
        self.mjinfer: MjInfer = mjinfer

        # Number of footsteps taken at the begining
        self.warmup_footsteps: int = 3

        # Data generated
        self.footsteps = []

    def reset(self):
        """
        Resets the simulation and sample a random linear velocity
        """
        self.mjinfer.reset()
        self.sample_command()

    def sample_command(self):
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

    def compute_footstep(self, support_foot, landing_foot):
        """
        Compute the footstep (dx, dy, dtheta) in support foot from MuJoCo frames
        """
        T_support_landing = np.linalg.inv(
            self.mjinfer.get_T_world_site(support_foot)
        ) @ self.mjinfer.get_T_world_site(landing_foot)
        dx = T_support_landing[0, 3]
        dy = T_support_landing[1, 3]
        dtheta = np.arctan2(T_support_landing[1, 0], T_support_landing[0, 0])

        return float(dx), float(dy), float(dtheta)

    def sample(self, samples: int = 1):
        # Letting simulation stabilize
        for _ in range(self.warmup_footsteps):
            self.mjinfer.walk_one_step()

        # Waiting for the left foot to reach the ground
        while self.mjinfer.support != "left":
            self.mjinfer.walk_one_step()

        for _ in range(samples):
            dx, dy, dtheta = self.compute_footstep("right_foot", "left_foot")
            self.footsteps.append(
                {
                    "support": "right",
                    "command": self.mjinfer.commands[:3],
                    "footstep": [dx, dy, dtheta],
                }
            )
            self.mjinfer.walk_one_step()

            dx, dy, dtheta = self.compute_footstep("left_foot", "right_foot")
            self.footsteps.append(
                {
                    "support": "left",
                    "command": self.mjinfer.commands[:3],
                    "footstep": [dx, dy, dtheta],
                }
            )
            self.mjinfer.walk_one_step()

            # print(self.footsteps)
            # input()

    def save(self, filename: str):
        """
        Save footsteps data to filename
        """
        with open(filename, "w") as f:
            json.dump(self.footsteps, f)


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
sampler.save(args.output)
