import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


class FootstepsMapper:
    def __init__(self, filename: str):
        with open(filename, "r") as f:
            self.footsteps = json.load(f)

    def show_plot(self):
        fig, axs = plt.subplots(3, 2)

        for k, side in enumerate(["left", "right"]):
            commands = np.array(
                [
                    entry["command"]
                    for entry in self.footsteps
                    if entry["support"] == side
                ]
            )
            footsteps = np.array(
                [
                    entry["footstep"]
                    for entry in self.footsteps
                    if entry["support"] == side
                ]
            )
            axs[0][k].set_title(f"Support {side}")
            axs[0][k].scatter(commands[:, 0], footsteps[:, 0], label="dx", s=0.1)
            axs[0][k].set_xlabel("velocity x")
            axs[0][k].grid()
            axs[0][k].legend()

            axs[1][k].scatter(commands[:, 1], footsteps[:, 1], label="dy", s=0.1)
            axs[1][k].set_xlabel("velocity y")
            axs[1][k].grid()
            axs[1][k].legend()

            axs[2][k].scatter(commands[:, 2], footsteps[:, 2], label="dtheta", s=0.1)
            axs[2][k].set_xlabel("velocity theta")
            axs[2][k].grid()
            axs[2][k].legend()

        plt.tight_layout()
        plt.legend()
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--footsteps",
    type=str,
    default="footsteps.json",
)
parser.add_argument("--plot", action="store_true", default=False)
args = parser.parse_args()

mapper = FootstepsMapper(args.footsteps)

if args.plot:
    mapper.show_plot()
