import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


class FootstepsMapper:
    def __init__(self, filename: str):
        with open(filename, "r") as f:
            self.footsteps = json.load(f)

    def model(self, dx, dy, dtheta):
        return [
            # bias
            1,
            
            # 1 term
            dx,
            dy,
            dtheta,

            # 2 terms
            # dx**2,
            # dx * dy,
            # dx * dtheta,
            # dy**2,
            # dy * dtheta,
            # dtheta**2,

            # 3 terms
            # dx**3,
            # dx**2 * dy,
            # dx**2 * dtheta,
            # dx * dy**2,
            # dx * dtheta**2,
            # dx * dy * dtheta,
            # dy**3,
            # dy**2 * dtheta,
            # dy * dtheta**2,
            # dtheta**3,
        ]

    def get_data(self, side):
        """
        Get data for a given side, return commands (n, 3) and footsteps (n, 3)
        """
        return np.array(
            [entry["command"] for entry in self.footsteps if entry["support"] == side]
        ), np.array(
            [entry["footstep"] for entry in self.footsteps if entry["support"] == side]
        )

    def fit(self):
        commands_right, footsteps_right = self.get_data("right")
        commands_left, footsteps_left = self.get_data("left")

        feet_spacing = (np.mean(footsteps_right[:, 1]) - np.mean(footsteps_left[:, 1]))/2
        print(f"Average feet spacing is {feet_spacing}")

        footsteps_right[:, 1] -= feet_spacing
        footsteps_left[:, 1] += feet_spacing

        commands = np.vstack((commands_left, commands_right))
        footsteps = np.vstack((footsteps_left, footsteps_right))

        A = [self.model(*footstep) for footstep in footsteps]
        b = commands

        result = np.linalg.lstsq(A, b)
        
        x = result[0].T
        with open("footsteps_mapping.json", "w") as f:
            json.dump(x.tolist(), f)
        
        print(f"Residuals: {result[1]}")
        print(x)

        print(x @ self.model(0.0, -0.03, 0.0))

        # for command, footstep in zip(commands, footsteps):
        #     print(f"===")
        #     print(f"Command: {command}")
        #     print(f"Footstep: {footstep}")
        #     predicted_command = self.model(*footstep) @ x
        #     print(f"Predicted command: {predicted_command}")

    def show_plot(self):
        fig, axs = plt.subplots(3, 2)

        for k, side in enumerate(["left", "right"]):
            commands, footsteps = self.get_data(side)

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

mapper.fit()
