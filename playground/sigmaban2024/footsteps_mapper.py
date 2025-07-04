import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


class FootstepsMapper:
    def __init__(self, filename: str | None = None):

        self.feet_spacing = None
        self.M = None
        self.trained = False

        if filename:
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

    def remap(self, dx, dy, dtheta):
        return self.M @ self.model(dx, dy, dtheta)

    def prepare(self):
        commands_right, footsteps_right = self.get_data("right")
        commands_left, footsteps_left = self.get_data("left")

        self.feet_spacing = (
            np.mean(footsteps_right[:, 1]) - np.mean(footsteps_left[:, 1])
        ) / 2

        footsteps_right[:, 1] -= self.feet_spacing
        footsteps_left[:, 1] += self.feet_spacing

        commands = np.vstack((commands_left, commands_right))
        footsteps = np.vstack((footsteps_left, footsteps_right))

        return commands, footsteps

    def save(self, filename: str = "footsteps_mapping.json"):
        with open(filename, "w") as f:
            json.dump({"feet_spacing": self.feet_spacing, "M": self.M.tolist()}, f)

    def load(self, filename: str):
        with open(filename, "r") as f:
            data = json.load(f)
            self.feet_spacing = data["feet_spacing"]
            self.M = np.array(data["M"])

    def fit(self):
        commands, footsteps = self.prepare()

        A = [self.model(*footstep) for footstep in footsteps]
        b = commands

        result = np.linalg.lstsq(A, b)

        self.M = result[0].T
        self.save()

        print(f"Residuals: {result[1]}")
        print("")
        print(f"Average feet spacing is {self.feet_spacing}")
        print(f"Mapping matrix: ")
        print(self.M)

        print("")
        rows, cols = self.M.shape
        print(f"// Feet spacing: {self.feet_spacing}")
        print(f"Eigen::MatrixXd remapping({rows}, {cols});")
        row_str = "remapping << "
        for row in range(rows):
            for col in range(cols):
                row_str += str(self.M[row, col])
                if row == rows - 1 and col == cols - 1:
                    row_str += ";"
                else:
                    row_str += ", "
            print(row_str)
            row_str = ""

        # print(self.remap(0.0, 0.03, 0.0))

        print("")
        self.trained = True

    def evaluate(self):
        commands, footsteps = self.prepare()
        commands_pred = np.array([self.remap(*f) for f in footsteps])

        mse = np.mean((commands - commands_pred) ** 2)
        print(f"Model MSE: {mse}")

    def show_plot(self):
        # fig, axs = plt.subplots(3, 2)

        # for k, side in enumerate(["left", "right"]):
        #     commands, footsteps = self.get_data(side)

        #     axs[0][k].set_title(f"Support {side}")
        #     axs[0][k].scatter(commands[:, 0], footsteps[:, 0], label="dx", s=0.1)
        #     if self.trained:
        #         dx_range = np.min(footsteps[:, 0]), np.max(footsteps[:, 0])
        #         dxs = np.linspace(*dx_range, 100)
        #         xvels = np.array([self.remap(dx, 0, 0) for dx in dxs])
        #         axs[0][k].plot(xvels[:, 0], dxs, label="remap", c='orange')
        #     axs[0][k].set_xlabel("velocity x")
        #     axs[0][k].grid()
        #     axs[0][k].legend()

        #     axs[1][k].scatter(commands[:, 1], footsteps[:, 1], label="dy", s=0.1)
        #     offset = self.feet_spacing if side == "right" else -self.feet_spacing
        #     if self.trained:
        #         dy_range = np.min(footsteps[:, 1] - offset), np.max(footsteps[:, 1] - offset)
        #         dys = np.linspace(*dy_range, 100)
        #         yvels = np.array([self.remap(0, dy, 0) for dy in dys])
        #         axs[1][k].plot(yvels[:, 1], dys + offset, label="remap", c='orange')
        #     axs[1][k].set_xlabel("velocity y")
        #     axs[1][k].grid()
        #     axs[1][k].legend()

        #     axs[2][k].scatter(commands[:, 2], footsteps[:, 2], label="dtheta", s=0.1)
        #     if self.trained:
        #         dtheta_range = np.min(footsteps[:, 2]), np.max(footsteps[:, 2])
        #         dthetas = np.linspace(*dtheta_range, 100)
        #         thetavels = np.array([self.remap(0, 0, dtheta) for dtheta in dthetas])
        #         axs[2][k].plot(thetavels[:, 2], dthetas, label="remap", c='orange')
        #     axs[2][k].set_xlabel("velocity theta")
        #     axs[2][k].grid()
        #     axs[2][k].legend()

        commands, footsteps = self.prepare()

        fig, axs = plt.subplots(3, 1)
        axs[0].scatter(commands[:, 0], footsteps[:, 0], label="dx", s=0.1)
        if self.trained:
            dx_range = np.min(footsteps[:, 0]), np.max(footsteps[:, 0])
            dxs = np.linspace(*dx_range, 100)
            xvels = np.array([self.remap(dx, 0, 0) for dx in dxs])
            axs[0].plot(xvels[:, 0], dxs, label="remap", c="orange")
        axs[0].set_xlabel("velocity x")
        axs[0].grid()
        axs[0].legend()

        axs[1].scatter(commands[:, 1], footsteps[:, 1], label="dy", s=0.1)
        if self.trained:
            dy_range = np.min(footsteps[:, 1]), np.max(footsteps[:, 1])
            dys = np.linspace(*dy_range, 100)
            yvels = np.array([self.remap(0, dy, 0) for dy in dys])
            axs[1].plot(yvels[:, 1], dys, label="remap", c="orange")
        axs[1].set_xlabel("velocity y")
        axs[1].grid()
        axs[1].legend()

        axs[2].scatter(commands[:, 2], footsteps[:, 2], label="dtheta", s=0.1)
        if self.trained:
            dtheta_range = np.min(footsteps[:, 2]), np.max(footsteps[:, 2])
            dthetas = np.linspace(*dtheta_range, 100)
            thetavels = np.array([self.remap(0, 0, dtheta) for dtheta in dthetas])
            axs[2].plot(thetavels[:, 2], dthetas, label="remap", c="orange")
        axs[2].set_xlabel("velocity theta")
        axs[2].grid()
        axs[2].legend()

        plt.tight_layout()
        plt.legend()
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--footsteps",
        type=str,
        default="footsteps.json",
    )
    parser.add_argument("--plot", action="store_true", default=False)
    args = parser.parse_args()

    mapper = FootstepsMapper(args.footsteps)

    mapper.fit()

    if args.plot:
        mapper.show_plot()
