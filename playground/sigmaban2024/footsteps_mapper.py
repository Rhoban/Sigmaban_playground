import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

from playground.sigmaban2024.footsteps_sampler import FootstepsSamples


class FootstepsMapper:
    """
    Least-square mapper
    """

    def __init__(self):
        # Feet spacing obtained from samples
        self.feet_spacing: float | None = None

        # Remapping matrix
        self.M: np.ndarray | None = None

        self.trained: bool = False

    def model(self, dx, dy, dtheta):
        """
        Least-square model.

        :param dx: footstep dx
        :param dy: footstep dy
        :param dtheta: footstep dtheta
        :return: vector used for least-square
        """
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

    def remap(self, dx, dy, dtheta):
        """
        Remaps a footstep into target velocity

        :param dx: footstep dx
        :param dy: footstep dy
        :param dtheta: footstep dtheta
        :return: target velocity
        """
        return self.M @ self.model(dx, dy, dtheta)

    def save(self, filename: str = "footsteps_mapping.json"):
        """
        Save the footstep model to json file

        :param filename: json file, defaults to "footsteps_mapping.json"
        """
        with open(filename, "w") as f:
            json.dump({"feet_spacing": self.feet_spacing, "M": self.M.tolist()}, f)

    def load(self, filename: str):
        """
        Loads footstep model from json file

        :param filename: json file
        """
        with open(filename, "r") as f:
            data = json.load(f)
            self.feet_spacing = data["feet_spacing"]
            self.M = np.array(data["M"])

    def fit(self, samples: FootstepsSamples):
        """
        Fit given samples using least squares

        :param samples: footstep samples
        """
        self.feet_spacing, commands, footsteps = samples.prepare()

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
        self.trained = True

        self.print_cpp()

    def print_cpp(self):
        """
        Print C++ code to embed the footsteps remapping, using Eigen
        """

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

        print("")

    def evaluate(self):
        """
        Computes the empirical model MSE
        """
        _, commands, footsteps = self.prepare()
        commands_pred = np.array([self.remap(*f) for f in footsteps])

        mse = np.mean((commands - commands_pred) ** 2)
        print(f"Model MSE: {mse}")

    def show_plot(self, samples: FootstepsSamples):
        """
        Displays a plot showing the samples. If the remap is trained, it will be overlayed for reference.
        """
        _, commands, footsteps = samples.prepare()

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
    parser.add_argument(
        "-w", "--write_metadata_in_onnx", action="store_true", default=True
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to the ONNX model file.",
        default=None,
    )
    args = parser.parse_args()

    if args.write_metadata_in_onnx:
        assert args.model_path is not None, (
            "If write_metadata_in_onnx is True, you must provide the path to the onnx model file in which to write"
        )

    mapper = FootstepsMapper()

    samples = FootstepsSamples()
    samples.load(args.footsteps)
    mapper.fit(samples)
    matrix = [float(x) for x in mapper.M.flatten()]
    if args.write_metadata_in_onnx:
        import onnx
        model_onnx = onnx.load(args.model_path)
        data_json = json.loads(model_onnx.metadata_props[0].value)
        data_json["feet_spacing"] = mapper.feet_spacing
        data_json["mapping_matrix"] = matrix
        metadata = model_onnx.metadata_props[0]
        metadata.key = "metadata"
        metadata.value = json.dumps(data_json)


        onnx.save(model_onnx, args.model_path)



    if args.plot:
        mapper.show_plot(samples)
