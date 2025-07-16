import json
import sys
import numpy as np
from scipy.interpolate import CubicSpline


class ReadSplines:
    def __init__(self, spline_file):
        self.spline_data = json.load(open(spline_file, "r"))

        self.trunk_z = self.spline_data["trunkZ"]
        self.use_steady = self.spline_data.get("useSteady", False)
        self.remap = self.spline_data.get("remap", None)
        self.kick = self.spline_data.get("kick", None)

        self.splines = {}
        self.max_ts = {}
        for k, v in self.spline_data.items():
            if k == "trunkZ" or k == "useSteady" or k == "remap" or k == "kick":
                continue
            angle = k != "com_y_traj"
            x = np.array([point[0] for point in v])
            if angle:
                y = np.array([np.deg2rad(point[1]) for point in v])
            else:
                y = np.array([point[1] for point in v])
            self.splines[k] = CubicSpline(x, y, bc_type="natural")
            self.max_ts[k] = x[-1]

        self.all_max_ts = 0
        for k in self.max_ts:
            if self.max_ts[k] > self.all_max_ts:
                self.all_max_ts = self.max_ts[k]

    def get_remap_factor(self, t):
        if self.remap is None:
            return 1.0

        for i in range(len(self.remap) - 1):
            if self.remap[i][0] <= t < self.remap[i + 1][0]:
                return self.remap[i + 1][1]

        if t >= self.remap[-1][0]:
            return self.remap[-1][1]

        return 1.0

    def get_spline_at(self, name, t):
        if name not in self.splines:
            raise ValueError(f"Spline '{name}' not found in the data.")
        if t > self.max_ts[name]:
            t = self.max_ts[name]
        if t < 0:
            t = 0
        return self.splines[name](t)

    def get_all_splines_at(self, names, t):
        all = {}
        for name in names:
            all[name] = float(self.get_spline_at(name, t))

        return all


if __name__ == "__main__":
    # plot spline "shoot_hip_pitch"
    import matplotlib.pyplot as plt

    rs = ReadSplines(sys.argv[1])

    spline_to_plot = "com_y_traj"
    # spline_to_plot = "shoot_hip_pitch"

    x = []
    y = []
    remap = []
    for i in range(7 * 100):
        t = i / 100
        x.append(t)
        y.append(rs.get_spline_at(spline_to_plot, t))

    plt.plot(x, y)
    plt.title(f"{spline_to_plot} Spline")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.grid()
    plt.show()
