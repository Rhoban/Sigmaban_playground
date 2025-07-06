import numpy as np
import pickle


class PolyReferenceMotion:
    def __init__(self, polynomial_coefficients: str, convert_to_speeds: bool = True):
        data = pickle.load(open(polynomial_coefficients, "rb"))
        self.dx_range = [0, 0]
        self.dy_range = [0, 0]
        self.dtheta_range = [0, 0]
        self.dxs = []
        self.dys = []
        self.dthetas = []
        self.data_array = []
        self.period = None
        self.fps = None
        self.frame_offsets = None
        self.startend_double_support_ratio = None
        self.start_offset = None
        self.nb_steps_in_period = None
        self.feet_spacing = None
        self.convert_to_speeds = convert_to_speeds

        self.process(data)

    # # convert to linear and angular velocity
    def steps_to_vel(self, step_size):
        return (step_size * 2) / self.period

    def process(self, data):
        print("[Poly ref data] Processing ...")
        _data = {}
        for name in data.keys():
            split = name.split("_")
            dx = float(split[0])
            dy = float(split[1])
            dtheta = float(split[2])


            if self.period is None:
                self.period = data[name]["period"]
                self.fps = data[name]["fps"]
                self.frame_offsets = data[name]["frame_offsets"]
                self.startend_double_support_ratio = data[name][
                    "startend_double_support_ratio"
                ]
                self.feet_spacing = data[name]["feet_spacing"]
                self.start_offset = int(self.startend_double_support_ratio * self.fps)
                self.nb_steps_in_period = int(self.period * self.fps)
                self.placo_parameters = data[name]["Placo"]

            if self.convert_to_speeds:
                dx = self.steps_to_vel(dx)
                dy = self.steps_to_vel(dy)
                dtheta = self.steps_to_vel(dtheta)

            if dx not in self.dxs:
                self.dxs.append(dx)

            if dy not in self.dys:
                self.dys.append(dy)

            if dtheta not in self.dthetas:
                self.dthetas.append(dtheta)

            self.dx_range = [min(dx, self.dx_range[0]), max(dx, self.dx_range[1])]
            self.dy_range = [min(dy, self.dy_range[0]), max(dy, self.dy_range[1])]
            self.dtheta_range = [
                min(dtheta, self.dtheta_range[0]),
                max(dtheta, self.dtheta_range[1]),
            ]

            if dx not in _data:
                _data[dx] = {}

            if dy not in _data[dx]:
                _data[dx][dy] = {}

            if dtheta not in _data[dx][dy]:
                _data[dx][dy][dtheta] = data[name]

            _coeffs = data[name]["coefficients"]
            left_target = data[name]["left_target"]
            right_target = data[name]["right_target"]

            coeffs = []
            for k, v in _coeffs.items():
                coeffs.append(v)
            _data[dx][dy][dtheta] = (coeffs, left_target, right_target)

        self.dxs = sorted(self.dxs)
        self.dys = sorted(self.dys)
        self.dthetas = sorted(self.dthetas)

        nb_dx = len(self.dxs)
        nb_dy = len(self.dys)
        nb_dtheta = len(self.dthetas)

        self.data_array = nb_dx * [None]
        for x, dx in enumerate(self.dxs):
            self.data_array[x] = nb_dy * [None]
            for y, dy in enumerate(self.dys):
                self.data_array[x][y] = nb_dtheta * [None]
                for th, dtheta in enumerate(self.dthetas):
                    self.data_array[x][y][th] = _data[dx][dy][dtheta][0]

        self.feet_targets_data_array = nb_dx * [None]
        for x, dx in enumerate(self.dxs):
            self.feet_targets_data_array[x] = nb_dy * [None]
            for y, dy in enumerate(self.dys):
                self.feet_targets_data_array[x][y] = nb_dtheta * [None]
                for th, dtheta in enumerate(self.dthetas):
                    self.feet_targets_data_array[x][y][th] = 2 * [None]
                    for foot_i in range(2):
                        self.feet_targets_data_array[x][y][th][foot_i] = _data[dx][dy][dtheta][foot_i + 1]

        print("[Poly ref data] Done processing")

    def vel_to_index(self, dx, dy, dtheta):
        dx = np.clip(dx, self.dx_range[0], self.dx_range[1])
        dy = np.clip(dy, self.dy_range[0], self.dy_range[1])
        dtheta = np.clip(dtheta, self.dtheta_range[0], self.dtheta_range[1])

        ix = np.argmin(np.abs(np.array(self.dxs) - dx))
        iy = np.argmin(np.abs(np.array(self.dys) - dy))
        itheta = np.argmin(np.abs(np.array(self.dthetas) - dtheta))

        return int(ix), int(iy), int(itheta)

    def sample_polynomial(self, t, coeffs):
        ret = []
        for c in coeffs:
            ret.append(np.polyval(np.flip(c), t))

        return ret

    def get_reference_motion(self, dx, dy, dtheta, i):
        ix, iy, itheta = self.vel_to_index(dx, dy, dtheta)
        t = i % self.nb_steps_in_period / self.nb_steps_in_period
        t = np.clip(t, 0.0, 1.0)  # safeguard
        ret = self.sample_polynomial(t, self.data_array[ix][iy][itheta])
        return (
            ret,
            self.feet_targets_data_array[ix][iy][itheta][0],
            self.feet_targets_data_array[ix][iy][itheta][1],
        )


if __name__ == "__main__":
    PRM = PolyReferenceMotion(
        "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
    )
    vals = []
    select_dim = -1
    for i in range(PRM.nb_steps_in_period):
        vals.append(PRM.get_reference_motion(0.0, -0.05, -0.1, i)[select_dim])

    # plot
    import matplotlib.pyplot as plt
    import numpy as np

    ts = np.arange(0, PRM.nb_steps_in_period)
    plt.plot(ts, vals)
    plt.show()
