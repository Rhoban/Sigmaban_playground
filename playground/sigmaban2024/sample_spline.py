from playground.sigmaban2024.read_spline import ReadSplines
import sys
import json

FPS = 50

rs = ReadSplines(sys.argv[1])
spline_length = rs.all_max_ts  # s

support = "left"
shoot = "right" if support == "left" else "left"
actuators_used = list(rs.splines.keys())
# for name in actuators_used:
#     nname = name.replace("shoot", shoot).replace("support", support)

walk_pose = {
    "head_yaw": 0,
    "head_pitch": 0,
    "left_shoulder_pitch": 0.35,
    "left_shoulder_roll": 0,
    "left_elbow": -0.86,
    "right_shoulder_pitch": 0.35,
    "right_shoulder_roll": 0,
    "right_elbow": -0.86,
    "left_hip_yaw": 0,
    "left_hip_roll": 0.12,
    "left_hip_pitch": -0.65,
    "left_knee": 1.11276,
    "left_ankle_pitch": -0.567338,
    "left_ankle_roll": -0.129063,
    "right_hip_yaw": 0,
    "right_hip_roll": -0.12,
    "right_hip_pitch": -0.64,
    "right_knee": 1.11276,
    "right_ankle_pitch": -0.567338,
    "right_ankle_roll": 0.129063,
}

t = 0
dt = 0.01

nb_steps = 0
while t < spline_length:
    remap_factor = rs.get_remap_factor(t)
    t += dt * remap_factor
    nb_steps += 1

mean_dt = spline_length / nb_steps

# very weird but seems to work
t = 0
straight_t = 0
last_sample = straight_t
tmp_splines = []
while t < spline_length:
    remap_factor = rs.get_remap_factor(t)
    t += dt * remap_factor

    straight_t += mean_dt
    if straight_t - last_sample >= 1 / FPS:
        last_sample = straight_t
        tmp_splines.append(rs.get_all_splines_at(actuators_used, t))

splines = []
for t, spline in enumerate(tmp_splines):
    splines.append(walk_pose.copy())
    for i, name in enumerate(actuators_used):
        nname = name.replace("shoot", shoot).replace("support", support)
        splines[-1][nname] += spline[name]


print(splines[0])
data = {
    "duration": float(spline_length),
    "FPS": float(FPS),
    "nb_steps": len(splines),
    "Frames": splines,
}
json.dump(data, open("sampled_splines.json", "w"), indent=2)


