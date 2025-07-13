import jax
import numpy as np
import os
import mujoco
import mujoco.viewer
import jax.numpy as jp
# from playground.sigmaban2024.joystick import Joystick
from playground.sigmaban2024.parkour import Parkour
from playground.sigmaban2024.constants import task_to_xml
from playground.common.onnx_infer import OnnxInfer

os.makedirs(".tmp", exist_ok=True)
jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)
os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

key = jax.random.key(1)
task = "flat_terrain_backlash_parkour"
scene = task_to_xml(task)

model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(
    model,
    data,
    show_left_ui=False,
    show_right_ui=False,
)

# policy = OnnxInfer("ONNX.onnx", awd=True)
# Viewer only
# while True:
#     data.qpos[:]= model.keyframe("home").qpos
#     mujoco.mj_forward(model, data)
#     viewer.sync()

# exit()

env = Parkour(task=task)
print("Resetting...")
state = env.reset(key)

step_fn = jax.jit(env.step)

print("Stepping...")
while True:
    obs = np.array(state.obs['state'])
    # action = policy.infer(obs)
    action = np.zeros(20)

    state = step_fn(state, jp.array(action))

    # print(state.data.qpos)
    data.qpos[:] = state.data.qpos
    mujoco.mj_forward(model, data)
    viewer.sync()
    # input()
