# Sigmaban Playground

# Installation

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Training

If you want to use the [imitation reward](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf), you can generate reference motion with [this repo](https://github.com/apirrone/Open_Duck_reference_motion_generator) (in the branch `sigmaban)

Then copy `polynomial_coefficients.pkl` in `playground/sigmaban2024/data/`

You'll also have to set `USE_IMITATION_REWARD=True` in it's `joystick.py` file

Run a training:

```bash
uv run playground/sigmaban2024/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```

## Tensorboard

```bash
uv run tensorboard --logdir=<yourlogdir>
```

# Inference

Infer mujoco

```bash
uv run playground/sigmaban_2024/mujoco_infer.py -o <path_to_.onnx> --model_path playground/sigmaban2024/xmls/scene_flat_terrain_backlash.xml
```

# Mapper

Run the sampler:

```bash
uv run playground/sigmaban2024/footsteps_sampler.py -o <path_to_onnx>
```

This will generate `footsteps.json` 

Run the mapper:

```bash
uv run playground/sigmaban2024/mapper.py --plot
```

