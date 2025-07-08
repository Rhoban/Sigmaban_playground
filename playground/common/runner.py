"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""

from pathlib import Path
from abc import ABC
import socket
import argparse
import functools
from datetime import datetime
from flax.training import orbax_utils
from tensorboardX import SummaryWriter

import os
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax
import wandb

from playground.common.export_onnx import export_onnx


class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self.args = args
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_dir = Path.cwd() / Path(self.output_dir)

        self.env_config = None
        self.env = None
        self.eval_env = None
        self.randomizer = None
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.num_envs = args.num_envs
        self.restore_checkpoint_path = args.restore_checkpoint_path

        # CACHE STUFF
        os.makedirs(".tmp", exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

        if args.wandb:
            self.wandb = True
            run = wandb.init(
                project="sigmaban_playground",
                name=f"{socket.gethostname()} {os.path.basename(self.output_dir)}",
                save_code=True,
            )

            def include_fn(x):
                extensions = [".py", ".yaml", ".json", ".toml", ".xml"]
                for ext in extensions:
                    if x.endswith(ext):
                        return True
                return False

            run.log_code(root="playground", include_fn=include_fn)
        else:
            self.wandb = False

    def progress_callback(self, num_steps: int, metrics: dict) -> None:
        if self.wandb:
            wandb.log(metrics, step=num_steps)

        for metric_name, metric_value in metrics.items():
            # Convert to float, but watch out for 0-dim JAX arrays
            self.writer.add_scalar(metric_name, metric_value, num_steps)

        if "eval/episode_reward" in metrics:
            print("-----------")
            print(
                f"STEP: {num_steps} reward: {metrics['eval/episode_reward']} reward_std: {metrics['eval/episode_reward_std']}"
            )
            print("-----------")

    def policy_params_fn(self, current_step, make_policy, params):
        # save checkpoints

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        print(f"Saving checkpoint (step: {current_step}): {path}")
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)

        onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
        onnx_export_path_with_metadata = f"{self.output_dir}/{d}_{current_step}_with_metadata.onnx"
        export_onnx(
            params,
            self.action_size,
            self.ppo_params,
            self.obs_size,  # may not work
            [float(self.env.dx_range[0]), float(self.env.dx_range[1])],
            [float(self.env.dy_range[0]), float(self.env.dy_range[1])],
            [float(self.env.dtheta_range[0]), float(self.env.dtheta_range[1])],
            self.env.kind,
            output_path=onnx_export_path,
        )

        latest_path = f"{self.output_dir}/latest.onnx"
        # Copy the latest ONNX model to the latest path
        if Path(latest_path).exists():
            Path(latest_path).unlink()
        os.symlink(onnx_export_path_with_metadata, latest_path)

    def train(self) -> None:
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )  # TODO
        self.ppo_training_params = dict(self.ppo_params)
        # self.ppo_training_params["num_timesteps"] = 150000000 * 20

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        self.ppo_training_params["num_envs"] = self.num_envs
        self.ppo_training_params["log_training_metrics"] = True
        self.ppo_training_params["training_metrics_steps"] = 100_000
        print(f"PPO params: {self.ppo_training_params}")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
