import argparse
import concurrent.futures as futures
import json
from typing import List
import threading

import numpy as np
import placo
import tqdm

from playground.sigmaban2024.mujoco_infer import MjInfer


class FootstepsSamples:
    """Footstep samples dataset obtained by the sampler"""

    def __init__(self):
        self.footsteps: List[dict] = []

    # ------------------------------------------------------------------
    # Basic CRUD helpers ------------------------------------------------
    # ------------------------------------------------------------------
    def add(self, support: str, command: tuple, footstep: tuple):
        self.footsteps.append(
            {"support": support, "command": command, "footstep": footstep}
        )

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.footsteps, f)

    def load(self, filename):
        if filename:
            with open(filename, "r") as f:
                self.footsteps = json.load(f)

    # ------------------------------------------------------------------
    # Convenience: numpy views
    # ------------------------------------------------------------------
    def get_data(self, side: str):
        return (
            np.array(
                [entry["command"] for entry in self.footsteps if entry["support"] == side]
            ).copy(),
            np.array(
                [entry["footstep"] for entry in self.footsteps if entry["support"] == side]
            ).copy(),
        )

    def prepare(self):
        commands_right, footsteps_right = self.get_data("right")
        commands_left, footsteps_left = self.get_data("left")

        feet_spacing = np.median(
            np.concatenate((footsteps_right[:, 1], -footsteps_left[:, 1]))
        )
        footsteps_right[:, 1] -= feet_spacing
        footsteps_left[:, 1] += feet_spacing

        commands = np.vstack((commands_left, commands_right))
        footsteps = np.vstack((footsteps_left, footsteps_right))
        return feet_spacing, commands, footsteps


class FootstepsSampler:
    """Gets the robot walking and sample footsteps (dx, dy, dtheta) vs commands (vx, vy, vtheta)"""

    def __init__(self, mjinfer: MjInfer):
        self.mjinfer = mjinfer
        self.warmup_footsteps = 3
        self.samples = FootstepsSamples()
        self.humanoid_parameters = mjinfer.make_humanoid_parameters()

    # ------------------------------------------------------------------
    #  Public API -------------------------------------------------------
    # ------------------------------------------------------------------
    def reset(self):
        self.mjinfer.reset()
        self._sample_command()

    def sample(self, samples: int = 1):
        """Collect *samples* (each call yields a left+right footstep pair)."""
        # Warm‑up so controller converges
        for _ in range(self.warmup_footsteps):
            self.mjinfer.walk_one_step()
        while self.mjinfer.support != "left":
            self.mjinfer.walk_one_step()

        for _ in range(samples):
            # right support ➜ left landing
            self._record_footstep("right_foot", "left_foot", "right")
            self.mjinfer.walk_one_step()

            # left support ➜ right landing
            self._record_footstep("left_foot", "right_foot", "left")
            self.mjinfer.walk_one_step()

    # ------------------------------------------------------------------
    #  Internals --------------------------------------------------------
    # ------------------------------------------------------------------
    def _sample_command(self):
        self.mjinfer.commands = [
            float(np.random.uniform(*self.mjinfer.COMMANDS_RANGE_X)),
            float(np.random.uniform(*self.mjinfer.COMMANDS_RANGE_Y)),
            float(np.random.uniform(*self.mjinfer.COMMANDS_RANGE_THETA)),
        ]

    def _record_footstep(self, support_foot: str, landing_foot: str, side: str):
        T = np.linalg.inv(self.mjinfer.get_T_world_site(support_foot)) @ self.mjinfer.get_T_world_site(landing_foot)
        dx, dy = T[0, 3], T[1, 3]
        dtheta = np.arctan2(T[1, 0], T[0, 0])
        self.samples.add(side, self.mjinfer.commands, [float(dx), float(dy), float(dtheta)])


# ---------------------------------------------------------------------------
#                           Parallel utilities
# ---------------------------------------------------------------------------

def _worker(n_worker_samples: int, args_dict: dict, pbar: "tqdm.tqdm") -> list[dict]:
    """Each thread gets its own MuJoCo simulator + sampler."""
    mjinfer = MjInfer(
        args_dict["model_path"],
        args_dict["reference_data"],
        args_dict["onnx_model_path"],
    )

    sampler = FootstepsSampler(mjinfer)
    local_steps: list[dict] = []
    for _ in range(n_worker_samples):
        sampler.reset()
        sampler.sample()
        local_steps.extend(sampler.samples.footsteps)
        sampler.samples.footsteps.clear()

        # Atomically update the global progress‑bar (thread‑safe via set_lock)
        pbar.update(1)

    return local_steps


# ---------------------------------------------------------------------------
#                                    Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="playground/sigmaban2024/xmls/scene_flat_terrain.xml")
    parser.add_argument("--reference_data", type=str, default="playground/sigmaban2024/data/polynomial_coefficients.pkl")
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--output", type=str, default="footsteps.json")
    parser.add_argument("--view", action="store_true", default=False, help="Display MuJoCo viewer (forces serial execution)")
    parser.add_argument("-j", "--jobs", type=int, default=1, metavar="N", help="Number of worker threads (default: 1)")
    args = parser.parse_args()

    # When a MuJoCo viewer is requested, we must run single‑threaded
    if args.view and args.jobs > 1:
        print("[Warning] --view forces serial run; ignoring -j>1")
        args.jobs = 1

    # ------------------------------------------------------------------
    # Serial run (original behaviour) ----------------------------------
    # ------------------------------------------------------------------
    if args.jobs == 1:
        mjinfer = MjInfer(args.model_path, args.reference_data, args.onnx_model_path)
        if args.view:
            mjinfer.enable_viewer(key=False)
        sampler = FootstepsSampler(mjinfer)
        for _ in tqdm.tqdm(range(args.n_samples), desc="Sampling", unit="step"):
            sampler.reset()
            sampler.sample()
        sampler.samples.save(args.output)
        print(f"Saved {len(sampler.samples.footsteps)} steps ➜ {args.output}")
        exit()

    # ------------------------------------------------------------------
    # Multi‑threaded run -----------------------------------------------
    # ------------------------------------------------------------------
    base = args.n_samples // args.jobs
    per_worker: List[int] = [base] * args.jobs
    for i in range(args.n_samples % args.jobs):
        per_worker[i] += 1

    shared_args = vars(args)

        # ------------------------------------------------------------------
    # Thread‑safe progress bar ----------------------------------------
    # ------------------------------------------------------------------
    shared_args = vars(args)
    _lock = threading.RLock()

    # tqdm>=4.42 exposes set_lock; older versions do not. Use if present.
    if hasattr(tqdm, "tqdm") and hasattr(tqdm.tqdm, "set_lock"):
        tqdm.tqdm.set_lock(_lock)

    aggregated_steps: list[dict] = []
    with tqdm.tqdm(total=args.n_samples,
                   desc="Sampling", unit="step",
                   lock_args=(_lock,)) as pbar:
        with futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_map = {
                executor.submit(_worker, n, shared_args, pbar): n for n in per_worker if n
            }
            for fut in futures.as_completed(future_map):
                aggregated_steps.extend(fut.result())

    # ------------------------------------------------------------------
    # Save aggregated dataset ------------------------------------------
    # ------------------------------------------------------------------
    with open(args.output, "w") as f:
        json.dump(aggregated_steps, f)
    print(f"Saved {len(aggregated_steps)} steps ➜ {args.output}")
    # ------------------------------------------------------------------
    with open(args.output, "w") as f:
        json.dump(aggregated_steps, f)
    print(f"Saved {len(aggregated_steps)} steps ➜ {args.output}")
