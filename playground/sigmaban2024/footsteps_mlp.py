import json
import random
import argparse
import torch as th

from playground.sigmaban2024.footsteps_mapper import FootstepsMapper
from playground.sigmaban2024.footsteps_sampler import FootstepsSamples


class MLP(th.nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int, device="cpu"):
        super().__init__()

        self.net = th.nn.Sequential(
            th.nn.Linear(input_dimension, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 256),
            th.nn.ReLU(),
            # th.nn.Linear(128, 128),
            # th.nn.ELU(),
            th.nn.Linear(256, output_dimension),
        ).to(device)

    def forward(self, x):
        return self.net(x)

    def load(self):
        self.load_state_dict(th.load("footsteps.weights"))

    def save(self):
        th.save(self.state_dict(), "footsteps.weights")


class FootstepsMapperMLP(FootstepsMapper):
    def __init__(self):
        """
        MLP mapper, not very tested
        """

    def fit(self, samples: FootstepsSamples):
        self.feet_spacing, commands, footsteps = samples.prepare()

        self.mlp = MLP(3, 3, device="cuda")
        # commands = th.tensor(commands, dtype=th.float32).to("cuda")
        # footsteps = th.tensor(footsteps, dtype=th.float32).to("cuda")

        optimizer = th.optim.Adam(self.mlp.parameters(), lr=1e-3)
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.95, patience=512
        )

        commands = th.tensor(commands, dtype=th.float32).to("cuda")
        footsteps = th.tensor(footsteps, dtype=th.float32).to("cuda")

        batch_size = 4096
        indices = range(len(commands))

        for k in range(100_000):
            batch_idx = random.choices(indices, k=batch_size)

            loss = th.nn.functional.mse_loss(
                commands[batch_idx], self.mlp(footsteps[batch_idx])
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            print(f"[{k}] loss={loss}, lr={scheduler.get_last_lr()}")

            if scheduler.get_last_lr()[0] < 1e-6:
                print("LR very low, stopping")
                break

        self.trained = True
        self.mlp.save()

    def remap(self, dx, dy, dtheta):
        with th.no_grad():
            return self.mlp(th.tensor([dx, dy, dtheta], dtype=th.float32)).numpy()

    def load(self):
        mapper.prepare()
        self.mlp = MLP(3, 3)
        self.mlp.load()
        self.trained = True

        print(self.mlp(th.tensor([0.0, 0.03, 0.0], dtype=th.float32)))
        print(self.mlp(th.tensor([0.0, -0.03, 0.0], dtype=th.float32)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--footsteps",
        type=str,
        default="footsteps.json",
    )
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    args = parser.parse_args()

    mapper = FootstepsMapperMLP()

    samples = FootstepsSamples()
    samples.load(args.footsteps)

    if args.train:
        mapper.fit(samples)
    if args.plot:
        mapper.load()
        mapper.evaluate(samples)
        mapper.show_plot(samples)
