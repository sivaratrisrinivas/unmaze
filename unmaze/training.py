"""Train the Denoiser on freshly generated Puzzles, and save / load checkpoints."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from unmaze.diffusion import Schedule, Solver
from unmaze.maze import HELD_OUT_START, Puzzle, generate
from unmaze.model import UNet


@dataclass(frozen=True)
class TrainConfig:
    maze_size: int = 11
    steps: int = 6_000
    batch_size: int = 64
    lr: float = 1e-3
    base: int = 48
    ema_decay: float = 0.999
    seed: int = 0
    styles: tuple[str, ...] = ("uniform",)       # which kinds of maze to train on...
    style_weights: tuple[float, ...] = (1.0,)    # ...and how often each is drawn


@dataclass
class Run:
    """A finished (or in-progress) training run: the averaged weights and the loss history."""

    model: UNet
    config: TrainConfig
    losses: list[float] = field(default_factory=list)

    def solver(self) -> Solver:
        return Solver(self.model.eval())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.model.state_dict()}, path)


def load_run(path: str | Path) -> Run:
    """Rebuild a Run from a checkpoint with no further configuration (the loss history is not kept)."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    saved["config"] = {k: tuple(v) if isinstance(v, list) else v for k, v in saved["config"].items()}
    config = TrainConfig(**saved["config"])
    model = UNet(base=config.base)
    model.load_state_dict(saved["state_dict"])
    return Run(model.eval(), config)


def load_solver(path: str | Path) -> Solver:
    """A Solver ready to use, rebuilt from a checkpoint."""
    return load_run(path).solver()


def draw_batch(config: TrainConfig, rng: np.random.Generator) -> list[tuple[str, Puzzle]]:
    """A fresh batch of (style, Puzzle) pairs, each style drawn in proportion to its weight."""
    seeds = rng.integers(0, HELD_OUT_START, config.batch_size)
    if len(config.styles) == 1:  # draws nothing extra, so single-style runs stay reproducible
        chosen = [0] * config.batch_size
    else:
        weights = np.array(config.style_weights, dtype=float)
        chosen = rng.choice(len(config.styles), size=config.batch_size, p=weights / weights.sum())
    return [(config.styles[k], generate(config.maze_size, int(seed), config.styles[k]))
            for k, seed in zip(chosen, seeds)]


def train(config: TrainConfig, on_progress: Callable[[int, float, Run], None] | None = None,
          every: int = 100) -> Run:
    """Teach a Denoiser to predict the clean Path Mask from a noised one.

    `on_progress(step, loss, run)` is called every `every` steps and at the end.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    schedule = Schedule()
    model = UNet(base=config.base)
    averaged = copy.deepcopy(model)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
    run = Run(averaged, config)

    warmup = min(100, config.steps // 10)
    for step in range(config.steps):
        lr_scale = (step + 1) / warmup if step < warmup else _cosine(step - warmup, config.steps - warmup)
        for group in optimiser.param_groups:
            group["lr"] = config.lr * lr_scale

        puzzles = [puzzle for _, puzzle in draw_batch(config, rng)]
        cond = torch.from_numpy(np.stack([p.encode() for p in puzzles]))
        x0 = torch.from_numpy(np.stack([p.solution() for p in puzzles])).float().unsqueeze(1) * 2 - 1
        t = torch.randint(0, schedule.timesteps, (config.batch_size,))
        x_t = schedule.q_sample(x0, t, torch.randn_like(x0))

        loss = (model(x_t, t, cond) - x0).pow(2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        decay = min(config.ema_decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for avg, new in zip(averaged.parameters(), model.parameters()):
                avg.lerp_(new, 1 - decay)
        run.losses.append(loss.item())
        if on_progress and ((step + 1) % every == 0 or step + 1 == config.steps):
            on_progress(step + 1, loss.item(), run)
    return run


def _cosine(done: int, total: int, floor: float = 0.05) -> float:
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * done / max(total, 1)))
