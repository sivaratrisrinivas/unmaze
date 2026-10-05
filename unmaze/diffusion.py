"""Diffusion over the Path Mask: the noise schedule, and the Solver that runs the reverse process.

A Denoiser is any callable `(x_t, t, cond) -> x0_hat`:
  x_t     (B, 1, H, W) float, the noisy Path Mask scaled to [-1, 1]
  t       (B,) long, the noise level (0 = almost clean, T-1 = pure noise)
  cond    (B, 3, H, W) float, the Puzzle's wall / start / goal channels
  x0_hat  (B, 1, H, W) float, its guess at the clean Path Mask
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch

from unmaze.maze import Puzzle

Denoiser = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

TRAIN_TIMESTEPS = 1000


class Schedule:
    """Cosine noise schedule (Nichol & Dhariwal, 2021). `alpha_bar[t]` is the signal kept at level t."""

    def __init__(self, timesteps: int = TRAIN_TIMESTEPS, offset: float = 0.008):
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        f = torch.cos((steps + offset) / (1 + offset) * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
        self.timesteps = timesteps
        self.alpha_bar = torch.cumprod(1 - betas, dim=0).float()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Noise a clean Path Mask `x0` to level `t`."""
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


class Solver:
    """Turns Puzzles into Attempts by denoising pure noise with a Denoiser (deterministic DDIM)."""

    def __init__(self, denoiser: Denoiser, schedule: Schedule | None = None):
        self.denoiser = denoiser
        self.schedule = schedule or Schedule()

    def solve(self, puzzles: Sequence[Puzzle], steps: int = 50, seed: int = 0) -> list[np.ndarray]:
        """One Attempt (a boolean Path Mask) per Puzzle."""
        final = self._denoise(puzzles, steps, seed)[-1]
        return [mask > 0 for mask in final.squeeze(1).numpy()]

    def trace(self, puzzle: Puzzle, steps: int = 50, seed: int = 0) -> list[np.ndarray]:
        """The Denoising Trace: the clean guess (values in [-1, 1]) at every step, noisiest first."""
        return [guess[0, 0].numpy() for guess in self._denoise([puzzle], steps, seed)]

    @torch.no_grad()
    def _denoise(self, puzzles: Sequence[Puzzle], steps: int, seed: int) -> list[torch.Tensor]:
        """The clean guess at each step, from the first (noisiest) to the last."""
        cond = torch.from_numpy(np.stack([p.encode() for p in puzzles]))
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(len(puzzles), 1, *cond.shape[2:], generator=generator)

        levels = np.linspace(self.schedule.timesteps - 1, 0, steps).round().astype(int)
        guesses = []
        for i, level in enumerate(levels):
            t = torch.full((len(puzzles),), int(level), dtype=torch.long)
            x0_hat = self.denoiser(x, t, cond).clamp(-1, 1)
            guesses.append(x0_hat)
            ab = self.schedule.alpha_bar[level]
            ab_next = self.schedule.alpha_bar[levels[i + 1]] if i + 1 < steps else torch.tensor(1.0)
            eps_hat = (x - ab.sqrt() * x0_hat) / (1 - ab).sqrt()
            x = ab_next.sqrt() * x0_hat + (1 - ab_next).sqrt() * eps_hat
        return guesses
