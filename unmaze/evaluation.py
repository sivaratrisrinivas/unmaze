"""Measure a Solver: the Solve Rate on a fixed set of held-out Puzzles."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from unmaze.diffusion import Solver
from unmaze.maze import HELD_OUT_START, TEST_START, generate


@dataclass(frozen=True)
class Report:
    solve_rate: float
    mean_iou: float
    failures: dict[str, int]  # why Attempts were not solved -> how many
    count: int
    ci95: tuple[float, float]  # Wilson 95% confidence interval on the solve rate


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a rate; unlike the usual normal one it behaves at 0% and 100%."""
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(solver: Solver, maze_size: int, count: int = 200, steps: int = 50, batch: int = 100,
             split: str = "dev", style: str = "uniform") -> Report:
    """Solve the same `count` held-out Puzzles every time, so Reports are comparable between solvers.

    `split` is "dev" (fine to look at while building) or "test" (for final reporting only);
    `style` is the kind of maze (see unmaze.maze.STYLES).
    """
    start = {"dev": HELD_OUT_START, "test": TEST_START}[split]
    verdicts = []
    for first in range(0, count, batch):
        puzzles = [generate(maze_size, start + i, style) for i in range(first, min(first + batch, count))]
        attempts = solver.solve(puzzles, steps=steps, seed=first)
        verdicts += [puzzle.judge(attempt) for puzzle, attempt in zip(puzzles, attempts)]
    return Report(
        solve_rate=sum(v.solved for v in verdicts) / count,
        mean_iou=sum(v.iou for v in verdicts) / count,
        failures=dict(Counter(v.reason for v in verdicts if not v.solved)),
        count=count,
        ci95=wilson_interval(sum(v.solved for v in verdicts), count),
    )
