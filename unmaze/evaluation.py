"""Measure a Solver: the Solve Rate on a fixed set of held-out Puzzles."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from unmaze.diffusion import Solver
from unmaze.maze import HELD_OUT_START, generate


@dataclass(frozen=True)
class Report:
    solve_rate: float
    mean_iou: float
    failures: dict[str, int]  # why Attempts were not solved -> how many
    count: int


def evaluate(solver: Solver, maze_size: int, count: int = 200, steps: int = 50, batch: int = 100) -> Report:
    """Solve the same `count` held-out Puzzles every time, so Reports are comparable between solvers."""
    verdicts = []
    for first in range(0, count, batch):
        puzzles = [generate(maze_size, HELD_OUT_START + i) for i in range(first, min(first + batch, count))]
        attempts = solver.solve(puzzles, steps=steps, seed=first)
        verdicts += [puzzle.judge(attempt) for puzzle, attempt in zip(puzzles, attempts)]
    return Report(
        solve_rate=sum(v.solved for v in verdicts) / count,
        mean_iou=sum(v.iou for v in verdicts) / count,
        failures=dict(Counter(v.reason for v in verdicts if not v.solved)),
        count=count,
    )
