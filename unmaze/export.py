"""Write the data file the web page replays: the real model's frames and the real judge's verdicts."""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from unmaze.diffusion import Schedule
from unmaze.maze import HELD_OUT_START, generate
from unmaze.training import load_run

NOISE_RANGE = 3.0  # noisy masks are N(0, 1)-ish; values beyond three sigma are clipped for storage


def _pack(array: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(array.astype(dtype)).tobytes()).decode("ascii")


def export_web(checkpoint: str | Path, out: str | Path, seeds: Sequence[int], steps: int = 50) -> list[bool]:
    """Solve the held-out Puzzles `seeds` with `checkpoint` and write them to `out` as `window.UNMAZE = {...};`.

    A script file rather than JSON so the page works when opened straight from disk. Failures are
    exported like successes. Returns whether each Puzzle was solved.
    """
    run = load_run(checkpoint)
    solver, n = run.solver(), run.config.maze_size
    mazes, levels = [], []
    for seed in seeds:
        puzzle = generate(n, HELD_OUT_START + seed)
        replay = solver.replay(puzzle, steps=steps, seed=seed)
        verdict = puzzle.judge(replay.guesses[-1] > 0)
        levels = replay.levels
        mazes.append({
            "seed": seed,
            "walls": ["".join("#" if wall else " " for wall in row) for row in puzzle.walls],
            "start": list(puzzle.start),
            "goal": list(puzzle.goal),
            "solution": ["".join("." if on else " " for on in row) for row in puzzle.solution()],
            "solved": verdict.solved,
            "iou": round(verdict.iou, 4),
            "reason": verdict.reason,
            "guesses": _pack(np.round((replay.guesses + 1) / 2 * 255), np.uint8),
            "noisy": _pack(np.round(replay.noisy.clip(-NOISE_RANGE, NOISE_RANGE) / NOISE_RANGE * 127), np.int8),
        })
    alpha_bar = Schedule().alpha_bar
    sigma = [round(float((1 - alpha_bar[level]).sqrt()), 4) for level in levels]
    data = {"version": 1, "mazeSize": n, "grid": 2 * n + 1, "steps": steps, "levels": levels,
            "sigma": sigma, "mazes": mazes}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.UNMAZE = " + json.dumps(data, separators=(",", ":")) + ";\n")
    return [m["solved"] for m in mazes]
