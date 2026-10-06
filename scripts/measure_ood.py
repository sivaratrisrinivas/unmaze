"""Out-of-distribution probe: Overlook mazes carved by randomised depth-first search instead of Wilson's algorithm.

The model trains only on uniformly random spanning-tree mazes. DFS mazes are a different family: long winding
corridors and far fewer branches, closer to how a designed hedge maze feels. This script asks how the shipped
checkpoint does on them, without retraining. Not part of the package: it is an audit probe.

    python scripts/measure_ood.py --count 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from measure import solve_all, wilson  # noqa: E402

from unmaze.maze import Puzzle  # noqa: E402
from unmaze.training import load_run  # noqa: E402

OOD_START = 5_000_000_000


def dfs_overlook(n: int, seed: int) -> Puzzle:
    """A perfect maze carved by randomised depth-first search, entered from the outer ring, heart at the centre."""
    rng = np.random.default_rng(seed)
    walls = np.ones((2 * n + 1, 2 * n + 1), dtype=bool)
    here = (int(rng.integers(n)), int(rng.integers(n)))
    walls[2 * here[0] + 1, 2 * here[1] + 1] = False
    stack, seen = [here], {here}
    while stack:
        r, c = stack[-1]
        options = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if 0 <= r + dr < n and 0 <= c + dc < n and (r + dr, c + dc) not in seen]
        if not options:
            stack.pop()
            continue
        nr, nc = options[int(rng.integers(len(options)))]
        walls[r + nr + 1, c + nc + 1] = False
        walls[2 * nr + 1, 2 * nc + 1] = False
        seen.add((nr, nc))
        stack.append((nr, nc))
    ring = [(r, c) for r in range(n) for c in range(n) if r in (0, n - 1) or c in (0, n - 1)]
    er, ec = ring[int(rng.integers(len(ring)))]
    return Puzzle(walls, (2 * er + 1, 2 * ec + 1), (n, n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("unmaze/checkpoints/unmaze.pt"))
    ap.add_argument("--count", type=int, default=500)
    args = ap.parse_args()
    run = load_run(args.checkpoint)
    n, solver = run.config.maze_size, run.solver()

    puzzles = [dfs_overlook(n, OOD_START + i) for i in range(args.count)]
    attempts = []
    for b, first in enumerate(range(0, len(puzzles), 100)):
        attempts += solver.solve(puzzles[first:first + 100], steps=50, seed=b)
    verdicts = [p.judge(a) for p, a in zip(puzzles, attempts)]
    k = sum(v.solved for v in verdicts)
    routes = np.array([int(p.solution().sum()) for p in puzzles])
    lo, hi = wilson(k, len(puzzles))
    print(f"DFS-carved {n}x{n} Overlook mazes (never trained on): {k}/{len(puzzles)} solved = {k / len(puzzles):.1%}, "
          f"95% CI {lo:.1%}-{hi:.1%}")
    print(f"mean route {routes.mean():.0f}px (Wilson-maze mean is about 33px), median {np.median(routes):.0f}px, max {routes.max()}px")
    ok = np.array([v.solved for v in verdicts])
    for a, b in ((0, 40), (40, 80), (80, 160), (160, 10_000)):
        m = (routes >= a) & (routes < b)
        if m.any():
            print(f"  routes {a}-{b if b < 10_000 else '+'}px: {int(ok[m].sum())}/{int(m.sum())} solved")
    reasons = {}
    for v in verdicts:
        if not v.solved:
            reasons[v.reason] = reasons.get(v.reason, 0) + 1
    print("failures by reason:", reasons)


if __name__ == "__main__":
    main()
