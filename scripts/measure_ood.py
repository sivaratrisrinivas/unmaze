"""Out-of-distribution probe: Overlook mazes carved by randomised depth-first search ("winding" style).

The audit's first model trained only on uniformly random spanning-tree mazes. Winding mazes are a different
family: long corridors and far fewer branches, closer to how a designed hedge maze feels. This script asks how a
checkpoint does on them. (It is now just `unmaze eval --style winding` with a different seed range.)

    python scripts/measure_ood.py --count 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from measure import solve_all, wilson  # noqa: E402

from unmaze.maze import Puzzle, generate  # noqa: E402
from unmaze.training import load_run  # noqa: E402

OOD_START = 5_000_000_000


def dfs_overlook(n: int, seed: int) -> Puzzle:
    """The depth-first-search ("winding") Overlook maze; the package's own generator makes exactly these."""
    return generate(n, seed, "winding")


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
