"""Measure the shipped checkpoint the way an eval audit asks, and print the numbers as Markdown.

What it checks (see docs/eval-audit.md for what the numbers mean):
  1. A FRESH test range that nothing has ever been evaluated on, against the range used while building.
  2. The judge itself: does it agree with exact comparison against the BFS solution, and does it catch
     deliberately corrupted answers for the right reason?
  3. Sampler-seed sensitivity: are failures about the maze, or about the noise the sampler started from?
  4. Stratified results (route length, entrance side) and a tail stress test on the longest routes.
  5. Anatomy of the failures actually observed (not brainstormed).

    python scripts/measure.py                 # about 15 minutes on 4 CPU cores
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from unmaze.evaluation import wilson_interval as wilson
from unmaze.maze import HELD_OUT_START, Puzzle, generate
from unmaze.training import load_run

FRESH_START = 2_000_000_000   # a range no script, test or training run has ever touched before this audit
TAIL_START = 3_000_000_000
STEPS = 50


def ordered_path(puzzle: Puzzle) -> list[tuple[int, int]]:
    """The Solution's pixels from the entrance to the heart."""
    truth, here, seen, path = puzzle.solution(), puzzle.start, {puzzle.start}, [puzzle.start]
    while here != puzzle.goal:
        nxt = next(((here[0] + dr, here[1] + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if truth[here[0] + dr, here[1] + dc] and (here[0] + dr, here[1] + dc) not in seen), None)
        if nxt is None:
            break
        seen.add(nxt)
        path.append(nxt)
        here = nxt
    return path


def solve_all(solver, n: int, seeds: list[int], steps: int, batch: int = 100, seed_base: int = 0):
    """Attempts for each seed's Puzzle, solved in batches; the sampler seed is `seed_base` + batch number."""
    puzzles = [generate(n, s) for s in seeds]
    attempts = []
    for b, first in enumerate(range(0, len(puzzles), batch)):
        attempts += solver.solve(puzzles[first:first + batch], steps=steps, seed=seed_base + b)
    return puzzles, attempts


def side(puzzle: Puzzle) -> str:
    r, c = puzzle.start
    last = puzzle.walls.shape[0] - 2
    return "top" if r == 1 else "bottom" if r == last else "left" if c == 1 else "right"


def anatomy(puzzle: Puzzle, attempt: np.ndarray, verdict) -> dict:
    """What, concretely, was wrong with a failed Attempt."""
    truth = puzzle.solution()
    missing, extra = truth & ~attempt, attempt & ~truth
    path = ordered_path(puzzle)
    where = [i / max(len(path) - 1, 1) for i, p in enumerate(path) if missing[p]]
    touching = any(extra[r, c] and any(truth[r + dr, c + dc] for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                   for r, c in zip(*np.nonzero(extra)))
    return {
        "reason": verdict.reason, "iou": round(verdict.iou, 3), "route_px": int(truth.sum()),
        "missing_px": int(missing.sum()), "extra_px": int(extra.sum()),
        "extra_on_wall": bool((extra & puzzle.walls).any()),
        "extra_touches_route": bool(touching),
        "missing_where": "none" if not where else
        ("near the entrance" if np.mean(where) < 0.34 else "near the heart" if np.mean(where) > 0.66 else "mid-route"),
        "heart_missing": bool(missing[puzzle.goal]), "entrance_missing": bool(missing[puzzle.start]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("unmaze/checkpoints/unmaze.pt"))
    ap.add_argument("--fresh", type=int, default=5000)
    ap.add_argument("--tail-pool", type=int, default=30_000)
    ap.add_argument("--zero-shot", type=int, default=1000, help="mazes per untrained size")
    ap.add_argument("--leak-seeds", type=int, default=20_000)
    ap.add_argument("--out", type=Path, default=Path("docs/eval-audit-data.json"))
    args = ap.parse_args()

    run = load_run(args.checkpoint)
    solver, n = run.solver(), run.config.maze_size
    result: dict = {"maze_size": n, "steps": STEPS}
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)  # noqa: E731

    # 1. fresh test range vs the range used while building
    fresh_seeds = [FRESH_START + i for i in range(args.fresh)]
    puzzles, attempts = solve_all(solver, n, fresh_seeds, STEPS)
    verdicts = [p.judge(a) for p, a in zip(puzzles, attempts)]
    k = sum(v.solved for v in verdicts)
    result["fresh"] = {"n": args.fresh, "solved": k, "ci95": wilson(k, args.fresh)}
    log(f"fresh test: {k}/{args.fresh} solved, 95% CI {wilson(k, args.fresh)}")
    dev_p, dev_a = solve_all(solver, n, [HELD_OUT_START + i for i in range(1000)], STEPS)
    dk = sum(p.judge(a).solved for p, a in zip(dev_p, dev_a))
    result["dev"] = {"n": 1000, "solved": dk, "ci95": wilson(dk, 1000)}
    log(f"dev range (seen while building): {dk}/1000")
    one_p, one_a = solve_all(solver, n, fresh_seeds[:1000], 1)
    ok = sum(p.judge(a).solved for p, a in zip(one_p, one_a))
    result["fresh_one_step"] = {"n": len(one_p), "solved": ok}
    log(f"fresh, 1 step: {ok}/{len(one_p)}")

    # leakage: do any fresh mazes also appear among training seeds?
    train_walls = {generate(n, s).walls.tobytes() for s in range(args.leak_seeds)}
    result["leak"] = {"fresh_in_training_sample": sum(p.walls.tobytes() in train_walls for p in puzzles),
                      "duplicates_within_fresh": len(puzzles) - len({p.walls.tobytes() for p in puzzles})}
    log(f"leak check: {result['leak']}")

    # 2. judge validation
    agree = sum(v.solved == bool((a == p.solution()).all()) for p, a, v in zip(puzzles, attempts, verdicts))
    exact = sum(p.judge(p.solution()).solved for p in puzzles)
    rng = np.random.default_rng(7)
    kinds = {"deleted a pixel mid-route": ("gap", 0), "deleted the entrance or heart": ("is not marked", 0),
             "marked a wall pixel": ("crosses a wall", 0), "added a spur beside the route": ("stray", 0),
             "added a detached blob": ("stray", 0)}
    table = {}
    for kind, (needle, _) in kinds.items():
        caught = right = tried = 0
        for p in puzzles[:2000]:
            truth, path = p.solution(), ordered_path(p)
            a = truth.copy()
            if kind.startswith("deleted a pixel"):
                a[path[int(rng.integers(1, len(path) - 1))]] = False
            elif kind.startswith("deleted the"):
                a[path[0] if rng.random() < 0.5 else path[-1]] = False
            elif kind.startswith("marked a wall"):
                rs, cs = np.nonzero(p.walls)
                i = int(rng.integers(len(rs)))
                a[rs[i], cs[i]] = True
            else:
                rs, cs = np.nonzero(~p.walls & ~truth)
                near = [(r, c) for r, c in zip(rs, cs) if any(truth[r + dr, c + dc] for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
                pool = near if "spur" in kind else [(r, c) for r, c in zip(rs, cs) if (r, c) not in set(near)]
                if not pool:
                    continue
                r, c = pool[int(rng.integers(len(pool)))]
                a[r, c] = True
            v = p.judge(a)
            tried += 1
            caught += not v.solved
            right += (not v.solved) and needle in v.reason
        table[kind] = {"tried": tried, "judged_unsolved": caught, "right_reason": right}
    result["judge"] = {"agrees_with_exact_comparison": agree, "n": len(puzzles), "exact_solutions_solved": exact,
                       "corruptions": table}
    log(f"judge: agree {agree}/{len(puzzles)}, exact solutions solved {exact}; corruptions {table}")

    # 3. sampler-seed sensitivity
    outcomes = []
    for s in range(3):
        _, att = solve_all(solver, n, fresh_seeds[:1000], STEPS, seed_base=1000 * (s + 1))
        outcomes.append([p.judge(a).solved for p, a in zip(puzzles[:1000], att)])
    fails = np.array(outcomes) == False  # noqa: E712
    result["seed_sensitivity"] = {"per_seed_failures": [int(f.sum()) for f in fails],
                                  "failed_in_at_least_one": int(fails.any(0).sum()),
                                  "failed_in_all_three": int(fails.all(0).sum())}
    log(f"seed sensitivity: {result['seed_sensitivity']}")

    # 4. strata and tail
    lens = np.array([int(p.solution().sum()) for p in puzzles])
    solved = np.array([v.solved for v in verdicts])
    qs = np.quantile(lens, [0.25, 0.5, 0.75])
    bucket = np.digitize(lens, qs)
    result["by_route_length"] = [{"bucket": f"{lo}-{hi} px", "n": int((bucket == b).sum()), "solved": int(solved[bucket == b].sum())}
                                 for b, (lo, hi) in enumerate(zip([lens.min(), *qs + 1], [*qs, lens.max()]))]
    sides = np.array([side(p) for p in puzzles])
    result["by_entrance"] = {s: {"n": int((sides == s).sum()), "solved": int(solved[sides == s].sum())} for s in ("top", "bottom", "left", "right")}
    tail_all = [generate(n, TAIL_START + i) for i in range(args.tail_pool)]
    tail_len = np.array([int(p.solution().sum()) for p in tail_all])
    keep = max(1, args.tail_pool // 100)
    top = np.argsort(tail_len)[-keep:]
    tp, ta = solve_all(solver, n, [TAIL_START + int(i) for i in top], STEPS)
    tk = sum(p.judge(a).solved for p, a in zip(tp, ta))
    result["tail"] = {"pool": args.tail_pool, "kept": keep, "mean_route_px_pool": float(tail_len.mean()),
                      "mean_route_px_kept": float(tail_len[top].mean()), "solved": tk, "ci95": wilson(tk, keep)}
    log(f"strata and tail done: tail {tk}/{keep} (routes avg {tail_len[top].mean():.0f}px vs pool {tail_len.mean():.0f}px)")

    # 5. anatomy of observed failures: fresh at 50 steps, fresh at 1 step, and two sizes it never trained on
    groups = {"fresh, 50 steps": (puzzles, attempts, verdicts), "fresh, 1 step": (one_p, one_a, [p.judge(a) for p, a in zip(one_p, one_a)])}
    for size in (13, 15):
        zp, za = solve_all(solver, size, [FRESH_START + i for i in range(args.zero_shot)], STEPS)
        groups[f"{size}x{size} (untrained size), 50 steps"] = (zp, za, [p.judge(a) for p, a in zip(zp, za)])
    result["anatomy"] = {}
    for name, (ps, as_, vs) in groups.items():
        rows = [anatomy(p, a, v) for p, a, v in zip(ps, as_, vs) if not v.solved]
        result["anatomy"][name] = {
            "n": len(ps), "failures": len(rows),
            "by_reason": dict(Counter(r["reason"] for r in rows)),
            "missing_where": dict(Counter(r["missing_where"] for r in rows)),
            "extra_touches_route": sum(r["extra_touches_route"] for r in rows),
            "extra_on_wall": sum(r["extra_on_wall"] for r in rows),
            "heart_missing": sum(r["heart_missing"] for r in rows),
            "entrance_missing": sum(r["entrance_missing"] for r in rows),
            "median_missing_px": float(np.median([r["missing_px"] for r in rows])) if rows else 0,
            "median_extra_px": float(np.median([r["extra_px"] for r in rows])) if rows else 0,
            "median_iou": float(np.median([r["iou"] for r in rows])) if rows else 1,
            "mean_route_px_failures": float(np.mean([r["route_px"] for r in rows])) if rows else 0,
            "mean_route_px_all": float(np.mean([int(p.solution().sum()) for p in ps])),
        }
        log(f"anatomy {name}: {result['anatomy'][name]['failures']}/{len(ps)} failures")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
