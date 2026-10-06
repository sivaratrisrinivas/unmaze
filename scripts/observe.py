"""Two observations about how the model behaves, measured on the dev split (not the spent test split).

1. The cliff: solve rate on 1,000 winding mazes, bucketed by how long the true route is, for the shipped model
   and for the mixed-style model (runs/mixed.pt, trained by `unmaze train` with the default style mix).
2. Lock-in: at which step does the drawn path stop changing? For each solved maze, the first step after which
   the thresholded guess equals the final answer at every later step.

    python scripts/observe.py        # about 6 minutes on 4 CPU cores
"""

import sys, time
import numpy as np
from unmaze.maze import HELD_OUT_START, generate
from unmaze.training import load_run

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

def solve_all(solver, n, style, count, steps=50):
    ps = [generate(n, HELD_OUT_START + i, style) for i in range(count)]
    atts = []
    for b, first in enumerate(range(0, count, 100)):
        atts += solver.solve(ps[first:first + 100], steps=steps, seed=b)
    return ps, atts

# 1. the cliff: solve rate by route length on 1,000 winding dev mazes, shipped vs mixed model
BUCKETS = [(0, 49), (50, 79), (80, 109), (110, 999)]
for name, path in (("shipped", "unmaze/checkpoints/unmaze.pt"), ("mixed", "runs/mixed.pt")):
    run = load_run(path); solver = run.solver(); n = run.config.maze_size
    ps, atts = solve_all(solver, n, "winding", 1000)
    lens = np.array([int(p.solution().sum()) for p in ps])
    ok = np.array([p.judge(a).solved for p, a in zip(ps, atts)])
    log(f"{name} model, winding, {ok.sum()}/1000 solved")
    for lo, hi in BUCKETS:
        m = (lens >= lo) & (lens <= hi)
        if m.any():
            print(f"   routes {lo}-{hi if hi < 999 else '+'}px: {int(ok[m].sum())}/{int(m.sum())} = {ok[m].mean():.1%}", flush=True)

# 2. when does the path lock in? first step after which the thresholded guess never changes again
def lock_in(solver, n, style, count=100):
    steps_locked, solved = [], 0
    for i in range(count):
        p = generate(n, HELD_OUT_START + i, style)
        g = solver.replay(p, steps=50, seed=i).guesses > 0          # (50, H, W)
        final = g[-1]
        same = (g == final).all(axis=(1, 2))                         # does step t already equal the final answer?
        tail_same = np.flip(np.cumprod(np.flip(same)))               # equal at t and at every later step
        locked = int(np.argmax(tail_same)) + 1                       # 1-based step number
        if p.judge(final).solved:
            solved += 1
            steps_locked.append(locked)
    return np.array(steps_locked), solved

run = load_run("unmaze/checkpoints/unmaze.pt")
s, k = lock_in(run.solver(), run.config.maze_size, "uniform")
log(f"lock-in, shipped model on uniform mazes: {k}/100 solved")
print("   locked in at step: median", int(np.median(s)), "| 90th percentile", int(np.percentile(s, 90)), "| max", int(s.max()),
      "| share locked by step 3:", f"{(s <= 3).mean():.0%}", "| by step 10:", f"{(s <= 10).mean():.0%}", flush=True)
run = load_run("runs/mixed.pt")
s, k = lock_in(run.solver(), run.config.maze_size, "winding")
log(f"lock-in, mixed model on winding mazes: {k}/100 solved")
print("   locked in at step: median", int(np.median(s)), "| 90th percentile", int(np.percentile(s, 90)), "| max", int(s.max()),
      "| share locked by step 3:", f"{(s <= 3).mean():.0%}", "| by step 10:", f"{(s <= 10).mean():.0%}", flush=True)
