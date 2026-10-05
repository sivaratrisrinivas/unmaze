"""Command line: python -m unmaze {train,eval,solve,demo}."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from unmaze.evaluation import evaluate
from unmaze.maze import HELD_OUT_START, generate
from unmaze.training import TrainConfig, load_run, train
from unmaze.viz import save_gif, save_strip

DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "unmaze.pt"


def _train(args: argparse.Namespace) -> None:
    config = TrainConfig(maze_size=args.maze_size, steps=args.steps, batch_size=args.batch_size,
                         lr=args.lr, base=args.base, seed=args.seed)
    start = time.time()

    def progress(step, loss, run):
        line = f"step {step:>6}/{config.steps}  loss {loss:.4f}  {time.time() - start:6.0f}s"
        if args.eval_every and (step % args.eval_every == 0 or step == config.steps):
            report = evaluate(run.solver(), config.maze_size, count=args.eval_count, steps=args.sample_steps)
            line += f"  | held-out solve rate {report.solve_rate:.1%} (iou {report.mean_iou:.2f})"
            run.save(args.out)  # keep the latest averaged weights on disk as we go
        print(line, flush=True)

    run = train(config, on_progress=progress, every=args.log_every)
    run.save(args.out)
    print(f"saved {args.out}")


def _eval(args: argparse.Namespace) -> None:
    run = load_run(args.checkpoint)
    report = evaluate(run.solver(), run.config.maze_size, count=args.count, steps=args.sample_steps)
    print(f"{run.config.maze_size}x{run.config.maze_size} mazes, {report.count} held-out puzzles, "
          f"{args.sample_steps} sampling steps")
    print(f"solve rate {report.solve_rate:.1%}   mean iou {report.mean_iou:.3f}")
    for reason, count in sorted(report.failures.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>4} unsolved: {reason}")


def _solve(args: argparse.Namespace) -> None:
    run = load_run(args.checkpoint)
    puzzle = generate(run.config.maze_size, HELD_OUT_START + args.seed)
    solver = run.solver()
    trace = solver.trace(puzzle, steps=args.sample_steps, seed=args.seed)
    attempt = trace[-1] > 0
    verdict = puzzle.judge(attempt)
    print(puzzle.render(attempt))
    print(f"\n{'SOLVED' if verdict.solved else 'NOT SOLVED'}  iou {verdict.iou:.2f}  {verdict.reason}".rstrip())
    if args.out:
        save_strip(puzzle, trace, args.out)
        print(f"saved {args.out}")
    if args.gif:
        save_gif(puzzle, trace, args.gif)
        print(f"saved {args.gif}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="unmaze", description="A diffusion model that solves mazes.")
    sub = parser.add_subparsers(required=True)

    t = sub.add_parser("train", help="train a denoiser on freshly generated mazes (CPU is fine)")
    t.add_argument("--maze-size", type=int, default=7, help="cells per side (the grid is 2n+1 pixels)")
    t.add_argument("--steps", type=int, default=10_000)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--base", type=int, default=48, help="U-Net width")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, default=Path("runs/model.pt"))
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--eval-every", type=int, default=1000, help="0 disables held-out evaluation")
    t.add_argument("--eval-count", type=int, default=100)
    t.add_argument("--sample-steps", type=int, default=50)
    t.set_defaults(run=_train)

    e = sub.add_parser("eval", help="solve rate of a checkpoint on a fixed held-out set")
    e.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    e.add_argument("--count", type=int, default=200)
    e.add_argument("--sample-steps", type=int, default=50)
    e.set_defaults(run=_eval)

    for name, helptext in (("solve", "solve one held-out maze and show it"),
                           ("demo", "solve one maze and save a picture of the denoising")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
        s.add_argument("--seed", type=int, default=0, help="which held-out maze")
        s.add_argument("--sample-steps", type=int, default=50)
        s.add_argument("--out", type=Path, default=Path("demo.png") if name == "demo" else None,
                       help="PNG strip of the denoising")
        s.add_argument("--gif", type=Path, default=Path("demo.gif") if name == "demo" else None,
                       help="animated GIF of the denoising")
        s.set_defaults(run=_solve)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
