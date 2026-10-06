"""Behaviour of training: it learns, it is repeatable, and a saved checkpoint solves like the original."""

import numpy as np
import torch

from unmaze.maze import generate
from unmaze.training import TrainConfig, load_solver, train

TINY = TrainConfig(maze_size=3, steps=80, batch_size=16, base=8, seed=0)


def test_training_makes_the_denoiser_better_at_predicting_the_clean_mask():
    run = train(TINY)

    first, last = np.mean(run.losses[:10]), np.mean(run.losses[-10:])
    assert last < 0.9 * first


def test_the_same_seed_trains_the_same_denoiser():
    a, b = train(TINY), train(TINY)

    assert a.losses == b.losses


def test_a_saved_checkpoint_loads_back_into_a_solver_that_gives_the_same_attempts(tmp_path):
    run = train(TINY)
    run.save(tmp_path / "tiny.pt")
    puzzles = [generate(3, seed) for seed in range(6)]

    original = run.solver().solve(puzzles, steps=4, seed=5)
    reloaded = load_solver(tmp_path / "tiny.pt").solve(puzzles, steps=4, seed=5)

    assert all((a == b).all() for a, b in zip(original, reloaded))
