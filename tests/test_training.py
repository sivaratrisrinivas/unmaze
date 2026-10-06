"""Behaviour of training: it learns, it is repeatable, and a saved checkpoint solves like the original."""

import numpy as np
import torch

from unmaze.maze import generate
from unmaze.training import TrainConfig, draw_batch, load_run, load_solver, train

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


def test_a_batch_draws_each_style_in_proportion_to_its_weight_and_each_label_matches_its_maze():
    config = TrainConfig(maze_size=11, batch_size=1500, styles=("uniform", "winding"), style_weights=(0.5, 0.5))

    drawn = draw_batch(config, np.random.default_rng(0))

    names = [name for name, _ in drawn]
    assert 0.45 < names.count("winding") / len(names) < 0.55
    mean_route = {s: np.mean([p.solution().sum() for name, p in drawn if name == s]) for s in ("uniform", "winding")}
    assert mean_route["winding"] > 1.5 * mean_route["uniform"]  # a "winding" label really is a winding maze


def test_by_default_training_uses_only_uniform_mazes():
    drawn = draw_batch(TrainConfig(maze_size=5, batch_size=64), np.random.default_rng(0))

    assert {name for name, _ in drawn} == {"uniform"}


def test_the_styles_a_model_was_trained_on_survive_a_checkpoint_round_trip(tmp_path):
    config = TrainConfig(maze_size=3, steps=3, batch_size=8, base=8, styles=("winding", "bushy"), style_weights=(0.7, 0.3))
    train(config).save(tmp_path / "mixed.pt")

    loaded = load_run(tmp_path / "mixed.pt").config

    assert loaded.styles == ("winding", "bushy") and loaded.style_weights == (0.7, 0.3)
