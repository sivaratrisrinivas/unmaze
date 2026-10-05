"""Behaviour of the Solver: noise in, Attempt out, with the Denoiser injected.

The sampler is tested through `Solver.solve` only. The Denoiser is the seam: tests hand it
stand-ins whose behaviour is known in advance.
"""

import numpy as np
import pytest
import torch

from unmaze.diffusion import Schedule, Solver
from unmaze.maze import Puzzle, generate
from unmaze.model import UNet


def puzzle_from_channels(channels: torch.Tensor) -> Puzzle:
    walls, start, goal = (c.numpy() for c in channels)
    return Puzzle(
        walls=walls > 0.5,
        start=tuple(int(i) for i in np.argwhere(start > 0.5)[0]),
        goal=tuple(int(i) for i in np.argwhere(goal > 0.5)[0]),
    )


def oracle(x_t, t, cond):
    """A perfect Denoiser: reads the Puzzle from its conditioning and returns the clean Solution."""
    solutions = [puzzle_from_channels(c).solution() for c in cond]
    return torch.tensor(np.stack(solutions), dtype=torch.float32).unsqueeze(1) * 2 - 1


def test_a_perfect_denoiser_leads_the_solver_to_the_exact_solution_of_every_puzzle():
    puzzles = [generate(7, seed) for seed in range(8)]

    attempts = Solver(oracle).solve(puzzles, steps=10, seed=0)

    for puzzle, attempt in zip(puzzles, attempts):
        assert puzzle.judge(attempt).solved


def test_a_denoiser_that_says_everything_is_wall_or_everything_is_path_never_solves_anything():
    puzzles = [generate(7, seed) for seed in range(8)]
    all_wall = lambda x_t, t, cond: torch.full_like(x_t, -1.0)  # noqa: E731
    all_path = lambda x_t, t, cond: torch.full_like(x_t, 1.0)  # noqa: E731

    for denoiser in (all_wall, all_path):
        for puzzle, attempt in zip(puzzles, Solver(denoiser).solve(puzzles, steps=5, seed=0)):
            assert not puzzle.judge(attempt).solved


def test_the_same_seed_gives_the_same_attempts_and_a_different_seed_gives_different_ones():
    puzzles = [generate(7, seed) for seed in range(4)]
    echo = lambda x_t, t, cond: torch.tanh(3 * x_t)  # noqa: E731  # an answer that depends on the noise

    a = Solver(echo).solve(puzzles, steps=5, seed=1)
    b = Solver(echo).solve(puzzles, steps=5, seed=1)
    c = Solver(echo).solve(puzzles, steps=5, seed=2)

    assert all((x == y).all() for x, y in zip(a, b))
    assert any((x != z).any() for x, z in zip(a, c))


def test_the_denoiser_is_asked_about_progressively_less_noisy_masks_starting_from_pure_noise():
    puzzles = [generate(7, seed) for seed in range(16)]
    calls = []

    def recorder(x_t, t, cond):
        calls.append((x_t.clone(), t.clone()))
        return torch.zeros_like(x_t)

    Solver(recorder).solve(puzzles, steps=8, seed=0)

    levels = [int(t[0]) for _, t in calls]
    assert len(calls) == 8
    assert levels == sorted(levels, reverse=True) and len(set(levels)) == 8
    assert levels[0] == 999 and levels[-1] == 0  # trained on 1000 levels, from pure noise to clean
    first = calls[0][0]
    assert abs(first.mean().item()) < 0.1 and abs(first.std().item() - 1.0) < 0.1
    assert first.shape == (16, 1, 15, 15)


def test_the_noise_schedule_starts_almost_clean_ends_as_pure_noise_and_never_recovers_signal():
    schedule = Schedule()

    assert schedule.alpha_bar[0] > 0.999
    assert schedule.alpha_bar[-1] < 1e-3
    assert (schedule.alpha_bar[1:] < schedule.alpha_bar[:-1]).all()


def test_noising_keeps_the_clean_mask_at_level_zero_and_forgets_it_at_the_top_level():
    schedule = Schedule()
    x0 = torch.where(torch.rand(64, 1, 15, 15) > 0.5, 1.0, -1.0)
    noise = torch.randn_like(x0)

    low = schedule.q_sample(x0, torch.zeros(64, dtype=torch.long), noise)
    high = schedule.q_sample(x0, torch.full((64,), 999), noise)

    assert (low.sign() == x0).float().mean() > 0.99
    assert abs(torch.corrcoef(torch.stack([high.flatten(), x0.flatten()]))[0, 1]) < 0.05


def test_the_real_denoiser_accepts_mazes_of_any_size_and_returns_an_attempt_the_size_of_the_grid():
    torch.manual_seed(0)
    denoiser = UNet(base=8).eval()

    for n in (2, 3, 5, 7, 8):
        puzzles = [generate(n, seed) for seed in range(2)]

        attempts = Solver(denoiser).solve(puzzles, steps=3, seed=0)

        assert [a.shape for a in attempts] == [(2 * n + 1, 2 * n + 1)] * 2
        assert all(a.dtype == bool for a in attempts)


def test_the_denoising_trace_has_one_clean_guess_per_step_and_ends_at_the_attempt():
    puzzle = generate(7, 0)
    echo = lambda x_t, t, cond: torch.tanh(3 * x_t)  # noqa: E731

    solver = Solver(echo)
    trace = solver.trace(puzzle, steps=6, seed=3)
    (attempt,) = solver.solve([puzzle], steps=6, seed=3)

    assert len(trace) == 6
    assert all(frame.shape == (15, 15) and np.abs(frame).max() <= 1 for frame in trace)
    assert ((trace[-1] > 0) == attempt).all()


def test_asking_for_zero_sampling_steps_is_an_error_not_a_crash_somewhere_inside():
    with pytest.raises(ValueError, match="at least one"):
        Solver(oracle).solve([generate(3, 0)], steps=0)
