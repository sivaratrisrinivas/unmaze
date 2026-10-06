"""Behaviour of evaluation: the Solve Rate on a fixed held-out set, with reasons for failures."""

import numpy as np
import pytest
import torch

from unmaze.diffusion import Solver
from unmaze.evaluation import evaluate
from unmaze.maze import HELD_OUT_START, TEST_START, generate
from tests.test_solver import oracle


def test_a_perfect_solver_has_a_solve_rate_of_one():
    report = evaluate(Solver(oracle), maze_size=5, count=20)

    assert report.solve_rate == 1.0
    assert report.mean_iou == 1.0
    assert report.failures == {}


def test_a_solver_that_answers_blank_has_a_solve_rate_of_zero_and_says_why():
    blank = Solver(lambda x_t, t, cond: torch.full_like(x_t, -1.0))

    report = evaluate(blank, maze_size=5, count=20)

    assert report.solve_rate == 0.0
    assert report.failures == {"the start is not marked": 20}


def test_evaluation_always_uses_the_same_held_out_mazes():
    seen = []

    def spy(x_t, t, cond):
        seen.append(cond.clone())
        return torch.full_like(x_t, -1.0)

    evaluate(Solver(spy), maze_size=5, count=10, steps=1)

    expected = np.stack([generate(5, HELD_OUT_START + i).encode() for i in range(10)])
    assert (seen[0].numpy() == expected).all()


def test_the_report_carries_a_95_percent_confidence_interval_so_small_samples_are_not_oversold():
    perfect = evaluate(Solver(oracle), maze_size=5, count=20)
    blank = evaluate(Solver(lambda x_t, t, cond: torch.full_like(x_t, -1.0)), maze_size=5, count=20)

    # Wilson interval for 20/20 and 0/20, worked by hand: [1 / (1 + z^2/n), 1] and [0, (z^2/n) / (1 + z^2/n)].
    assert perfect.ci95 == pytest.approx((0.8389, 1.0), abs=1e-3)
    assert blank.ci95 == pytest.approx((0.0, 0.1611), abs=1e-3)


def test_the_test_split_is_a_different_set_of_mazes_from_the_dev_split():
    seen = []

    def spy(x_t, t, cond):
        seen.append(cond.clone())
        return torch.full_like(x_t, -1.0)

    evaluate(Solver(spy), maze_size=5, count=10, steps=1, split="test")

    expected = np.stack([generate(5, TEST_START + i).encode() for i in range(10)])
    assert (seen[0].numpy() == expected).all()
    assert TEST_START != HELD_OUT_START


def test_evaluation_can_be_asked_for_one_style_of_maze_at_a_time():
    seen = []

    def spy(x_t, t, cond):
        seen.append(cond.clone())
        return torch.full_like(x_t, -1.0)

    evaluate(Solver(spy), maze_size=7, count=10, steps=1, style="winding")

    expected = np.stack([generate(7, HELD_OUT_START + i, "winding").encode() for i in range(10)])
    assert (seen[0].numpy() == expected).all()
