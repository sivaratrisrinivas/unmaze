"""Behaviour of evaluation: the Solve Rate on a fixed held-out set, with reasons for failures."""

import numpy as np
import torch

from unmaze.diffusion import Solver
from unmaze.evaluation import evaluate
from unmaze.maze import HELD_OUT_START, generate
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
