"""The committed checkpoint, driven through the command line exactly as a user would.

Slow (it samples hundreds of mazes): run with `pytest -m slow`.
"""

import re
import subprocess
import sys

import pytest

from unmaze.__main__ import DEFAULT_CHECKPOINT

SOLVE_RATE_FLOOR = 0.97  # the shipped checkpoint measures 99.9%; a real drop means the model path regressed


def unmaze(*args: str) -> str:
    result = subprocess.run([sys.executable, "-m", "unmaze", *args], capture_output=True, text=True, check=True)
    return result.stdout


@pytest.mark.slow
def test_the_committed_checkpoint_solves_most_held_out_mazes():
    output = unmaze("eval", "--checkpoint", str(DEFAULT_CHECKPOINT), "--count", "200")

    rate = float(re.search(r"solve rate ([\d.]+)%", output).group(1)) / 100
    assert rate >= SOLVE_RATE_FLOOR, output


@pytest.mark.slow
def test_solve_prints_the_maze_the_attempt_and_the_verdict():
    output = unmaze("solve", "--seed", "0")

    assert "S" in output and "G" in output and "#" in output
    assert re.search(r"(SOLVED|NOT SOLVED)\s+iou \d\.\d\d", output)
