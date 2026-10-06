"""Behaviour of the web export: the page's data file is the real model's output, judged by the real judge."""

import base64
import json

import numpy as np

from unmaze.export import export_web
from unmaze.maze import HELD_OUT_START, generate
from unmaze.training import TrainConfig, load_solver, train

STEPS = 4
SEEDS = [0, 3]


def read_data_file(path) -> dict:
    text = path.read_text()
    assert text.startswith("window.UNMAZE = ") and text.rstrip().endswith(";")
    return json.loads(text[len("window.UNMAZE = "):].rstrip().rstrip(";"))


def unpack(b64: str, dtype, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=dtype).reshape(shape)


def test_the_exported_file_holds_the_models_own_frames_and_the_judges_own_verdicts(tmp_path):
    train(TrainConfig(maze_size=3, steps=60, batch_size=16, base=8, seed=0)).save(tmp_path / "tiny.pt")
    solver = load_solver(tmp_path / "tiny.pt")

    export_web(tmp_path / "tiny.pt", tmp_path / "data.js", seeds=SEEDS, steps=STEPS)
    data = read_data_file(tmp_path / "data.js")

    assert data["mazeSize"] == 3 and data["grid"] == 7 and data["steps"] == STEPS
    assert [m["seed"] for m in data["mazes"]] == SEEDS
    for entry in data["mazes"]:
        puzzle = generate(3, HELD_OUT_START + entry["seed"])
        replay = solver.replay(puzzle, steps=STEPS, seed=entry["seed"])
        verdict = puzzle.judge(replay.guesses[-1] > 0)

        assert data["levels"] == replay.levels
        assert [[c == "#" for c in row] for row in entry["walls"]] == puzzle.walls.tolist()
        assert entry["start"] == list(puzzle.start) and entry["goal"] == list(puzzle.goal)
        assert entry["solved"] == verdict.solved and abs(entry["iou"] - verdict.iou) < 1e-3
        assert entry["reason"] == verdict.reason
        guesses = unpack(entry["guesses"], np.uint8, (STEPS, 7, 7)) / 255 * 2 - 1  # 0..255 <-> -1..1
        noisy = unpack(entry["noisy"], np.int8, (STEPS, 7, 7)) / 127 * 3  # -127..127 <-> -3..3
        assert np.abs(guesses - replay.guesses).max() < 0.01
        assert np.abs(noisy - replay.noisy.clip(-3, 3)).max() < 0.02


def test_the_exported_noise_levels_fall_from_nearly_all_noise_to_nearly_none(tmp_path):
    train(TrainConfig(maze_size=3, steps=10, batch_size=8, base=8, seed=0)).save(tmp_path / "tiny.pt")

    export_web(tmp_path / "tiny.pt", tmp_path / "data.js", seeds=[0], steps=10)
    sigma = read_data_file(tmp_path / "data.js")["sigma"]  # the share of each step's input that is noise

    assert len(sigma) == 10
    assert sigma[0] > 0.99 and sigma[-1] < 0.02
    assert all(a > b for a, b in zip(sigma, sigma[1:]))
