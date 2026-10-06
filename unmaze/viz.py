"""Draw a Puzzle and the Denoising Trace of an Attempt as a PNG strip or an animated GIF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from unmaze.maze import Puzzle

_WALL, _OPEN, _PATH = np.array([34, 34, 48]), np.array([246, 244, 238]), np.array([255, 112, 0])
_START, _GOAL = (40, 170, 90), (220, 50, 60)


def frame(puzzle: Puzzle, guess: np.ndarray | None, scale: int = 18) -> Image.Image:
    """One picture of the maze with a guess at the path laid over it (stronger orange = more confident)."""
    pixels = np.where(puzzle.walls[..., None], _WALL, _OPEN).astype(float)
    if guess is not None:
        confidence = ((guess + 1) / 2).clip(0, 1)[..., None]
        pixels = pixels * (1 - confidence) + _PATH * confidence
    image = Image.fromarray(pixels.astype(np.uint8)).resize(
        (pixels.shape[1] * scale, pixels.shape[0] * scale), Image.NEAREST
    )
    draw = ImageDraw.Draw(image)
    for (r, c), colour in ((puzzle.start, _START), (puzzle.goal, _GOAL)):
        draw.rectangle([c * scale + 3, r * scale + 3, (c + 1) * scale - 4, (r + 1) * scale - 4], fill=colour)
    return image


def save_strip(puzzle: Puzzle, trace: list[np.ndarray], path: str | Path, shown: int = 8) -> None:
    """A PNG of the puzzle, evenly spaced moments of the denoising, and the true solution."""
    picks = np.unique(np.linspace(0, len(trace) - 1, shown).round().astype(int))
    panels = [("puzzle", frame(puzzle, None))]
    panels += [(f"step {i + 1}/{len(trace)}", frame(puzzle, trace[i])) for i in picks]
    panels.append(("solution", frame(puzzle, puzzle.solution().astype(float) * 2 - 1)))
    w, h = panels[0][1].size
    label = 18
    sheet = Image.new("RGB", (len(panels) * (w + 6), h + label), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for k, (title, image) in enumerate(panels):
        sheet.paste(image, (k * (w + 6), label))
        draw.text((k * (w + 6) + 2, 3), title, fill=(0, 0, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def save_gif(puzzle: Puzzle, trace: list[np.ndarray], path: str | Path, ms_per_frame: int = 80) -> None:
    """An animation of the path crystallising out of noise, paused on the final Attempt."""
    frames = [frame(puzzle, guess) for guess in trace]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[ms_per_frame] * (len(frames) - 1) + [1500], loop=0)
