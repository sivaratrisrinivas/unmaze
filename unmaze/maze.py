"""The maze world: puzzles, their true solutions, and the verdict on an attempt."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

Pixel = tuple[int, int]

# Seeds from here up are reserved for evaluation; training only ever draws below it.
HELD_OUT_START = 1_000_000_000
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _flood(mask: np.ndarray, source: Pixel) -> set[Pixel]:
    """Every pixel of `mask` reachable from `source` by 4-connected steps through `mask`."""
    seen = {source}
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _STEPS:
            nxt = (r + dr, c + dc)
            if nxt not in seen and mask[nxt]:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _is_simple_path(pixels: set[Pixel], start: Pixel, goal: Pixel) -> bool:
    """True when the connected `pixels` form one path with `start` and `goal` as its two ends."""
    neighbours = {
        p: sum((p[0] + dr, p[1] + dc) in pixels for dr, dc in _STEPS) for p in pixels
    }
    ends = {p for p, n in neighbours.items() if n == 1}
    return ends == {start, goal} and all(n <= 2 for n in neighbours.values())


@dataclass(frozen=True)
class Verdict:
    """The result of judging an Attempt: solved or not, overlap with the Solution, and why not."""

    solved: bool
    iou: float
    reason: str = ""


@dataclass(frozen=True)
class Puzzle:
    """A maze Grid (True = wall) with a Start and a Goal, both given as (row, col) pixels."""

    walls: np.ndarray
    start: Pixel
    goal: Pixel

    @classmethod
    def parse(cls, text: str) -> Puzzle:
        """Read a hand-drawn puzzle: '#' wall, ' ' open, 'S' start, 'G' goal."""
        rows = text.splitlines()
        walls = np.array([[c == "#" for c in row] for row in rows])
        (start,) = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "S"]
        (goal,) = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "G"]
        return cls(walls, start, goal)

    def render(self, attempt: np.ndarray | None = None) -> str:
        """The Puzzle as text (the format `parse` reads), with an Attempt's pixels drawn as '.'."""
        lines = []
        for r, row in enumerate(self.walls):
            line = ""
            for c, wall in enumerate(row):
                if (r, c) == self.start:
                    line += "S"
                elif (r, c) == self.goal:
                    line += "G"
                elif wall:
                    line += "#"
                else:
                    line += "." if attempt is not None and attempt[r, c] else " "
            lines.append(line)
        return "\n".join(lines)

    def solution(self) -> np.ndarray:
        """The Path Mask of the shortest Start-to-Goal path, found by breadth-first search."""
        came_from: dict[Pixel, Pixel | None] = {self.start: None}
        queue = deque([self.start])
        while queue:
            here = queue.popleft()
            if here == self.goal:
                break
            for dr, dc in _STEPS:
                nxt = (here[0] + dr, here[1] + dc)
                if nxt not in came_from and not self.walls[nxt]:
                    came_from[nxt] = here
                    queue.append(nxt)
        mask = np.zeros_like(self.walls)
        step: Pixel | None = self.goal
        while step is not None:
            mask[step] = True
            step = came_from[step]
        return mask

    def encode(self) -> np.ndarray:
        """The Puzzle as float32 conditioning channels for the Denoiser: wall, Start, Goal."""
        start, goal = np.zeros(self.walls.shape, np.float32), np.zeros(self.walls.shape, np.float32)
        start[self.start] = goal[self.goal] = 1.0
        return np.stack([self.walls.astype(np.float32), start, goal])

    def judge(self, attempt: np.ndarray) -> Verdict:
        """Judge an Attempt (a boolean Path Mask) exactly as it is: no repair, no clean-up."""
        truth = self.solution()
        iou = float((attempt & truth).sum() / (attempt | truth).sum())
        if not attempt[self.start]:
            return Verdict(False, iou, "the start is not marked")
        if not attempt[self.goal]:
            return Verdict(False, iou, "the goal is not marked")
        if (attempt & self.walls).any():
            return Verdict(False, iou, "the attempt crosses a wall")
        reached = _flood(attempt, self.start)
        if self.goal not in reached:
            return Verdict(False, iou, "there is a gap between the start and the goal")
        if len(reached) != attempt.sum() or not _is_simple_path(reached, self.start, self.goal):
            return Verdict(False, iou, "the attempt has stray pixels, so it is not one simple path")
        return Verdict(True, iou)


def generate(n: int, seed: int) -> Puzzle:
    """A uniformly random perfect n-by-n-cell maze with a random Start and Goal, repeatable from `seed`."""
    rng = np.random.default_rng(seed)
    cells = [(r, c) for r in range(n) for c in range(n)]
    walls = np.ones((2 * n + 1, 2 * n + 1), dtype=bool)
    for r, c in cells:
        walls[2 * r + 1, 2 * c + 1] = False

    def neighbours(cell: Pixel) -> list[Pixel]:
        r, c = cell
        return [(r + dr, c + dc) for dr, dc in _STEPS if 0 <= r + dr < n and 0 <= c + dc < n]

    # Wilson's algorithm: loop-erased random walks from each cell until they hit the maze so far.
    in_maze = {cells[rng.integers(len(cells))]}
    for first in cells:
        walk: dict[Pixel, Pixel] = {}
        here = first
        while here not in in_maze:
            options = neighbours(here)
            walk[here] = options[rng.integers(len(options))]
            here = walk[here]
        here = first
        while here not in in_maze:
            there = walk[here]
            walls[here[0] + there[0] + 1, here[1] + there[1] + 1] = False  # carve the passage
            in_maze.add(here)
            here = there

    while True:
        a, b = (cells[i] for i in rng.choice(len(cells), size=2, replace=False))
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 2:
            break
    to_pixel = lambda cell: (2 * cell[0] + 1, 2 * cell[1] + 1)  # noqa: E731
    return Puzzle(walls, to_pixel(a), to_pixel(b))
