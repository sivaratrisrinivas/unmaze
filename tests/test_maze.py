"""Behaviour of the maze world: puzzles, their solutions, and the verdict on an attempt.

Expected values are hand-drawn, never recomputed by the code under test.
"""

import numpy as np
import pytest

from unmaze.maze import Puzzle, generate

# Four cells joined S -> top-right -> bottom-right -> G. The only wall between
# cells is the pixel at row 2, col 1.
HOOK = """\
#####
#S  #
### #
#G  #
#####"""

HOOK_SOLUTION = """\
#####
#...#
###.#
#...#
#####"""


def mask_from(text: str) -> np.ndarray:
    return np.array([[c == "." for c in row] for row in text.splitlines()])


def test_solution_of_hand_drawn_maze_is_the_path_you_can_trace_by_eye():
    puzzle = Puzzle.parse(HOOK)

    assert (puzzle.solution() == mask_from(HOOK_SOLUTION)).all()


def test_the_exact_solution_is_judged_solved_with_perfect_overlap():
    puzzle = Puzzle.parse(HOOK)

    verdict = puzzle.judge(mask_from(HOOK_SOLUTION))

    assert verdict.solved
    assert verdict.iou == 1.0


def test_a_blank_attempt_is_unsolved_because_the_start_is_not_marked():
    puzzle = Puzzle.parse(HOOK)

    verdict = puzzle.judge(np.zeros((5, 5), dtype=bool))

    assert not verdict.solved
    assert verdict.iou == 0.0
    assert "start is not marked" in verdict.reason


def test_a_path_with_a_hole_in_it_is_unsolved_because_of_the_gap():
    puzzle = Puzzle.parse(HOOK)
    attempt = mask_from(HOOK_SOLUTION)
    attempt[2, 3] = False  # knock out the pixel joining the top and bottom corridors

    verdict = puzzle.judge(attempt)

    assert not verdict.solved
    assert "gap" in verdict.reason


def test_an_attempt_that_marks_a_wall_is_unsolved_because_it_crosses_a_wall():
    puzzle = Puzzle.parse(HOOK)
    attempt = mask_from(HOOK_SOLUTION)
    attempt[2, 1] = True  # the wall between the start and the goal

    verdict = puzzle.judge(attempt)

    assert not verdict.solved
    assert "wall" in verdict.reason


def test_an_attempt_that_marks_the_border_is_judged_without_crashing():
    puzzle = Puzzle.parse(HOOK)
    attempt = mask_from(HOOK_SOLUTION)
    attempt[0, :] = True
    attempt[:, 4] = True

    verdict = puzzle.judge(attempt)

    assert not verdict.solved


# A 3x3-cell maze with a long dead-end corridor hanging off the middle of the route.
DEAD_ENDS = """\
#######
#S    #
##### #
#     #
# ### #
#   #G#
#######"""

DEAD_ENDS_SOLUTION = """\
#######
#.....#
#####.#
#    .#
# ###.#
#   #.#
#######"""


def test_solution_ignores_dead_ends():
    puzzle = Puzzle.parse(DEAD_ENDS)

    assert (puzzle.solution() == mask_from(DEAD_ENDS_SOLUTION)).all()


def test_a_path_with_a_spur_is_unsolved_because_of_stray_pixels_and_overlap_is_nine_tenths():
    puzzle = Puzzle.parse(DEAD_ENDS)
    attempt = mask_from(DEAD_ENDS_SOLUTION)
    attempt[3, 4] = True  # one open pixel poking into the dead-end corridor

    verdict = puzzle.judge(attempt)

    assert not verdict.solved
    assert "stray" in verdict.reason
    assert verdict.iou == 0.9


def test_a_path_plus_a_detached_blob_is_unsolved_because_of_stray_pixels():
    puzzle = Puzzle.parse(DEAD_ENDS)
    attempt = mask_from(DEAD_ENDS_SOLUTION)
    attempt[5, 1] = True  # open pixel far from the route

    verdict = puzzle.judge(attempt)

    assert not verdict.solved
    assert "stray" in verdict.reason


def test_an_attempt_that_stops_short_of_the_goal_is_unsolved_because_the_goal_is_not_marked():
    puzzle = Puzzle.parse(DEAD_ENDS)
    attempt = mask_from(DEAD_ENDS_SOLUTION)
    attempt[5, 5] = False

    verdict = puzzle.judge(attempt)

    assert not verdict.solved
    assert "goal is not marked" in verdict.reason


def test_an_attempt_that_marks_every_pixel_is_unsolved():
    puzzle = Puzzle.parse(DEAD_ENDS)

    verdict = puzzle.judge(np.ones((7, 7), dtype=bool))

    assert not verdict.solved


def reachable_open_pixels(puzzle: Puzzle) -> int:
    seen, stack = {puzzle.start}, [puzzle.start]
    while stack:
        r, c = stack.pop()
        for nxt in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if nxt not in seen and not puzzle.walls[nxt]:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen)


def test_a_generated_maze_is_perfect():
    n = 7
    for seed in range(20):
        puzzle = generate(n, seed)
        open_pixels = int((~puzzle.walls).sum())

        assert puzzle.walls.shape == (2 * n + 1, 2 * n + 1)
        # n*n cells joined by exactly n*n - 1 passages is a tree; reaching all of it makes it connected.
        assert open_pixels == n * n + (n * n - 1)
        assert reachable_open_pixels(puzzle) == open_pixels


def test_the_same_seed_gives_the_same_puzzle_and_a_different_seed_gives_a_different_one():
    a, b, c = generate(7, 3), generate(7, 3), generate(7, 4)

    assert (a.walls == b.walls).all() and a.start == b.start and a.goal == b.goal
    assert not (a.walls == c.walls).all()


def test_start_and_goal_are_distinct_non_adjacent_cells_and_vary_between_puzzles():
    puzzles = [generate(7, seed) for seed in range(200)]

    for p in puzzles:
        assert not p.walls[p.start] and not p.walls[p.goal]
        assert p.start[0] % 2 == 1 and p.start[1] % 2 == 1  # on a cell, not a passage
        assert p.goal[0] % 2 == 1 and p.goal[1] % 2 == 1
        assert abs(p.start[0] - p.goal[0]) + abs(p.start[1] - p.goal[1]) >= 4  # at least two cells apart
    assert len({p.start for p in puzzles}) > 10
    assert len({p.goal for p in puzzles}) > 10


def test_mazes_are_uniformly_random_among_perfect_mazes():
    # A 2x2 lattice is a 4-cycle: it has exactly 4 spanning trees, one per omitted passage.
    counts: dict[bytes, int] = {}
    for seed in range(2000):
        key = generate(2, seed).walls.tobytes()
        counts[key] = counts.get(key, 0) + 1

    assert len(counts) == 4
    assert all(400 <= count <= 600 for count in counts.values())  # 25% each, wide margin


def test_the_true_solution_of_a_generated_puzzle_is_judged_solved():
    for seed in range(100):
        puzzle = generate(7, seed)

        assert puzzle.judge(puzzle.solution()).solved


def test_encoding_a_puzzle_gives_wall_start_and_goal_channels():
    puzzle = Puzzle.parse(HOOK)

    channels = puzzle.encode()

    assert channels.shape == (3, 5, 5) and channels.dtype == np.float32
    walls, start, goal = channels
    assert (walls == np.array([[c == "#" for c in row] for row in HOOK.splitlines()])).all()
    assert start.sum() == 1 and start[1, 1] == 1
    assert goal.sum() == 1 and goal[3, 1] == 1


def test_rendering_round_trips_a_puzzle_and_overlays_an_attempt():
    puzzle = Puzzle.parse(HOOK)

    assert puzzle.render() == HOOK
    assert puzzle.render(mask_from(HOOK_SOLUTION)) == """\
#####
#S..#
###.#
#G..#
#####"""


def test_a_maze_with_no_route_says_so_instead_of_crashing_obscurely():
    walled_off = Puzzle.parse("#####\n#S#G#\n#####")

    with pytest.raises(ValueError, match="no route"):
        walled_off.solution()
