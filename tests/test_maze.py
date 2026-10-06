"""Behaviour of the maze world: puzzles, their solutions, and the verdict on an attempt.

Expected values are hand-drawn, never recomputed by the code under test.
"""

import numpy as np
import pytest

from unmaze.maze import STYLES, Puzzle, generate

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


def test_the_heart_of_an_overlook_maze_is_always_its_centre_cell():
    for n in (3, 7, 11):
        for seed in range(30):
            puzzle = generate(n, seed)

            assert puzzle.goal == (n, n)  # cell (n // 2, n // 2) sits at pixel (n, n) on the Grid
            assert not puzzle.walls[puzzle.goal]


def test_the_entrance_is_a_cell_on_the_outer_ring_and_varies_over_all_four_sides():
    n = 7
    last = 2 * n - 1  # pixel of the outermost cell row/column
    puzzles = [generate(n, seed) for seed in range(400)]

    for p in puzzles:
        r, c = p.start
        assert not p.walls[p.start]
        assert r % 2 == 1 and c % 2 == 1  # on a cell, not a passage
        assert r in (1, last) or c in (1, last)  # on the outer ring
    assert {p.start[0] == 1 for p in puzzles if p.start[1] not in (1, last)} == {True, False}  # top and bottom
    assert {p.start[1] == 1 for p in puzzles if p.start[0] not in (1, last)} == {True, False}  # left and right
    assert len({p.start for p in puzzles}) == 4 * (n - 1)  # every ring cell turns up as the entrance


def test_an_overlook_maze_needs_an_odd_size_of_at_least_three_so_the_heart_is_truly_central():
    for n in (1, 2, 4, 8):
        with pytest.raises(ValueError, match="odd"):
            generate(n, 0)


def test_mazes_are_uniformly_random_among_perfect_mazes():
    # A 3x3 lattice has exactly 192 spanning trees (Kirchhoff), so each should turn up 1/192 of the time.
    samples = 30_000
    counts: dict[bytes, int] = {}
    for seed in range(samples):
        key = generate(3, seed).walls.tobytes()
        counts[key] = counts.get(key, 0) + 1

    assert len(counts) == 192
    assert all(100 <= count <= 220 for count in counts.values())  # mean 156, about 4.5 sigma either way


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


def test_there_are_three_styles_of_overlook_maze():
    assert STYLES == ("uniform", "winding", "bushy")


@pytest.mark.parametrize("style", STYLES)
def test_every_style_makes_perfect_repeatable_overlook_mazes_that_can_be_solved(style):
    n = 7
    for seed in range(20):
        puzzle = generate(n, seed, style)
        open_pixels = int((~puzzle.walls).sum())
        r, c = puzzle.start

        assert open_pixels == n * n + (n * n - 1)                    # a tree over all n*n cells...
        assert reachable_open_pixels(puzzle) == open_pixels          # ...that is connected
        assert puzzle.goal == (n, n)                                 # the heart is the centre
        assert r in (1, 2 * n - 1) or c in (1, 2 * n - 1)            # the entrance is on the outer ring
        assert puzzle.judge(puzzle.solution()).solved
        assert (generate(n, seed, style).walls == puzzle.walls).all()


def test_an_unknown_style_is_an_error_that_lists_the_choices():
    with pytest.raises(ValueError, match="winding"):
        generate(7, 0, "spiral")


def route_pixels(style: str, count: int, n: int = 11) -> float:
    return sum(int(generate(n, seed, style).solution().sum()) for seed in range(count)) / count


def dead_end_fraction(style: str, count: int = 60, n: int = 11) -> float:
    """Share of cells that have exactly one open passage out of them (a dead end)."""
    dead = 0
    for seed in range(count):
        walls = generate(n, seed, style).walls
        for r in range(n):
            for c in range(n):
                y, x = 2 * r + 1, 2 * c + 1
                exits = sum(not walls[y + dy, x + dx] for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                dead += exits == 1
    return dead / (count * n * n)


def test_winding_mazes_have_much_longer_routes_than_uniform_ones():
    assert route_pixels("winding", 100) > 1.5 * route_pixels("uniform", 100)


def test_winding_mazes_have_few_dead_ends_and_bushy_ones_many():
    # Known from the maze-generation literature: depth-first carving leaves about 10% of cells as
    # dead ends, randomised Prim about 36%.
    assert dead_end_fraction("winding") < 0.2
    assert dead_end_fraction("bushy") > 0.3
