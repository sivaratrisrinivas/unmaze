Status: implemented (see README.md for results; spec kept in sync with what was built)

# Unmaze: a diffusion model that solves mazes

## Problem Statement

I want to see a diffusion model solve mazes, end to end, on my own machine without a GPU. Today there is nothing in the repo: no way to make mazes, no model, no way to tell whether a model's answer is actually a solution, and no way to watch the solving happen. A diffusion model's output is a picture, not a guaranteed path, so "it looks about right" is not good enough. I need an honest measure of whether it solved the maze.

## Solution

A small Python package, `unmaze`, with four things behind simple commands:

- It generates random perfect mazes (each with a random start and goal) and knows their true solutions.
- It trains a conditional diffusion model, on CPU in minutes, that turns random noise into the path from start to goal for a maze it is shown.
- It judges every answer strictly: solved only if the answer is exactly one simple path from start to goal, with no repair afterwards.
- It draws the denoising, so I can watch a path crystallise out of static.

A trained checkpoint is committed so `python -m unmaze demo` works straight after cloning.

## User Stories

1. As a learner, I want to run one command and see a diffusion model solve a maze, so that I understand the idea without reading the code first.
2. As a learner, I want to watch the path emerge from noise step by step, so that I can see what the model is doing at each noise level.
3. As a learner, I want the demo to work from a fresh clone, so that I am not blocked on training.
4. As an experimenter, I want to train a model with one command on a CPU, so that I do not need a GPU.
5. As an experimenter, I want to set the maze size, number of training steps, and sampling steps from the command line, so that I can trade quality against speed.
6. As an experimenter, I want training to print a loss and periodically a Solve Rate on held-out mazes, so that I know whether training is working before it finishes.
7. As an experimenter, I want training to be reproducible from a seed, so that two runs with the same seed give the same checkpoint.
8. As an experimenter, I want training to save a checkpoint that records the maze size and model settings, so that loading it later needs no guessing.
9. As an experimenter, I want an `eval` command that reports the Solve Rate of a checkpoint on a fixed held-out set, so that I can compare checkpoints fairly.
10. As an experimenter, I want the held-out mazes to be generated from a seed range training never touches, so that the Solve Rate is not inflated by memorisation.
11. As an experimenter, I want to know why an unsolved attempt failed (gap, stray blob, wall crossed, wrong endpoints), so that I can tell where the model is weak.
12. As an experimenter, I want a soft overlap number (iou) for each attempt, so that I can see near-misses that a yes/no verdict hides.
13. As a skeptic, I want an attempt to count as solved only if it is exactly one simple path from start to goal, so that the headline number means something.
14. As a skeptic, I want no clean-up step between the model's output and the verdict, so that the number measures the diffusion model and not a classical solver in disguise.
15. As a skeptic, I want the ground-truth solution to come from an independent breadth-first search, so that the model is never graded against its own conventions.
16. As a skeptic, I want a blank answer (all wall) to be judged unsolved, so that a degenerate model cannot score well by doing nothing.
17. As a skeptic, I want an answer that fills the whole maze to be judged unsolved, so that it cannot score well by shouting everything.
18. As a maze author, I want every generated maze to be perfect (one route between any two cells), so that each puzzle has exactly one solution.
19. As a maze author, I want mazes to be uniformly random among perfect mazes, so that the generator does not bias the model toward one style of corridor.
20. As a maze author, I want start and goal to be random distinct cells that are not adjacent, so that the model cannot memorise their location and no puzzle is trivial.
21. As a maze author, I want the same seed to give the same puzzle, so that experiments and tests are repeatable.
22. As a maze author, I want an ASCII rendering of any puzzle and attempt, so that I can read a failure in the terminal.
23. As a maintainer, I want the sampler to work with any denoiser handed to it, so that I can test the sampling maths without a trained network.
24. As a maintainer, I want a perfect (oracle) denoiser to always yield the exact solution through the sampler, so that a failure in a real model can be blamed on the model and not the sampler.
25. As a maintainer, I want sampling to be deterministic for a given seed, so that results and demo images are repeatable.
26. As a maintainer, I want the forward-noising step and noise schedule covered by tests with known values, so that schedule bugs are caught before a long training run.
27. As a maintainer, I want tests to run fast with no training, so that I can run them on every change.
28. As a maintainer, I want one slow test that loads the committed checkpoint and checks it still solves a fixed set of held-out mazes above a floor, so that a regression in the model path is caught.
29. As a maintainer, I want a glossary and ADRs in the repo, so that the next person uses the same words and knows why diffusion runs over the path mask and why there is no repair step.
30. As a user on a laptop, I want the denoiser to accept any maze size from the command line (odd grid padded internally), so that I can try bigger mazes without editing code.
31. As a user, I want `solve` to take a seed and print the maze, the attempt, and the verdict, so that I can look at one case at a time.
32. As a user, I want `demo` to save a PNG strip or GIF of the denoising trace to a path I choose, so that I can share it.

## Implementation Decisions

- **Maze world is one deep module** with a small interface: generate a Puzzle from (size, seed), get its Solution, judge an Attempt, and encode a Puzzle into the model's conditioning channels. Everything about grids, walls, cells and shortest paths lives behind it.
- **Puzzle:** a perfect maze on an n×n cell lattice drawn on a (2n+1)×(2n+1) Grid, with a random distinct, non-adjacent Start and Goal. Generated with Wilson's algorithm so mazes are uniform spanning trees.
- **Solution** is computed by breadth-first search on the Grid, independently of anything the model does. In a perfect maze it is unique.
- **Verdict:** `solved` (exactly one simple path over open pixels from Start to Goal), `iou` against the Solution, and a `reason` string when not solved. No repair of the Attempt. See ADR 0002.
- **Diffusion runs over the Path Mask**, scaled to ±1, conditioned on three channels: wall, Start, Goal. See ADR 0001.
- **Schedule:** cosine noise schedule, 1000 training timesteps. The Denoiser predicts the clean Path Mask directly (x0-prediction), trained with mean-squared error.
- **Sampler:** deterministic DDIM-style stepping over a configurable, much smaller number of steps than training used (default 50). It takes any Denoiser, so tests inject an oracle. The Attempt is the final clean guess thresholded at zero. It can also return the Denoising Trace.
- **Denoiser:** a small conditional U-Net with timestep embedding, coordinate channels, and self-attention at the lowest resolution so information can travel the whole maze. Pads the Grid internally to a multiple of 4 and crops back; the padding is plain wall (no path, no start, no goal) and the position channels are absolute pixel positions, so the amount of padding never leaks into what the model sees.
- **Training:** on-the-fly generated Puzzles (never stored), so the model sees far more mazes than it can memorise. Held-out evaluation uses a disjoint seed range.
- **CLI:** `python -m unmaze` with `train`, `eval`, `solve`, `demo`. A committed checkpoint backs `solve` and `demo` by default.
- **Hardware:** CPU only, default maze 7×7 cells (15×15 Grid). Training should finish in tens of minutes on 4 cores.
- **Dependencies:** PyTorch, NumPy, Pillow (only for drawing), pytest.

## Testing Decisions

A good test here states a fact about behaviour that a person could check by hand, with an expected value that comes from the spec or a hand-drawn example, never recomputed by the code under test. No mocking of our own modules; the only injected thing is the Denoiser, which is a real seam.

**Pre-agreed seams** (no test is written outside these):

1. **The maze world seam** (`generate`, `solution`, `judge`). Tested with hand-drawn literal mazes whose solution is known by eye, plus properties of generated mazes (perfect, solvable, seed-deterministic).
2. **The Solver seam** (`Solver(denoiser).solve(puzzles, steps, seed)`). Tested by injecting an oracle Denoiser that returns the true solution, and degenerate Denoisers (all-wall, all-open) that must be judged unsolved. This tests the Sampler and schedule through the public interface, not their internals.
3. **The command-line seam** (`python -m unmaze eval`). One slow test against the committed checkpoint on a fixed held-out seed set, asserting a Solve Rate floor.

Added during implementation, because each is a public interface the CLI is a thin shell over: the **Denoiser's size handling** (padding is wall, tested by comparing against padding drawn in by hand), **training** (`train`, `load_solver`: loss goes down, same seed gives the same run, a saved checkpoint solves identically after reload) and **evaluation** (`evaluate`: strict Solve Rate, failure reasons, always the same held-out set). Both are tested with tiny configs so they stay fast.

The ideal is one seam; this is the least that covers world, sampler, training, and trained model without testing the network's weights directly. The schedule and forward noising are tested only through seam 2 (the oracle must round-trip), plus a small number of known-value checks on the schedule's public shape (starts clean, ends noise).

**Prior art:** none in the repo.

## Out of Scope

- GPU training and any CUDA-specific code.
- Mazes with loops (imperfect mazes), weighted cells, or multiple solutions.
- Non-grid mazes (hex, 3D, irregular).
- Rendering mazes as photographs; input is always the clean wall Grid.
- Any repair, snapping, or classical clean-up of Attempts.
- Classifier-free guidance, latent diffusion, and other speed or quality tricks, unless the baseline fails to reach a useful Solve Rate.
- Serving the model over a network or a web UI.

## Further Notes

- Vocabulary is in `GLOSSARY.md`; the two deliberate, surprising decisions are in `docs/adr/`.
- If the baseline Solve Rate is poor, the levers in order are: more training steps, wider U-Net, more sampling steps, then a different parameterisation. Record what was tried.
