# unmaze

A diffusion model that solves mazes. Give it a maze with a start and a goal; it starts from random static and denoises it into the path that connects them. It trains on a plain CPU in about twenty minutes.

![The path crystallising out of the model's guesses](docs/demo.png)

*Left to right: the puzzle, the model's clean guess at eight moments of denoising (orange = confident it is path), and the true solution.*

## Try it

```bash
pip install -e ".[dev]"        # PyTorch, NumPy, Pillow (+ pytest)

python -m unmaze demo --seed 3 # solve a held-out maze with the shipped checkpoint -> demo.png, demo.gif
python -m unmaze solve --seed 3   # one held-out maze, printed in the terminal
python -m unmaze eval          # solve rate on 200 held-out mazes
```

Train your own (CPU is fine; the shipped checkpoint took about 20 minutes on 4 cores):

```bash
python -m unmaze train --maze-size 7 --steps 6000 --out runs/model.pt
python -m unmaze eval --checkpoint runs/model.pt
```

`--maze-size` is cells per side (the drawn grid is `2n+1` pixels). Any size works; the shipped model is 7×7.

## Results

The shipped checkpoint (2.7M parameters, trained 6,000 steps on 7×7 mazes, about 20 minutes on 4 CPU cores) on **1,000 held-out mazes** it never saw, judged strictly:

| sampling steps | solve rate |
|---:|---:|
| 1 | 98.7% |
| 2 | 100.0% |
| 5 | 100.0% |
| 50 (default) | 100.0% |

Reproduce with `python -m unmaze eval --count 1000 --sample-steps 50`. One step of "denoising" is already almost enough; a second step fixes the stragglers. That is the iterative part earning its keep.

Zero-shot on maze sizes it was **not** trained on (300 held-out mazes each, 50 steps):

| maze | grid | solve rate |
|---|---|---:|
| 5×5 | 11×11 | 100.0% |
| 6×6 | 13×13 | 100.0% |
| 8×8 | 17×17 | 100.0% |
| 9×9 | 19×19 | 99.7% |
| 11×11 | 23×23 | 96.7% |

Failures are mostly a gap in the path. Training on a larger size is one flag away (`--maze-size 11`), but only 7×7 is shipped and tested.

## How it works

1. **The data is endless.** Every training step generates fresh mazes (uniformly random *perfect* mazes, via Wilson's algorithm) with a random start and goal, and finds the true solution by breadth-first search. The model cannot memorise; the held-out mazes come from a seed range training never touches.
2. **Diffusion runs over the answer, not the maze.** The thing being noised is the *path mask*: a picture with the solution's pixels set to +1 and everything else −1. A small U-Net sees the noisy mask, three channels describing the puzzle (walls, start, goal), and the noise level, and predicts the clean mask. Self-attention at the two coarse resolutions lets it follow a corridor across the whole maze.
3. **Solving is denoising.** Starting from pure Gaussian noise, a deterministic DDIM sampler asks the model for its clean guess, steps a little less noisy, and repeats (50 steps by default). The final guess, thresholded at zero, is the answer.
4. **The judge is strict.** An answer counts as *solved* only if its pixels are exactly one simple path from start to goal through open pixels. No gaps, no stray blobs, no wall crossings, and **no repair step** afterwards. Otherwise a clean-up pass would be doing the solving.

An honest note on what diffusion is doing here: because a perfect maze has exactly one solution, the "distribution" the model learns is a single point. It is not sampling among many valid paths; the noise mostly serves as an iterative-refinement scaffold. Look at the strip above: the very first guess already has the right route plus a faint spurious branch, and later steps clean it up. A maze with many solutions (loops) would make the generative part more interesting, and is a natural next step.

## Repo map

| | |
|---|---|
| `unmaze/maze.py` | the maze world: generate, solve (BFS), judge, encode, render |
| `unmaze/diffusion.py` | noise schedule and the Solver (DDIM sampler, takes any denoiser) |
| `unmaze/model.py` | the U-Net denoiser |
| `unmaze/training.py` | training loop, checkpoints |
| `unmaze/evaluation.py` | the solve rate on held-out mazes |
| `unmaze/viz.py` | PNG strip and GIF of the denoising |
| `GLOSSARY.md` | the project's vocabulary (puzzle, attempt, verdict, …) |
| `docs/adr/` | the two decisions worth remembering, and why |
| `.scratch/unmaze/spec.md` | the spec this was built from |

## Tests

```bash
pytest                # fast: no training needed beyond tiny configs
pytest -m slow        # loads the shipped checkpoint and checks its solve rate through the CLI
```

The sampler is tested with an *oracle denoiser* that knows the answer: if the sampler and noise schedule are right, a perfect denoiser must lead to exactly the true solution, so any failure of a trained model is the model's, not the sampler's.

## How this was built

Built with [Matt Pocock's skills](https://github.com/mattpocock/skills): vocabulary first (`domain-modeling` → `GLOSSARY.md`, ADRs), then a spec with the test seams agreed up front (`to-spec`), then red-green TDD in vertical slices at those seams (`tdd`, `implement`), then a two-axis review against the spec and the coding standards (`code-review`).
