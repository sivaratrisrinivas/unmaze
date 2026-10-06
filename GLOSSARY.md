# Unmaze

A diffusion model that solves mazes: given a maze with a start and a goal, it denoises random noise into the path that connects them.

## Language

### The puzzle

**Maze**:
A perfect maze on an n×n lattice of cells: exactly one simple route joins any two cells (a spanning tree of the lattice). Drawn on a **Grid**.
_Avoid_: labyrinth, board, map

**Grid**:
The (2n+1)×(2n+1) boolean picture of a Maze in which every pixel is either **wall** or **open**. Cells sit on odd coordinates; the pixel between two adjacent cells is open only when the wall between them is carved.
_Avoid_: image (that is what the Grid becomes inside the model, not what it is), matrix

**Puzzle**:
A Maze together with a **Start** and a **Goal**: the thing a Solver is asked to solve.
_Avoid_: problem, instance, task

**Start** / **Goal**:
The two distinct cells a Solution must connect. Both are random, never fixed corners, so the model cannot memorise their position.
_Avoid_: source/target, entrance/exit

**Solution**:
The unique path from Start to Goal through open pixels, stored as a **Path Mask**. Because a Maze is perfect, the Solution is also the shortest path.
_Avoid_: answer, route, label

**Path Mask**:
A boolean Grid marking the pixels on a path (the cells and the passages between them), Start and Goal included.
_Avoid_: segmentation, heatmap

### The solver

**Solver**:
The thing that turns a Puzzle into an Attempt: a Denoiser driven by a Sampler. It is the only thing a caller needs to know about.
_Avoid_: agent, model (the model is only the Denoiser inside it)

**Denoiser**:
The neural network that, given a noisy Path Mask, the Puzzle and a noise level, predicts the clean Path Mask.
_Avoid_: network, backbone

**Sampler**:
The reverse-diffusion loop that starts from pure noise and repeatedly asks the Denoiser for its best clean guess, stepping a little less noisy each time.
_Avoid_: generator, decoder

**Attempt**:
The Path Mask a Solver ends up with after the final denoising step, thresholded to wall/open. It may be wrong.
_Avoid_: prediction, output, sample

**Denoising Trace**:
The sequence of intermediate clean-guess Path Masks recorded while an Attempt is being produced. Shows the path crystallising out of noise.
_Avoid_: history, frames

### Judging

**Verdict**:
The result of judging an Attempt against a Puzzle: whether it is **solved**, how much it overlaps the Solution (**iou**), and a plain-English **reason** when it is not solved.
_Avoid_: score, result

**Solved**:
An Attempt is solved only when its pixels are exactly one simple path from Start to Goal over open pixels. No stray pixels, no gaps, no detours. The Attempt is never repaired before judging.
_Avoid_: correct, valid

**Solve Rate**:
The fraction of Puzzles in a held-out set whose Attempt is Solved. The headline number for any trained Solver.
_Avoid_: accuracy

## Relationships

- A **Puzzle** has exactly one **Solution**.
- A **Solver** produces one **Attempt** per **Puzzle**, and may record a **Denoising Trace** on the way.
- A **Verdict** compares one **Attempt** with one **Puzzle**.
