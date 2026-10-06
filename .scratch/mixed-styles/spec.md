Status: implemented and trained; the model missed the ship rule (see ADR 0004, Outcome), so the shipped checkpoint is unchanged

# Mixed-style training

## Problem Statement

The eval audit (docs/eval-audit.md) showed the shipped model only reliably solves the kind of maze it trained on: 99.8% uniform, 66% winding (depth-first), with all failures on long routes. "Solves hedge mazes" is only honest if it holds across ways of building a hedge maze.

## Solution

Three Styles of Overlook maze (uniform, winding, bushy). Training draws from a weighted mix. Evaluation reports each Style separately. A retrained model ships only if it clears a rule fixed in advance (ADR 0004).

## User Stories

1. As a maintainer, I want to make mazes in three Styles from one function, so that training and tests can ask for any of them.
2. As a maintainer, I want every Style to be a perfect maze with the heart at the centre and the entrance on the outer ring, so that one judge works for all.
3. As a maintainer, I want each Style to behave the way the literature says (winding: long routes, few dead ends; bushy: many dead ends), so that the labels mean something.
4. As a trainer of the model, I want to set which Styles to train on and how often, so that I can weight the hard one.
5. As a trainer of the model, I want a run with one Style to draw random numbers exactly as before, so that old runs stay reproducible.
6. As a reader of results, I want `unmaze eval` to report each Style separately, so that a strong Style cannot hide a weak one.
7. As a reader of results, I want the Styles a checkpoint was trained on stored in it, so that the numbers can be read correctly later.
8. As a skeptic, I want the ship rule written down before the run, so that the result cannot move it.

## Implementation Decisions

- `generate(n, seed, style)`; `STYLES = (uniform, winding, bushy)`; unknown style is an error listing the choices. Winding's random-number order matches the audit probe exactly, so the 66% finding reproduces.
- `TrainConfig.styles` / `style_weights`; default is uniform only (so old checkpoints read truthfully); the `train` command defaults to the 40 / 40 / 20 mix.
- `draw_batch(config, rng)` is the one place a training batch is made.
- `evaluate(..., style=)`; `eval --style all` is the default.
- Model: same U-Net, base width 64 instead of 48 (for long routes), 8,000 steps.

## Testing Decisions

At the existing seams. Maze: perfect / repeatable / solvable for every Style; winding routes more than 1.5 times uniform; dead-end shares match the literature (winding under 20%, bushy over 30%). Training: draws in proportion to weights and labels match the maze; default is uniform-only; Styles survive a checkpoint round trip. Evaluation: asking for a Style evaluates that Style's mazes.

## Out of Scope

Long-route oversampling beyond what the winding Style provides; verifier-guided resampling (a decision against ADR 0002); GPU training; CI.
