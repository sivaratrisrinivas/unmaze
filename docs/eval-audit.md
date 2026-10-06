# Eval audit: how is unmaze actually performing?

Done with the `evals-start` router, which sent this to `eval-audit` (an existing eval pipeline, and the question is whether it can be trusted). The skills come from [ai-evals-course/evals-skills](https://github.com/ai-evals-course/evals-skills). They are written for LLM pipelines, so each check below was applied to what unmaze has instead: a deterministic code judge, a held-out solve rate, and exact ground truth.

Everything here is reproducible:

```bash
python scripts/measure.py          # the measurements below, ~25 min on 4 CPU cores -> docs/eval-audit-data.json
python scripts/measure_ood.py      # the out-of-distribution probe
python -m unmaze eval --count 1000 # the quick version, with a confidence interval
```

## The headline, corrected

**99.8% solved (4,990 / 5,000), 95% CI 99.6% to 99.9%**, on 5,000 fresh 11×11 mazes, 50 sampling steps, judged strictly with no repair. The range (seeds from 2,000,000,000) had never been used for any decision before this audit (a 200-maze smoke run of the script touched its first 200 seeds), and shares no maze with the training sample (checked against 20,000 training seeds: 0 overlaps, 0 duplicates).

That replaces the "99.9%" quoted earlier from the dev range. It was not wrong, but it sat at the lucky end of sampler noise: the dev range scored 999 / 1,000 in the earlier run and 995 / 1,000 with different sampler seeds, and 1,000 fresh mazes scored 994, 997 and 993 under three seeds.

**But that number only describes mazes from the generator the model trained on.** The two findings below are why it should not be read as "solves hedge mazes".

## Findings, by impact

### 1. Labeled data: the eval covers one maze generator, and the model does not transfer
**Status:** Problem exists.
Every held-out maze comes from the same generator as the training data (uniformly random spanning trees, Wilson's algorithm). On 500 mazes carved by randomised depth-first search instead, which makes the long, winding corridors a designed hedge maze tends to have, the model solves **332 / 500 = 66.4% (CI 62.1% to 70.4%)**. Routes of 80 to 160 pixels: 43 / 140 = 31%. Routes under 40 pixels: 127 / 136 = 93%.
**Fix:** Build the eval set from several generators and report each separately (`generate-synthetic-data`: dimensions generator × size × route length × entrance side). Train on a mixture, then re-measure. Not done here; it needs a retrain (about an hour).

### 2. Error analysis: the failure that exists is "long routes", and uniform sampling hides it
**Status:** Problem exists.
All 10 failures out of 5,000 fresh mazes are in the longest quarter of routes (42 pixels or more): 1,329 / 1,339 = 99.25% there, against 3,661 / 3,661 = 100% in the other three quarters. Failed routes average 76 pixels against 33 overall. A stress set of the 300 longest routes from a pool of 30,000 (average 91 pixels) is solved 279 / 300 = **93.0% (CI 89.5% to 95.4%)**. Entrance side makes no difference (99.7% to 100% on all four sides).
**Fix:** Report route-length strata in `eval` and keep a standing "longest routes" slice; train with more long routes (oversample the top route-length decile). Re-run `scripts/measure.py` after each change.

### 3. Error analysis: the failure modes, observed (not brainstormed)
**Status:** Problem existed (no failure taxonomy anywhere in the repo); first pass done here.
Taken from the failed attempts themselves, not from a list of generic categories:

| Observed failure | Fresh, 50 steps (10 failures) | Fresh, 1 step (64 failures) | 15×15, untrained size (50 failures) |
|---|---:|---:|---:|
| **A gap in the route** (at 1 step usually a hole of a pixel or two near the heart; at 50 steps a long stretch, median 41 pixels missing; at 15×15, 57) | 7 | 33 | 36 |
| **An end left unmarked** (the heart or the entrance itself is missing) | 3 | 0 | 6 |
| **A stray spur or blob** (extra pixels next to the route; usually 1 or 2 pixels) | 0 | 31 | 8 |
| **The path drawn through a hedge** | 0 | 0 | 0 |

Two things stand out. The model has never drawn through a wall (0 across 135 failures), so it has learned the hedge rule perfectly; what it loses is long-range *continuity*. And iteration changes the *kind* of failure: at 1 step most failures are near-misses (median IoU 0.93), at 50 steps what remains is catastrophic (median IoU 0.39, a long stretch lost).
**Fix:** The taxonomy above is the starting point. The remaining failures are about continuity over long routes, which points at model capacity and long-route training data (finding 2), not at the sampler.

### 4. Pipeline hygiene: the headline had no interval, and moved with the sampler seed
**Status:** Problem existed; fixed in code.
`unmaze eval` printed a bare percentage. Solving the same 1,000 fresh mazes under three sampler seeds gives 994, 997 and 993 solved (6, 3 and 7 failures), and failures are not about the mazes: 13 different mazes failed at least once and **none failed under all three seeds**.
**Fix (done):** `eval` now prints a Wilson 95% confidence interval. Quote results with the number of mazes, and the interval.
**Worth a decision:** because failures follow the noise, not the maze, resampling with a different seed whenever the answer is not a valid simple path would very likely recover most of them. The check needs no ground truth, only the maze and the attempt. This is not "repair" (nothing is edited), but it does change what is being measured, so it is the project owner's call against ADR 0002.

### 5. Labeled data: there was no test split
**Status:** Problem existed; fixed in code.
One held-out range (seeds 1,000,000,000 and up) was used while building (training progress checks, spotting the padding bug, choosing 50 sampling steps) *and* for reporting. No decision was knowingly tuned on it, but nothing prevented that.
**Fix (done):** `TEST_START = 4,000,000,000` and `unmaze eval --split test`. Use it for final numbers only; every time it informs a decision it is spent. Caveats: 100 of its mazes were used once in a smoke run (all solved) and informed nothing; the same goes for the first 200 seeds of the 2,000,000,000 range, which a smoke run of the measurement script touched. The 2,000,000,000 range used for this audit is now spent.

### 6. Pipeline hygiene: nothing re-runs the evals
**Status:** Problem exists.
There is no CI. The fast tests and the slow checkpoint tests run only when someone remembers.
**Fix (not done, your call):** a GitHub Actions workflow running `pytest` on pull requests, and `scripts/measure.py` before a release. Re-audit after any change to the model, the generator or the sampler.

### 7. Evaluator design: mean IoU is a vanity number here
**Status:** Minor problem.
Mean IoU is printed next to the solve rate. At 99.8% solved it reads `1.000`, while the failures that remain are catastrophic (median IoU 0.39). The pass criterion is binary and correct; the IoU is only a diagnostic and should not be quoted as a headline.
**Fix:** Quote the solve rate and its interval; keep IoU for diagnosing individual failures.

## What checked out

### Evaluator design: OK
Binary pass/fail, deterministic code, no LLM judge and no similarity metric as the pass criterion. A code check is exactly right for "is this one simple path from entrance to heart through open pixels?".

### Judge validation: OK
There is no LLM judge to calibrate against human labels, so the audit's TPR/TNR was measured against exact answers instead:
- On the 5,000 fresh attempts the judge agrees with exact comparison against the BFS solution **5,000 / 5,000** times (a perfect maze has one solution, so "solved" must equal "is the solution").
- All 5,000 exact solutions are judged solved (TPR 100%).
- **10,000** deliberately corrupted answers (2,000 each of: a deleted mid-route pixel, a deleted entrance or heart, a marked wall pixel, a spur beside the route, a detached blob) are all judged unsolved (TNR 100%), and **every one for the right reason**.
Limit: only single-defect corruptions were tried. Because "solved" is equivalent to "equals the solution", the judge adds a diagnosis, not extra strictness.

### Leakage: OK
0 of 5,000 fresh mazes appear among 20,000 sampled training seeds; 0 duplicates inside the fresh set.

### Human review: not needed for the label, but nobody has looked
Ground truth is exact, so no annotators are needed. The failures above were categorised by code. Looking at a handful with your own eyes (the web page can show misses) would still be worth ten minutes.

## Not done

- `error-discovery`'s interactive review app and `validate-evaluator`'s human-label calibration: both are for subjective LLM outputs with human reviewers, and there is nothing of that kind here.
- Retraining on a mixture of generators, and long-route oversampling (findings 1 and 2).
- CI (finding 6).

## Follow-up: the mixed-style retrain

Finding 1 asked for training on more than one Style of maze. That was done (ADR 0004) and measured once on the untouched test split, 1,000 mazes per Style, 50 steps.

| Style | shipped model | mixed model |
|---|---:|---:|
| uniform | 99.7% | 99.8% |
| winding | 66.3% (CI 63.3 to 69.2) | 88.0% (CI 85.8 to 89.9) |
| bushy | 100% | 100% |

**It helps a lot and still misses.** Winding rose 22 points, but the bar set beforehand was 90%, so the shipped checkpoint was not replaced. The remaining 120 winding failures look like the earlier ones: 49 gaps in the route, 43 stray pixels, and 28 where the entrance or heart is left unmarked. That is still long-range continuity on long routes, so the next levers are more training, a heavier winding share, or a larger model, not the sampler. Failures are also still noise-driven, so resampling on an invalid answer remains the cheap option, and it remains a decision against ADR 0002.

The test split is now spent. Use a new range for the next round.

