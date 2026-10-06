# Train on a mix of maze Styles, and judge every Style separately

The eval audit found the model solves 99.8% of mazes from the Style it trained on (uniform) but 66% of winding ones, and that a single blended number would have hidden that. So the Solver is now trained on a 40 / 40 / 20 mix of uniform, winding and bushy mazes, and `unmaze eval` reports each Style on its own. The cost is a model that must cover more ground with the same capacity, and a longer training run.

**Ship rule, fixed before the run so the result cannot move it:** the mixed checkpoint replaces the shipped one only if, on the untouched test split (`TEST_START`, 1,000 mazes per Style, 50 steps, measured once), winding is solved at least 90% of the time and uniform at least 99%. Bushy is reported either way. If it misses, the current checkpoint stays and the miss is reported plainly.

## Outcome

The mixed model (40 / 40 / 20 mix, U-Net width 64, 8,000 steps, about 78 minutes on 4 CPU cores) **missed the rule**. On the test split, 1,000 mazes per Style, 50 steps:

| Style | shipped model | mixed model |
|---|---:|---:|
| uniform | 99.7% | 99.8% |
| winding | 66.3% (CI 63.3 to 69.2) | **88.0%** (CI 85.8 to 89.9) |
| bushy | 100% | 100% |

Winding needed 90%; even the top of its interval is below. The shipped checkpoint was therefore left unchanged. The mixed model is better or equal on every Style, so replacing it anyway is a defensible call, but it is the project owner's call, not something the rule allows by itself.

Worth knowing: the 80-maze dev check at the end of training said winding 92%. The 1,000-maze test said 88%. Small dev samples flatter; that is what the test split is for. The test split (`TEST_START`) is now spent.
