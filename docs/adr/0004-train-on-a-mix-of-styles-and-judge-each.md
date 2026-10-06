# Train on a mix of maze Styles, and judge every Style separately

The eval audit found the model solves 99.8% of mazes from the Style it trained on (uniform) but 66% of winding ones, and that a single blended number would have hidden that. So the Solver is now trained on a 40 / 40 / 20 mix of uniform, winding and bushy mazes, and `unmaze eval` reports each Style on its own. The cost is a model that must cover more ground with the same capacity, and a longer training run.

**Ship rule, fixed before the run so the result cannot move it:** the mixed checkpoint replaces the shipped one only if, on the untouched test split (`TEST_START`, 1,000 mazes per Style, 50 steps, measured once), winding is solved at least 90% of the time and uniform at least 99%. Bushy is reported either way. If it misses, the current checkpoint stays and the miss is reported plainly.
