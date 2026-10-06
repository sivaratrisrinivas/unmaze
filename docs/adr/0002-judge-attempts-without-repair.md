# Judge Attempts exactly, with no repair step

An Attempt is Solved only if its thresholded pixels form exactly one simple path from Start to Goal. We deliberately do not clean it up afterwards (no BFS between the endpoints, no removal of stray blobs), even though a one-line repair would lift the Solve Rate. A repaired Attempt would measure the repair, which is a classical maze solver, not the diffusion model. The Verdict's `reason` and `iou` exist so failures stay diagnosable without hiding them.
