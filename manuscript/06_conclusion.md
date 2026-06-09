# Conclusion {#sec:conclusion}

iTrace shows that auditable webcam eye-movement analysis follows from a single
architectural commitment: keep the verifiable algorithms pure and verify them
against ground truth, and keep the fragile hardware in a thin, optional shell. The
science is dependable *because of*, not in spite of, that separation. On the
verified core sits an analysis surface — descriptive event statistics,
distribution fitting and comparison, scanpath spread and entropy metrics, a
bootstrap interval on the main-sequence exponent, benchmark scoring for supplied
truth files, bounded live empirical diagnostics, and a deterministic figure
gallery — that extends the package's descriptive reach without weakening any
correctness claim. Each layer is tested, seed-pinned, and explicit about the
truth source it was handed.

The contribution is honestly scoped: a verified, type-safe, reproducible
reference implementation of gaze, saccade, pupil, and scanpath analysis
algorithms, with a 3-D-forward-model internal-consistency check and an idealized
noise-sensitivity analysis whose most defensible result is that webcam saccade
timing is theoretically marginal under sub-pixel-to-few-pixel landmark
perturbations. What remains is the empirical step that this paper deliberately
does not claim to have completed: real frames, a reference device or public
dataset, and the upper bound on error those would finally establish.
