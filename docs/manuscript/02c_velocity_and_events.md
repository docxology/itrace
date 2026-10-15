# Methods: velocity and event detection {#sec:events}

## Velocity

Gaze velocity is estimated two ways, matching how data arrives. For uniformly
sampled streams, the implementation uses a Savitzky–Golay local-polynomial
derivative [@savitzky1964smoothing] as a deterministic
smoothing-and-differentiation choice; the window is clamped to an odd length not
exceeding the trace. For non-uniform timestamps a
central-difference gradient is taken on the actual time vector. Two-dimensional
speed is the Euclidean norm of the per-axis velocities.

The default velocity dispatcher checks timestamp spacing before selecting its
estimator: intervals constant within a relative tolerance of 0.001 use the
uniform path; irregular intervals use the actual time vector. Differentiation
rejects nonfinite, duplicate, and decreasing timestamps. Stream containers retain
such clocks for quality diagnostics, but they cannot be treated as valid timing
for event dynamics.

## Fixations and saccades (I-VT, I-DT)

**I-VT** [@salvucci2000identifying] labels each sample saccadic when its 2-D
speed exceeds a velocity threshold and collapses contiguous same-label runs into
events; runs shorter than a minimum duration are reabsorbed into the surrounding
fixation, suppressing single-sample velocity spikes. In noisy or intermittently
tracked streams the configurable `merge_gap_s` parameter optionally bridges short
subthreshold holes inside one high-velocity movement before the duration filter
is applied. Only finite, analysable samples can be bridged; missing gaze and
nonfinite velocity split the candidate intervals and are never labelled as
fixations. The historical merge default remains zero. **I-DT**
[@salvucci2000identifying] greedily extends a window while its spatial dispersion
$(\max x - \min x) + (\max y - \min y)$ stays below threshold, emitting one
fixation per maximal low-dispersion window.
It likewise splits windows at nonfinite gaze samples.

## Microsaccades (Engbert–Kliegl)

Microsaccades follow @engbert2003microsaccades. Velocity uses
their five-sample moving-average estimator, and a per-axis elliptic threshold

$$ (v_x / \eta_x)^2 + (v_y / \eta_y)^2 > 1 $$ {#eq:ek}

([@eq:ek]) flags an event when held for at least a minimum number of samples. The threshold
$\eta = \lambda\sigma$ uses the **median-based** robust scale estimator
$\sigma^2 = \mathrm{med}(v^2) - \mathrm{med}(v)^2$, clamped at zero, so a few
velocity outliers cannot inflate the threshold the way a plain standard deviation
would — a property the test suite checks directly ([@sec:results]).
