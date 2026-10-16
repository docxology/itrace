# iTrace scholarship and statistical interpretation review

Date: 2026-10-15. Reviewed base: `69d066a8a58a4691063a63c716eca71f9a5f2e77`.

## Scope and evidence

This independent research assignment concerns manuscript scholarship and statistical interpretation. It does not establish new participant, device, physiological, or publication evidence. Only this note was authored; manuscript, code, figures, ledgers, and sibling repositories were not changed, and nothing was committed or published.

Applicable ancestor, repository, `docs/`, and manuscript `AGENTS.md` instructions and `ISA.md` were read. `ISA.md` correctly separates constructed-truth verification from real-device validation, but some historical interpretations in ISC-78 and ISC-98/99 require the corrections below. Recommendations are against frozen Git blobs, read with `git show 69d066a8:<path>`, rather than other workers' live changes. All line numbers below refer to that base. Existing and subsequently arriving edits belong to their respective owners.

Selected blob identities:

| Base path | Git blob |
|---|---|
| `docs/manuscript/02i_descriptive_and_distribution_statistics.md` | `5f6a1a52647ac18182b881234cc6a6e6804c11f2` |
| `docs/manuscript/S01_statistical_methods.md` | `6808ed3b78fd530194a32438ea099af736e17564` |
| `src/itrace/scene.py` | `8a733a4c0ece2200a0565761c9f4c3e3e16818c7` |

Primary papers, authors' own research pages/manuscripts, and official software documentation were checked online. Bibliographic aggregators were discovery aids only. Access limitations are recorded in the source register; an abstract-level check is not represented as full-text verification. No Python environment, dependency installation, source execution, or new acquisition was needed. Simple arithmetic witnesses were evaluated with `awk`; these are mathematical examples, not experimental results.

## Prioritized corrections

P1 items change substantive interpretation or the agreement between methods and evidence. P2 items correct statistical description. An unsupported claim is distinguished from a demonstrated mathematical error; a proposed improvement is not evidence that its replacement method has already been validated.

### R1 — P1: bounded percentile intervals are not automatically valid confidence intervals

**Locations:** `docs/manuscript/S01_statistical_methods.md:14-20,24-37`; `02h_noise_model_and_statistics.md:35-43,54-60`; `03d_noise_sensitivity.md:41,47`; `02i_descriptive_and_distribution_statistics.md:135-144` (all under `docs/manuscript/`).

**Incorrect claim:** staying within the observed range makes an interval "always valid." Support preservation does not establish nominal repeated-sampling coverage. Efron's paper explicitly studies *approximate* intervals and accuracy refinements, rather than a coverage guarantee from bounded endpoints. [Efron 1987](https://doi.org/10.1080/01621459.1987.10478410).

**Mathematical negative control:** for 25 independent Bernoulli observations with true mean 0.99, an all-success sample yields percentile-bootstrap interval `[1,1]`, excluding 0.99. The probability of that sample is `0.99^25 = 0.777821359399`; thus universal 95% coverage is impossible. This is a counterexample to the wording, not a model of iTrace F1.

**Action:** replace "always valid" with "bound-respecting, approximate percentile-bootstrap interval; nominal coverage has not been established for these settings." Do not promise that BCa or a transformed interval automatically fixes boundary cases.

**Separate terminology correction:** the interval estimates Monte Carlo uncertainty in the expected metric under the specified noise generator and fixed scene. It is not itself a standard error. Report `MCSE(mean) = s_trial / sqrt(n_trials)` separately if desired. State the pseudorandom generator and seed policy as reproducibility controls; distinct seed integers alone are not a mathematical proof of independence. [Morris, White, and Crowther 2019, Sections 4-5](https://doi.org/10.1002/sim.8086).

**Display action:** show `mean [lower, upper]` in the noise table. A percentile interval can be asymmetric; `mean +/- half-CI` need not reproduce its endpoints. Retain pointwise wording for per-noise-level intervals; a shaded collection of them is not a simultaneous 95% band.

### R2 — P1: identify the resampling unit before assigning uncertainty

**Locations:** `docs/manuscript/02i_descriptive_and_distribution_statistics.md:26-32,61-64,129-144`; `02h_noise_model_and_statistics.md:54-60`; `04_discussion.md:28-42`; `03f_figure_gallery.md:40-58`.

**Unsupported interpretation:** calling event-wise bootstrap output a finite-record diagnostic does not supply its sampling assumptions. Amplitudes, fixation durations, and amplitude/velocity pairs from an ordered recording are not automatically exchangeable independent observations. Pairing amplitude and velocity preserves their within-event relationship, not serial or session dependence. No autocorrelation estimate for iTrace is asserted here.

**Action:** label existing event-wise resampling as an empirical-event perturbation/stability analysis conditional on the supplied event list. Keep it distinct from the noise sweep's whole independently generated trial values. For future inferential intervals, define the target and independent unit first: preserve participant/session clusters when those are the units; use contiguous blocks only with a justified stationary or locally stationary design, block-length sensitivity, and sufficient blocks. One participant cannot bootstrap between-participant variation. Künsch provides stationary block-bootstrap theory, not blanket approval for arbitrary nonstationary scanpaths. [Künsch 1989](https://doi.org/10.1214/aos/1176347265).

### R3 — P1: failed bootstrap fits must not become zero-width uncertainty

**Locations:** `docs/manuscript/02i_descriptive_and_distribution_statistics.md:141-144`; related figure claim at `03f_figure_gallery.md:48-58`.

**Incorrect interpretation:** the text says collapsing to the point estimate avoids a spurious narrow bound. It creates a zero-width bound. The frozen `src/itrace/stats/scanpath_metrics.py:368-384` discards resamples with fewer than three unique amplitudes or a fitting error and returns `(b,b,b)` if none survive.

**Action for the implementation owner:** retain the point estimate but mark interval availability false when no valid fits remain. Report attempted, successful, and failed counts, reasons, and the fact that any surviving interval conditions on fit success. Do not silently replace failure with precision. Simulation reporting should examine failed estimates because their absence can depend on the sampled data. [Morris et al. 2019, Section 5.1](https://doi.org/10.1002/sim.8086). Official SciPy also treats degenerate bootstrap distributions as requiring inspection rather than universal inferential reassurance; its documented NaN warning concerns BCa specifically. [SciPy `bootstrap`, Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

### R4 — P1: clarify small-sample AIC/AICc policy without overselling AICc

**Locations:** `docs/manuscript/02i_descriptive_and_distribution_statistics.md:49-64,73-78`; `03f_figure_gallery.md:40-58`; `04_discussion.md:28-33`.

**Qualification needed, not proof that AIC ranking is intrinsically wrong:** the base `output/figures/statistical_diagnostics.json` compares **15 amplitudes**, not its 1,500 gaze samples. It ranks by AIC, reports AICc and BIC, uses AIC weights, and bootstraps AIC winners. The displayed AICc expression is familiar, but Hurvich and Tsai derive corrections for regression/autoregressive models; their paper does not establish exact small-sample calibration for every gamma, log-normal, Weibull, or ex-Gaussian marginal fit. [Hurvich and Tsai 1989](https://doi.org/10.1093/biomet/76.2.297).

**Action:** explicitly give `n_events`, free-parameter count, fixed parameters, candidate set, and ranking criterion. If AICc is chosen as a small-sample heuristic, use that criterion consistently for deltas, weights, winner frequencies, and selected QQ/CDF model; retain AIC/BIC as sensitivity diagnostics. With the current formula, `n=15` adds 1 for `k=2` and `24/11` for `k=3`. These arithmetic differences do not show that the current winner changes. Report AICc undefined for `n <= k+1`; do not silently claim that ordinary AIC resolves the information shortage. Dependence also requires qualification of the product marginal likelihood, not an arbitrary replacement of `n` with a guessed effective sample size.

### R5 — P1: correct distribution support and the actual candidate set

**Locations:** `docs/manuscript/02i_descriptive_and_distribution_statistics.md:40-49`; `04_discussion.md:28-30`.

**Incorrect claims:** the ex-Gaussian is not a positive-support distribution. A Gaussian plus an independent exponential has support on the whole real line when Gaussian scale is positive. The official implementation says its argument is real. [SciPy `exponnorm`, Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html).

The publication payload actually compares **normal, gamma, log-normal, Weibull, and ex-Gaussian**, as specified in frozen `src/itrace/stats/diagnostics.py:20-26`; the manuscript describes three families. Frozen `stats/distributions.py:208-230` delegates fitting to SciPy, rather than implementing the manuscript's described bespoke optimization over `(mu,sigma,tau)`.

**Action:** describe gamma/log-normal/Weibull as positive-support candidates and normal/ex-Gaussian as unconstrained descriptive candidates that can assign negative mass. State SciPy's `(K,loc,scale)` parameterization and `K=tau/sigma`, with location fixed only where it actually is fixed. If positive truncation is later introduced, normalize the likelihood accordingly before comparing models; fitting only positive data is not itself a truncated model. Avoid asserting that all fixation-duration/amplitude samples are intrinsically right-skewed.

### R6 — P1: retain CDF distances, but qualify KS probabilities and DKW coverage separately

**Locations:** `docs/manuscript/02i_descriptive_and_distribution_statistics.md:65-69,82-95`; `03f_figure_gallery.md:58,65-68`.

The existing same-data fitted-parameter caveat is appropriate. **Remaining unsupported implication:** a DKW width is not distribution-free against arbitrary dependence. Massart's inequality assumes an iid sample from a common CDF. Label the plotted quantity an **iid DKW reference width**; no calibrated 95% coverage for ordered detected events is established here. [Massart 1990](https://doi.org/10.1214/aop/1176990746).

**Action:** keep KS `D`, QQ/P-P residuals, and explicitly named AD-style/CvM-style distances as descriptions. Label ordinary fixed-CDF KS probabilities computed after fitting as **uncalibrated**, rather than merely advising caution or implying that a large value accepts the fit. If a future inferential test is wanted under an iid fitted-family null, refit unknown parameters in each simulated sample. A claim about a data-selected family would additionally need to account for selection and any dependence. Official SciPy documents refitting and the error caused by treating estimated parameters as known. [SciPy `goodness_of_fit`, Algorithmic overview and Anti-Pattern](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html). That procedure is an option to validate later, not a test run in this review.

### R7 — P1: make future agreement evaluation stronger than correlation

**Locations:** `docs/manuscript/05_limitations.md:72-79`; preserve the correct consistency-check qualification in `03c_closed_loop.md:16-19` and `04_discussion.md:120-122`.

**Inadequate proposed evidence:** "error in degrees and pupil correlation" does not adequately specify agreement of pupil measurements. Correlation is insensitive to additive offset and positive rescaling: hypothetical `y=2x+1` can have `r=1` while disagreeing in measurement units. [Bland and Altman 1986, Inappropriate use of correlation coefficient](https://www-users.york.ac.uk/~mb55/meas/ba.htm).

**Action:** retain correlation as a temporal-shape descriptor. For future calibrated same-unit pupil measurements, prespecify paired differences, bias, absolute-error scale, difference-versus-mean plots, and acceptable disagreement. Estimate limits of agreement only under an appropriate design and distributional assumptions. Repeated samples from a continuous trace or person must not be treated as independent subjects. [Bland and Altman 2007](https://www-users.york.ac.uk/~mb55/meas/ba2007.htm). A relative proxy versus millimetres first requires a declared calibration/normalization target; a Bland-Altman plot alone cannot create comparable units. No new agreement result is claimed.

### R8 — P1: overlap F1 does not measure timing accuracy, and sweep matching differs from the prose

**Locations:** `docs/manuscript/02h_noise_model_and_statistics.md:44-49`; `S01_statistical_methods.md:39-46`; `03d_noise_sensitivity.md:24-31`; `05_limitations.md:14-17`; `06_conclusion.md:18-20`.

**Frozen implementation evidence:** `src/itrace/scene.py:246-248,265-280` supplies the sweep's F1 through many-to-many any-overlap counts. One detected interval spanning all true intervals can attain precision=recall=1; splitting one true interval into several overlapping detections can do likewise. Thus `02h`'s claim that fragmentation and merging are penalized is false for this sweep. The separate `stats/events.py:17-62` uses greedy one-to-one matching; its existence does not change the sweep path.

**Action:** accurately document the scorer actually used, or have its owner change it with retained controls and regenerate dependent results. Do not imply that existing numerical thresholds already represent a corrected scorer. Replace the "timing breaks down" conclusion with "overlap-based event recovery degrades under this perturbation model." An overlap criterion can also score a severely extended single event perfectly despite wrong boundaries. Future timing claims need matched-event onset/offset/duration errors and a declared tolerance or overlap-quality requirement. Eye-movement evaluation should specify the scoring unit: REMoDNaV's paper, for example, explicitly distinguishes sample-label comparison against expert annotations. [Dar, Wagner, and Hanke 2021, Validation analyses](https://link.springer.com/article/10.3758/s13428-020-01428-x).

### R9 — P1: the synthetic residual is not a proven lower bound on real error

**Locations:** `docs/manuscript/03c_closed_loop.md:8-23`; `04_discussion.md:111-122`; `02h_noise_model_and_statistics.md:25-27`; `06_conclusion.md:20-22`.

**Unsupported bound:** omitting error sources does not prove real-world RMS must exceed the model residual. Error components may be correlated, canceled, or altered by calibration. A nonzero residual also does not logically prove independent formulations; distinct source equations establish that design fact. The conclusion's promised empirical "upper bound on error" likewise needs a specified population, conditions, statistic, and coverage statement.

**Action:** use "residual under shared idealizations; no quantitative bound on real-device error is established." Describe recovery within tested tolerances rather than exact inversion or strongest-possible proof. Use "empirical error characterization under the evaluated conditions" for future work. Verification and operational validation are separate judgments in simulation methodology. [Sargent 2013](https://doi.org/10.1057/jos.2012.20). This correction preserves the usefulness of the closed loop while withdrawing an unsupported inequality; it does not predict whether measured errors will be larger or smaller.

### R10 — P1: narrow the field-wide novelty and software-proof statements

**Locations:** `docs/manuscript/01_introduction.md:26-33`; `04_discussion.md:5-11,98-109`; `06_conclusion.md:3-6`.

**Unsupported generalizations:** the introduction implies the cited ecosystem cannot exercise scientific algorithms without hardware and commonly validates without an oracle. REMoDNaV explicitly accepts gaze files, has no particular hardware requirement, and compares against expert-labelled data. The introduction itself later acknowledges file-oriented tools. [Dar et al. 2021, Operation and Algorithm comparison](https://link.springer.com/article/10.3758/s13428-020-01428-x).

**Action:** replace the blanket deficit with the project's specific contribution: one package combining optional capture, typed records, constructed-truth fixtures, and reproducible analyses. Do not imply synthetic or decoupled testing was absent from earlier tools without a defined comparative survey. Replace "a passing synthetic suite proves the estimation mathematics is correct" and "without putting the verified core at risk" with evidence of behavior on the tested cases and reduced coupling. New input distributions can still violate assumptions even when interfaces are stable. The verification/validation distinction does not license an exhaustive-correctness claim. [Sargent 2013](https://doi.org/10.1057/jos.2012.20).

### R11 — P2: entropy normalization and cross-record comparability are misstated

**Location:** `docs/manuscript/02i_descriptive_and_distribution_statistics.md:116-127`.

**Incorrect implementation description:** frozen `src/itrace/stats/scanpath_metrics.py:214-265,268-317` returns spatial Shannon entropy and weighted conditional transition entropy in bits; neither function divides by maximum alphabet entropy. The latter is `H(next | current)`, not simply entropy of the joint transition distribution. The spatial extent defaults to each recording's own data range.

**Action:** document bits and the conditional formula, or have the implementation owner explicitly add and name normalized outputs. A common alphabet/grid and screen extent are prerequisites for a like-for-like descriptive comparison. Dividing by a maximum would still not remove finite-sample estimation bias or make unequal-length recordings inferentially comparable. Paninski analyzes substantial sampling problems in information estimation. [Paninski 2003](https://doi.org/10.1162/089976603321780272). Show event/transition counts and label sparse-row uncertainty rather than interpreting an unobserved transition as established zero probability.

## Opportunities that are not demonstrated errors

### O1 — P2: support the introductory pupil motivation with primary experiments

**Locations:** `docs/manuscript/01_introduction.md:3-8`; `05_limitations.md:19-29`.

The general motivation is defensible but uncited at its point of use. Add primary evidence linking pupil diameter to memory load in the specific task studied by Kahneman and Beatty, and to neuromodulatory activity in the mouse experiments of Reimer and colleagues. Avoid presenting pupil diameter as a specific readout of human cognitive load or one transmitter. [Kahneman and Beatty 1966](https://pubmed.ncbi.nlm.nih.gov/5924930/), [Reimer et al. 2016](https://www.nature.com/articles/ncomms13289). These studies motivate measurement; neither validates iTrace's relative live proxy.

### O2 — P2: replace unsourced optical constants with a precise confound account

**Location:** `docs/manuscript/04_discussion.md:115-120`; corresponding assumptions in `05_limitations.md:64-68`.

The uncited `~0.5-1 mm` apparent-pupil difference and `~5 degrees` kappa statement are not verified universal constants in this review. Remove the numbers unless their definitions, optical conditions, and primary evidence are supplied. Fedtke et al.'s optical model is an appropriate source for entrance-pupil dependence on viewing geometry, not a universal additive size correction. [Fedtke, Manns, and Ho 2010](https://doi.org/10.1364/OE.18.022364). Hayes and Petrov's controlled artificial-eye experiments directly establish gaze-position-dependent pupil measurement error; add this concrete confound without claiming that iTrace implements their correction. [Hayes and Petrov 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4637269/).

### O3 — P2: make future benchmark targets and reference versions explicit

**Locations:** `docs/manuscript/01_introduction.md:14-17,38-49`; `05_limitations.md:72-79`; `docs/manuscript/references.bib`.

Separate **gaze** benchmarks from pupil-size and temporal-event validation. GazeCapture is a phone/tablet gaze dataset; MPIIGaze concerns gaze during laptop use. Their cited records do not by themselves establish synchronized reference pupil diameter or physiological saccade onset truth. Choose evaluation labels and units from each dataset's actual protocol. [GazeCapture authors' project](https://gazecapture.csail.mit.edu/), [MPIIGaze author preprint](https://arxiv.org/abs/1711.09017).

The checked L2CS-Net record correctly lists Abdelrahman first; its legacy key `hempel2022l2cs` is not an authorship error. The current PupilSense entry correctly reflects the 2025 v2 title rather than the 2024 v1 title; pin `arXiv:2407.11204v2` to make that choice explicit. [L2CS-Net](https://arxiv.org/abs/2203.03339), [PupilSense v2](https://arxiv.org/abs/2407.11204v2). Do not rename bibliography keys merely to match author/year styling. Prefer verified final publication metadata where available, without mixing preprint year and journal volume.

## Verified source register and short excerpts

Excerpts are deliberately short. Publisher access failures did not justify borrowing claims from citing papers. Titles and metadata below identify the source versions actually checked.

| ID | Bibliographic metadata and primary link | Verification and permitted use |
|---|---|---|
| S1 | Efron, B. (1987). *Better Bootstrap Confidence Intervals*. Journal of the American Statistical Association 82(397), 171-185. DOI [10.1080/01621459.1987.10478410](https://doi.org/10.1080/01621459.1987.10478410). | Publisher-indexed abstract/metadata checked; direct page later returned 403. Excerpt: "approximate confidence intervals". The separately checked [Stanford report](https://statistics.stanford.edu/technical-reports/better-bootstrap-confidence-intervals) is dated November 1984, not a different publication year for the journal article. |
| S2 | Künsch, H. R. (1989). *The Jackknife and the Bootstrap for General Stationary Observations*. The Annals of Statistics 17(3), 1217-1241. DOI [10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265). | Publisher security wall; metadata independently checked on [author's ETH publication list](https://people.math.ethz.ch/~hkuensch/papers/), abstract on the [author-uploaded article record](https://www.researchgate.net/publication/2355926_The_Jackknife_and_the_Bootstrap_for_General_Stationary_Observations). Excerpt: "We do not attempt a reduction to i.i.d. values." No claims rely on ResearchGate's citing-paper summaries. |
| S3 | Hurvich, C. M., and Tsai, C.-L. (1989). *Regression and time series model selection in small samples*. Biometrika 76(2), 297-307. DOI [10.1093/biomet/76.2.297](https://doi.org/10.1093/biomet/76.2.297). | Publisher-indexed abstract/metadata checked; direct retrieval failed. Excerpt: "derived for regression and autoregressive time series models". This verifies the stated derivation scope, not universal exactness of a density-fitting correction. |
| S4 | Morris, T. P., White, I. R., and Crowther, M. J. (2019). *Using simulation studies to evaluate statistical methods*. Statistics in Medicine 38(11), 2074-2102. DOI [10.1002/sim.8086](https://doi.org/10.1002/sim.8086). | Publisher text and [authors' institutional deposit](https://discovery.ucl.ac.uk/id/eprint/10066118/) checked; Sections 4-5 support Monte Carlo uncertainty and failure reporting. |
| S5 | Massart, P. (1990). *The Tight Constant in the Dvoretzky-Kiefer-Wolfowitz Inequality*. The Annals of Probability 18(3), 1269-1283. DOI [10.1214/aop/1176990746](https://doi.org/10.1214/aop/1176990746). | Publisher security wall; [author-uploaded article abstract](https://www.researchgate.net/publication/38360933_The_Tight_Constant_in_the_Dvoretzky-Kiefer-Wolfowitz_Inequality) checked. Excerpt: "a sample of n i.i.d. random variables". Use for the inequality's assumptions, not fitted-model acceptance. |
| S6 | Bland, J. M., and Altman, D. G. (1986). *Statistical methods for assessing agreement between two methods of clinical measurement*. The Lancet 327(8476), 307-310. DOI [10.1016/S0140-6736(86)90837-8](https://doi.org/10.1016/S0140-6736(86)90837-8). | [Author-hosted full text](https://www-users.york.ac.uk/~mb55/meas/ba.htm), including stated numerical corrections, checked. Excerpt on correlation: "not the agreement between them". |
| S7 | Bland, J. M., and Altman, D. G. (2007). *Agreement between methods of measurement with multiple observations per individual*. Journal of Biopharmaceutical Statistics 17(4), 571-582. DOI [10.1080/10543400701329422](https://doi.org/10.1080/10543400701329422). | [Author-hosted publication page](https://www-users.york.ac.uk/~mb55/meas/ba2007.htm) and article abstract/metadata checked. Supports distinguishing repeated observations from independent individuals; no full derivation reproduced. |
| S8 | Dar, A. H., Wagner, A. S., and Hanke, M. (2021). *REMoDNaV: robust eye-movement classification for dynamic stimulation*. Behavior Research Methods 53, 399-414. DOI [10.3758/s13428-020-01428-x](https://link.springer.com/article/10.3758/s13428-020-01428-x). | Publisher full text checked, especially Operation and Validation analyses. Supports file-based operation and expert-annotation comparison; does not establish consumer-webcam performance. |
| S9 | Sargent, R. G. (2013). *Verification and validation of simulation models*. Journal of Simulation 7(1), 12-24. DOI [10.1057/jos.2012.20](https://doi.org/10.1057/jos.2012.20). | Publisher-indexed abstract/metadata checked; a discovered university PDF URL returned 404. Used for the distinction between verification and operational validation, not a theorem about iTrace error bounds. |
| S10 | Paninski, L. (2003). *Estimation of Entropy and Mutual Information*. Neural Computation 15, 1191-1253. DOI [10.1162/089976603321780272](https://doi.org/10.1162/089976603321780272). | [Research-group record](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Paninski03) and [paper PDF](https://www.cns.nyu.edu/pub/lcv/paninski-infoEst-2003.pdf) checked. Supports caution about sampling bias; it is not validation of iTrace entropy outputs. |
| S11 | Kahneman, D., and Beatty, J. (1966). *Pupil diameter and load on memory*. Science 154(3756), 1583-1585. DOI [10.1126/science.154.3756.1583](https://doi.org/10.1126/science.154.3756.1583). | [Original article abstract/metadata in PubMed](https://pubmed.ncbi.nlm.nih.gov/5924930/) checked; publisher direct retrieval failed. Supports task-specific memory-load motivation only. |
| S12 | Reimer, J., McGinley, M. J., Liu, Y., Rodenkirch, C., Wang, Q., McCormick, D. A., and Tolias, A. S. (2016). *Pupil fluctuations track rapid changes in adrenergic and cholinergic activity in cortex*. Nature Communications 7, 13289. DOI [10.1038/ncomms13289](https://www.nature.com/articles/ncomms13289). | Publisher full text checked. Primary mouse experiments support neuromodulatory associations, without making pupil size a specific human cognitive-load measurement. |
| S13 | Hayes, T. R., and Petrov, A. A. (2016). *Mapping and correcting the influence of gaze position on pupil size measurements*. Behavior Research Methods 48(2), 510-527. DOI [10.3758/s13428-015-0588-x](https://doi.org/10.3758/s13428-015-0588-x). | [Original abstract/metadata](https://pubmed.ncbi.nlm.nih.gov/25953668/) and indexed [article text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4637269/) checked for artificial-eye experiments; author PDF retrieval timed out. Journal issue year is 2016 despite the 2015 DOI/online date. |
| S14 | Fedtke, C., Manns, F., and Ho, A. (2010). *The entrance pupil of the human eye: a three-dimensional model as a function of viewing angle*. Optics Express 18(21), 22364-22376. DOI [10.1364/OE.18.022364](https://doi.org/10.1364/OE.18.022364). | [Original article record](https://pmc.ncbi.nlm.nih.gov/articles/PMC3408927/) metadata checked; subsequent retrieval encountered a challenge. Used as a narrowly scoped optical-model source candidate, not verification of the manuscript's numerical constants. |
| S15 | SciPy community. Official `exponnorm`, `bootstrap`, and `goodness_of_fit` reference pages (accessed 2026-10-15; served as v1.18.0). | [Distribution support](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html), [bootstrap definition/degeneracy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), [fitted-null procedure](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html). Excerpt from `exponnorm`: "where x is a real number". Documentation of mathematical/API semantics, not proof of the local installed version or a test executed here. |

Existing-reference checks: Krafka, Khosla, Kellnhofer, Kannan, Bhandarkar, Matusik, and Torralba, *Eye Tracking for Everyone*, CVPR 2016, 2176-2184, verified through the [authors' paper/project](https://gazecapture.csail.mit.edu/cvpr2016_gazecapture.pdf); Zhang, Sugano, Fritz, and Bulling, *MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation*, [arXiv:1711.09017](https://arxiv.org/abs/1711.09017), DOI `10.48550/arXiv.1711.09017` (2017 preprint); Abdelrahman, Hempel, Khalifa, and Al-Hamadi, *L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments*, [arXiv:2203.03339](https://arxiv.org/abs/2203.03339), DOI `10.48550/arXiv.2203.03339` (2022 preprint); Shah, Watanabe, Moser, and Dengel, *PupilSense: A Novel Application for Webcam-Based Pupil Diameter Estimation*, [arXiv:2407.11204v2](https://arxiv.org/abs/2407.11204v2), DOI `10.48550/arXiv.2407.11204`, revised 2025-03-29. No unverified final-publication metadata is substituted for these records.

## Finite implementation handoff

1. Resolve R1-R3 and R8 before interpreting intervals or regenerated sweep thresholds. Preserve failing controls; do not relabel existing values as results of a changed method.
2. Resolve R4-R7 consistently across methods, payloads, figure annotations/captions, and discussion. Retain descriptive outputs where inferential assumptions are unestablished.
3. Correct R9-R11 and consider O1-O3 in prose/bibliography. Synchronize current ISA acceptance wording without rewriting its historical record of past decisions as if it had always been correct.
4. The implementing owner should run the applicable gates and rerender/inspect affected figures and manuscript. This research note does not claim those implementation or artifact checks passed.

The review supports a manuscript about reproducible implementation checks and bounded workflow diagnostics. It supplies no new empirical validation and no authorization to publish.
