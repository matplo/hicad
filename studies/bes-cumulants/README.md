# BES 9.2 GeV cumulant study

Implemented and run: synthetic Au+Au 9.2 GeV proton/net-proton cumulants with
100,000 independently generated **selected 0–5% central events**, a joint scan
of 1,008 signal settings (12,096 joint result rows), and an offline HTML browser.
The initial pilot of separate parameter slices contains 1,152 result rows. An independent seed repeats the primary
strength points with another 100,000 selected events.

Open [the joint-scan browser](results/joint-100k/index.html) directly; no server or
internet connection is needed. Select strength, temperature/boost or clump-number
scans; trends, heatmaps or distributions; particle type, statistic, acceptance,
centrality treatment and comparison models. Choose “All sizes (overlay)” or use
mean accepted signal yield as the horizontal axis. Model checkboxes default to
shared boost only. Background is green, shared signal red, mixed blue, Poisson
gold. Export selected rows to CSV or load another generated results.json.
Filters select computed results; they do not launch simulations.

## Joint parameter scan

The new joint campaign crosses all of these signal parameters:

| Parameter | Values |
| --- | --- |
| T [MeV] | 60, 100, 180 |
| Beta | 0, .4, .8 |
| Protons per clump n | 2, 4, 8, 16 |
| Event injection fraction f | 0, .01, .05, .1, .25, .5, 1 |
| Clumps per injected event k | 1, 2 |
| Direction | Transverse, isotropic 3D |

This is 1,008 joint signal settings, 12,096 results across three source models and
four rapidity windows, plus 336 reference strength-scan results. The same 100,000
independent central backgrounds are paired across settings; multiplying settings
does not create additional independent background events. Background T stays at
100 MeV. The full campaign took about 142 seconds before final validation/exports.

The joint browser starts with f versus beta, panelled by n, with T=100 MeV,
k=1, isotropic 3D directions. Choose any two heatmap axes and a third panel
parameter; the remaining controls select a slice through the grid. All panels
share a color scale; signed logarithmic colors make small effects visible next
to large ones. Choose a trend overlay to compare temperatures, clump sizes, beta,
f or k on the same axes. Switching to trend view gives uncertainty bars; hovering
map cells gives the value and interval. CSV exports exactly the selected slice.
No smoothing or interpolation across unsimulated parameter settings is applied.

At n=8/f=.1/k=1, isotropic beta=.8, the primary CBWC net-proton C4/C2 changes from
11.963 [11.295,12.799] at T=60 MeV to 8.371 [7.715,8.997] at 100 MeV and
4.749 [4.224,5.273] at 180 MeV. At beta=0 the temperature trend instead rises.
Temperature/boost interactions are therefore visible beyond an isolated strength
scan. These are total background-plus-source ratios, not background differences.

The joint validation checks 85,248 correlated analytic cumulant additions, with
99.85% within three paired standard errors, and retains the exact transverse
mixing null. Eleven scientific tests and joint/pilot JavaScript filter tests pass.
A standalone four-panel PNG/SVG is in `results/joint-100k/figures/joint_f_beta_n.*`.
The embedded joint HTML is approximately 60 MB; opening it can take a moment.

## Reproduce

From the repository root, using the project environment (NumPy, SciPy, Matplotlib):

```bash
.venv/bin/python studies/bes-cumulants/scripts/run_study.py --config studies/bes-cumulants/configs/joint-scan.json --output studies/bes-cumulants/results/joint-100k
.venv/bin/python studies/bes-cumulants/scripts/run_study.py
.venv/bin/python studies/bes-cumulants/scripts/run_study.py --config studies/bes-cumulants/configs/seed-check.json --output studies/bes-cumulants/results/seed-check-100k
```

Reruns resume completed tasks. A configuration/version fingerprint prevents
reusing an output directory with different settings. For a smoke run, use
`--events 10000 --strength-only --output /tmp/hicad-bes-smoke`. For larger
statistics, choose a new output directory and override `--events`; the count must
be divisible by the configured bootstrap-block count. Geometry, background and
source count prefixes are reproducible as event count increases.

Outputs per run:

- `index.html`: interactive browser with embedded results and validation.
- `results.json`: estimates, intervals, analytic predictions and provenance.
- `validation.json`: four-vector, analytic, bootstrap and convergence checks.
- `figures/`: standalone PNG and editable SVG scientific figures.
- `checkpoints/`: geometry, compact counts and completed scan tasks.
- `run_identity.json`: configuration, software versions and fingerprint.

Generated data/results are Git-ignored. No HDF5 files are generated. Existing
background files and event-display code are unchanged.

## Physics and estimators

Background and source both use the shared massive 3D thermal law
`exp[-(E-m)/T] d³p`, with the proton mass from the canonical core. Background T is
100 MeV and beta zero. Original BES toy Glauber geometry, multiplicity and species
probabilities are reused. Centrality is selected from geometry before injection.
New independent events replace the small existing HDF5 pool.

Primary acceptance: `|y| < 0.5, 0.4 < pT < 2 GeV`, full azimuth. Additional
rapidity half-widths are .25, .75 and 1.0. Full emitted parent distributions are
integrated without an inherited pseudorapidity cut; the initially proposed wide
parent file is unnecessary for these count observables.

The fast count engine integrates Lorentz-transformed thermal acceptance, then
samples conditional binomial/multinomial counts. This reproduces the count law of
independent thermal daughters sharing the specified clump boost; it does not
create momentum point clouds. The lab density includes the invariant-measure
Jacobian E*/E. Converged Gaussian quadrature and adaptive direction tables control
numerical error. Acceptance is checked against actual four-vectors from the
shared thermal sampler.

A source event occurs with probability f; it contains k independent clumps, each
emitting exactly n protons. Shared clumps have one common direction; the synthetic
event-mixed control independently redraws each daughter's direction from the same
ensemble. Both use the same event gate and backgrounds. The Poisson control is
**ungated**, matching expected accepted mean f*k*n*<A>; this removes the rare
source-event mixture while retaining the average excess.

Fixed transverse beta with full azimuth gives identical shared/mixed count laws;
the count sampler deliberately couples them exactly. This is a null result for
counts, not for momentum correlations. An isotropic 3D common boost changes the
shared rapidity acceptance between events and can separate count cumulants. Phi
scrambling preserves global accepted counts exactly and has no separate curve.

Finite-sample-corrected k-statistics give C1–C4, C2/C1, C3/C2, C4/C2 for protons,
antiprotons and signed net protons. Proton/antiproton factorial cumulants are
F2=C2−C1, F3=C3−3C2+2C1, F4=C4−6C3+11C2−6C1; normalized options divide by the
mean to the corresponding power. They are excluded for signed net counts.
Centrality-bin-width correction (CBWC) averages cumulants in five 1%-wide geometry
bins before forming ratios. Raw 0–5% results are also available. This does not
remove all volume fluctuations.

The same 400 bootstrap weight vectors resample 200 independent event blocks for
all backgrounds/signals. Difference intervals include covariance. Approximate
percentile intervals are checked against an exact paired event bootstrap of joint
count cells at representative points. Block/event standard-error ratios are 0.93
for baseline CBWC C4/C2 and 1.10 for the paired signal-minus-background ratio.
Rare-class/fourth-order estimates need larger samples for precision work.

## Completed grid

- Strength: n=1,2,4,8,16,32; f=0,.01,.05,.1,.25,.5,1; one clump,
  T=100 MeV, transverse beta=.4.
- Shape: T=60,80,100,140,180 MeV; beta=0,.2,.4,.6,.8;
  transverse and isotropic 3D directions; n=8, f=.1, one clump.
- Clump count: k=2,4; n=8, f=.1, T=100 MeV, beta=.4; both directions.
- All three models and all four rapidity windows at every point.

## First results

Primary-acceptance CBWC net-proton C4/C2, with 95% bootstrap intervals.
Strength examples use T=100 MeV, transverse beta=.4, one clump.

| Setting | C4/C2 | 95% interval |
| --- | ---: | ---: |
| Background | 1.137 | [0.885, 1.402] |
| n=8, f=.01 | 3.080 | [2.634, 3.545] |
| n=8, f=.1 | 7.953 | [7.292, 8.520] |
| n=16, f=.05 | 51.356 | [49.814, 52.823] |
| n=8, f=1 | 0.631 | [0.402, 0.854] |

The n=8/f=.1 and n=16/f=.05 cases have the **same expected accepted signal mean**
(.530 protons/event), yet very different higher cumulants. A rare-class mixture
therefore cannot be summarized by average yield alone. Even the sign changes:
n=32/f=.5 gives C4/C2≈−209. These follow from this ordinary count model and are
not critical-point evidence. At f=1 the emitted count is fixed but accepted count
still fluctuates through acceptance. Adding a fixed *accepted* count instead
changes C1 only.

For isotropic 3D T=100 MeV, beta=.8, n=8/f=.1, the CBWC ratios are 8.371 shared,
4.312 mixed and 1.141 Poisson. A shared longitudinal boost adds acceptance
fluctuations beyond the event-class gate.

The independent seed gives background 1.022 [0.789,1.252] and n=8/f=.1 signal
7.221 [6.703,7.737]. The shift is about 1.8 combined standard errors; the
qualitative effect persists. Event-prefix checks at 10k/50k/100k also agree within
pilot uncertainties. Increase statistics at selected points before drawing
quantitative conclusions; these correlated scan rows are not independent tests.

## Verification and limits

```bash
.venv/bin/python -m unittest discover -s studies/bes-cumulants/tests -v
node studies/bes-cumulants/tests/check_browser.mjs studies/bes-cumulants/results/pilot-100k/index.html
```

Ten scientific/integration tests pass. Nine direct four-vector acceptance checks
are within 1.34 sampling standard errors. Of 8,640 correlated cumulant-addition
checks, 99.86% lie within three paired standard errors. Tests cover normalization,
k-statistics, acceptance nesting, RNG prefixes, shared-direction fluctuations,
Poisson factorial closure, fixed accepted shifts, CBWC, zero-signal closure and
checkpoint restart. The JavaScript harness checks filters/SVG generation against
actual embedded results. Native browser preview was unavailable in this session;
PNG figures were visually inspected.

This is a generator-level toy sensitivity study, not a calibrated RHIC data fit.
Detector efficiency/PID, realistic flow, resonance/feed-down effects, exact global
energy/baryon conservation and experimental corrections are absent. Background
and signal are independent apart from the specified common source boost. Fixed-n
and Poisson comparisons are implemented; additional fluctuating clump sizes and
detector response remain future extensions.
