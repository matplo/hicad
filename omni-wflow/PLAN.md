# OmniLearned whole-event classification workflow

Historical design recorded before implementation. See README.md and RESULTS.md
for the implemented workflow and executed campaigns. This folder owns the new workflow;
the existing cumulant study and event explorer remain separate.

The objective is to distinguish pure BES 9.2 GeV background from background plus
proton clumps, treating each accepted event as one particle cloud (one “jet” to
OmniLearned). Measure performance as a function of signal physics, then establish
whether it exceeds simple proton-count and event-summary classifiers.

## 1. Generate particle-level training data

The cumulant scan stores compact accepted counts, not the particle clouds needed
by a transformer. Reuse its physics configurations, but generate new full
four-vectors with the shared massive 3D thermal background and clump generators.
Do not turn histogram bins or cumulant estimates into event training examples.
The existing 1,000-event background files are too small for a training campaign.

Use original geometry to select 0–5% centrality before injection. Both background
and source receive the identical **|eta| < 1**, full-azimuth cut, with no extra pT
cut for the first experiment. This matches the canonical event-display background
acceptance. It differs from the cumulant study's |y| and pT cuts; compare those
acceptances later as a separately labelled experiment. Save the exact cuts in
each dataset manifest. Never refill after cuts to force a target accepted count.

Use all six charged toy species initially. Store ragged four-vectors, charge,
species, event offsets, background family IDs, source parameters and injection
truth. Truth is for labels/diagnostics only. Species and clump membership must not
be passed accidentally into the classifier. Save float32 physical data; avoid
using the explorer's lossy encoded point cloud as model input.

Start with 20,000 unique central background parents. Split parent families
70%/15%/15% into train/validation/test **before creating their twins**. Every pure
background, clump variant, scrambled twin, mixed twin and parameter variant of a
parent stays in its split. Include donor ancestry if event mixing uses real
stored donors. Preprocessing fits and pretraining may use training families only.

Evaluate each parameter setting using the held-out test background pool and its
corresponding signal twins. Reusing backgrounds across test settings is useful
for paired comparisons, but does not increase independent statistics. Stream
variants or cache compact shards rather than duplicating backgrounds on disk.

## 2. Adapt OmniLearned to whole events

Use the installed PET2 architecture in classifier mode, with a new two-class
head. No physical jet clustering is needed. A full-azimuth heavy-ion event can
have nearly cancelling total transverse momentum, so avoid defining a jet axis
from that sum. Preserve laboratory eta and periodic phi relative to a fixed
reference, and apply common random azimuth rotations to whole events during
training. Do not independently rotate particles in the primary signal sample.

First establish an explicit **ideal-kinematics benchmark** with native-style
features [eta, phi, log(pT/GeV), log(E/GeV)]. The true E of identified massive
particles contains mass information: this benchmark is not a detector-level
track-only result. Compare a track-like representation using momentum and charge
without true energy/species, and an explicitly labelled ideal-PID variant.
Recompute model features identically after composing background and source.

The adapter must handle periodic phi in neighbor construction and test invariance
under crossing the ±pi boundary. Inspect interaction-feature assumptions before
activating jet-specific interaction blocks. Keep the installed package unchanged;
implement adaptations locally. Dynamic padding must preserve the real-particle
mask. The installed model infers padding from feature slot 2 != 0, so real zero
features or normalization must not silently remove particles. Test empty/small
clouds and K-neighbor limits. No silent top-pT truncation: it could preferentially
remove soft source protons. Benchmark complete events and record any later
selection explicitly.

## 3. Train and compare

Run a short CPU/device smoke test first, including finite loss, backward pass,
padding checks and serialization/reload. Benchmark memory and epoch time before
choosing batch size, model depth and full event count. Use explicit device
selection: CUDA, tested MPS, otherwise CPU. The installed CLI needs a local
training wrapper for this setup; see API_NOTES.md.

The first usable result should be a supervised PET2 classifier trained from
scratch. This gives a reference before introducing jet-to-heavy-ion transfer.
Use balanced classes and uniform sampling of training signal settings, retain
geometry distributions in both classes, monitor validation loss and select the
checkpoint without looking at test AUC. Use three training seeds. Keep source
truth, file ordering, seed values and generator metadata out of input features;
shuffle particles so appended clump particles do not expose their origin.

Then compare, at matched data and model capacity:

- Official jet-pretrained encoder with a new classifier head, initially frozen.
- Fine-tuning that encoder on the toy event task.
- Optional toy-domain pretraining followed by the same classifier evaluation.

OmniLearned's `pretrain` mode combines generative and classification objectives;
it is not automatically an unlabeled pretraining prescription. Any toy-domain
pretraining needs its objectives and class/conditioning information declared.
A single-class background generative stage is a separate optional experiment.
Do not assume jet-pretrained weights help in this very different momentum regime.
Inspect feature/architecture compatibility before loading a checkpoint; do not
silently ignore unmatched weights. No 4-TB jet corpus is required for this study.

## 4. Scan signal parameters

The proposed grid in configs/signal-grid.json has 144 physical settings:
T = 60,100,180 MeV; beta = 0,.4,.8; n = 2,4,8,16; k = 1,2;
transverse/isotropic 3D directions. These match a useful subset of the cumulant
campaign. Start with shared boost, then evaluate mixed/scrambled and mean-matched
independent additions with properly specified momentum laws.

Train one pooled classifier across the grid, then report test performance at
every setting. Train specialists at a few weak/intermediate/strong settings as
additional references, not as a claim of an optimal-performance bound. Reserve
predeclared parameter combinations for a generalization experiment, in addition
to keeping event families disjoint. Label interpolation and extrapolation tests.
An optional parameter-conditioned classifier can follow; background examples must
receive parameter inputs independently of their labels and with matched priors.

## 5. Define the labels and AUC correctly

The primary positive label means **a clump was injected before acceptance**.
Generate positives with injection probability 1, including events whose clump
protons all fail acceptance. Do not quietly discard such events or relabel them
using the observed source count: acceptance loss is part of detectability.
Report the zero-accepted-source fraction per setting and provide a separate,
explicitly conditioned diagnostic for at least one accepted source particle.

ROC AUC is an appropriate first metric: 0.5 corresponds to chance ranking and 1
to perfect separation. Also report ROC curves, average precision, and signal
efficiency at 10% and 1% background acceptance with event-family bootstrap
intervals. Show FPR resolution and never quote infinite background rejection when
no false positives were observed. Include class priors with precision metrics.
Low-rate discovery sensitivity is not determined by AUC alone.

Distinguish this from a **mixture positive class** in which only fraction f of
events receive signal. For a fixed score and unchanged background distribution,
including half credit for ties:

    AUC(background vs mixture_f) = (1-f)/2 + f * AUC(background vs injected)

Thus even a perfect injected-event classifier cannot exceed 0.505 at f=.01 for
that mixture-labelled comparison. This dilution is not evidence of failed
training. Train on injected events first, and evaluate deployment mixtures at
f = 0,.01,.05,.1,.25,.5,1 with the fixed classifier. Check the identity numerically.
In contrast, changing the prevalence of two genuinely labelled classes does not
itself change population ROC AUC. Preserve that distinction in the browser.

## 6. Establish what the network learned

Compare with total accepted multiplicity; proton/antiproton/net-proton counts
when PID is available; and a simple classifier on counts, sum/mean pT and selected
pair observables. Fit any baseline/threshold selection on training/validation
only. Use multiplicity/proton-count-matched or reweighted diagnostics with the
matching procedure and target distributions declared. Report both inclusive and
matched results; matching changes the statistical question.

The source adds particles, so a high inclusive AUC could simply reflect count
excess. Compare shared and scrambled/mixed controls at matched mean yields;
where feasible match one-particle spectra and counts to isolate correlation
information. Full-azimuth phi scrambling preserves the count cumulants but can
change classifier performance through angular correlations. Enforce that mixed
controls and Poisson additions retain the intended one-particle distributions.
At beta=0, shared-versus-scrambled discrimination is a useful null when only a
rotation-invariant independent thermal law is present.

## 7. Validation, output and implementation order

Before a large campaign: family-ID overlap audit; background-vs-background and
permuted-label AUC≈0.5 on independent test families; acceptance parity;
source-truth exclusion; rotation/permutation/padding checks; reload consistency;
training-seed variation; bootstrap intervals clustered by parent family.
Evaluate a separate generation seed as a robustness check. Do not tune on the
reported final test set or treat correlated grid points as independent tests.

Implement in this order:

1. Particle generator, grouped splits and data manifest.
2. PET2 adapter, device probe, small smoke training and simple baselines.
3. Pooled supervised training plus per-setting held-out evaluation.
4. Pretrained/fine-tuned comparisons and correlation-control diagnostics.
5. Expanded statistics if pilot uncertainty warrants it.

Planned commands are generate, train, evaluate and build-browser; they are not
implemented yet. Each run will retain config/version/seed manifests, selected
checkpoint, history, held-out scores, ROC arrays and tabulated metrics. Extend
our offline HTML approach with selectable AUC parameter maps, third-parameter
panels, model/input comparisons, uncertainty intervals, ROC/score distributions
and CSV downloads. Generated particles and weight files remain Git-ignored.

References: [official OmniLearned repository](https://github.com/ViniciusMikuni/OmniLearned),
[shared toy generators](../toys/boltzmann/README_canonical.md),
[existing count study](../studies/bes-cumulants/README.md).
