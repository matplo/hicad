# OmniLearned whole-event workflow

Implemented particle generation, PET2 training, toy-domain denoising pretraining,
encoder transfer/fine-tuning, held-out classification, simple baselines, paired
AUC uncertainties and a standalone results browser. The original design remains
in [PLAN.md](PLAN.md); installed-package observations are in [API_NOTES.md](API_NOTES.md).

See [RESULTS.md](RESULTS.md) for the executed three-seed pilot and independent-generation check.

## Run

From the hicad repository root with the existing venv:

```bash
# End-to-end smoke test: 400 parent events, two signal settings, one model seed.
.venv/bin/python omni-wflow/workflow.py all --config omni-wflow/configs/run-smoke.json --output omni-wflow/runs/smoke-periodic

# Pilot: 4,000 parents, 144 signal settings, three training seeds, compact PET2.
.venv/bin/python omni-wflow/workflow.py all --config omni-wflow/configs/run-pilot.json --output omni-wflow/runs/pilot-periodic

# Larger follow-up: 20,000 parents, up to 20 epochs, 1,000 bootstrap replicates.
.venv/bin/python omni-wflow/workflow.py all --config omni-wflow/configs/run-full.json --output omni-wflow/runs/full
```

Open `runs/pilot-periodic/index.html` to select a training seed, or directly open
[seed 101's interactive browser](runs/pilot-periodic/seed-101/evaluation/index.html).
Every temperature/beta/n/k filter stays active: choose All values to scan, or a
single value to restrict that parameter, including a map axis. Extra unplotted
parameter values get separate panels. INFO opens the generated-physics and
interpretation modal (Close, Escape or backdrop dismiss it).
It shows any two parameter axes with a third parameter in separate panels; choose
source/control, classifier, direction, fixed remaining parameters and mixture f.
The inspection dropdown shows ROC curves, score distributions and baseline
comparisons. All maps use the same AUC color range. Stars denote parameter settings
withheld from training. CSV includes the selected physical parameters and both
injected-event and displayed mixture AUC. Results are embedded; no server/CDN is
needed. Static scientific PNG/SVG maps are exported under `evaluation/figures/`.

Commands `generate`, `train`, `evaluate`, `browser` can run separately. `--seed`
selects one configured training seed. Data and completed training runs are reused;
evaluation checkpoints each setting/control. Interrupted training restarts that
seed deterministically; optimizer-state continuation is not implemented. Config,
data and checkpoint fingerprints reject incompatible reuse. Choose a new output
directory when changing configuration. Do not use a previously generated dataset
with different acceptance or event counts under the same output path.

## Physics and data

Both background and source use the existing massive 3D thermal sampler. Geometry
selects 0–5% Au+Au 9.2 GeV before injection; background T=100 MeV. Each example is
a whole event of all six charged toy species with **|eta|<1**, full azimuth and no
extra pT cut. This matches canonical-background generation, but differs from the
cumulant study's rapidity/pT window. All emitted source daughters receive the same
acceptance; rejected daughters are never refilled.

The current source samples independent thermal daughters in a nominal rest frame
and applies a common boost. Their total three-momentum is zero only on average,
not event by event: this is not an exact clump-COM parent decay. Signal T is scanned
independently and equals the medium only at 100 MeV. A medium-temperature-locked,
exact-COM model is a distinct generator revision and is not retroactively applied
to the saved AUC results. This qualification is displayed in INFO.

Fresh particle four-vectors are generated: cumulant count tables are not used as
particle clouds. An HDF5 file holds ragged `p4`, `species`, `offsets`, `family_id`,
`split`, geometry and configuration attributes. Float32 physical vectors are used;
no top-pT truncation occurs. Source variants are reconstructed deterministically
from source seed namespaces while backgrounds are stored once.

The full grid is T=60,100,180 MeV; beta=0,.4,.8; n=2,4,8,16; k=1,2;
transverse/isotropic 3D directions (144 settings). Four combinations specified by
ID in the config are excluded from training/validation source sampling but tested.
A pooled classifier is trained on balanced background/signal pairs with a uniformly
sampled allowed source setting per training family. Each family gets one fixed
training setting per model seed. The test scan evaluates every setting against the
same held-out backgrounds. Test statistics do not grow by multiplying settings.

Parents and **all** their signal/control twins remain in one 70/15/15 split.
The pilot has 2,800 training, 600 validation and 600 test families. Signal labels
mean injection occurred before cuts (f=1 for primary training), including zero
accepted-source cases. Source membership, geometry, seeds and file ordering are
excluded from model inputs. Common whole-event rotations and shuffled particle
order augment training.

## PET2 and features

The compact model uses actual installed `omnilearned.network.PET2` components:
32-dimensional embeddings, two body blocks, one head block, four attention heads,
five neighbors and about 105k parameters. The classifier emits two logits.

A local adapter applies explicit length masks throughout the native body and head,
periodic phi distances and local differences, plus a sine/cosine angular embedding.
The raw phi slot is zeroed only for the global feature embedding, preserving its
native tensor width; local neighborhoods use physical phi. This makes the encoder
continuous across the ±pi boundary. All layers remain trainable except when an
encoder is explicitly frozen. The installed package is unchanged.

The main input is **ideal kinematics**: eta, phi, log(pT), log(E). True massive
energy contains mass/PID information: this is not a realistic tracker-only result.
Set training `representation` to `track` for massless energy computed from measured
momentum plus charge, or `pid` for a separately labelled ideal species-input run.
These latter inputs have five channels and require matching checkpoint widths.
No standardization is fitted; native physical log coordinates are retained.
Explicit masks allow real log(pT)=0 without treating the particle as padding.

`model_size: small` selects the official eight-block, dimension-128 PET2 architecture
for transfer experiments; `compact` is the CPU pilot. The CPU pilot is not a
measurement of the full jet-pretrained model's performance. Device selection uses
CUDA, available MPS, otherwise CPU. This session reports CPU. For an untested GPU
backend, first run the smoke configuration with that device explicitly selected;
backend errors are surfaced rather than silently changing execution semantics.

## Pretraining and fine-tuning

Toy-domain pretraining is implemented as **local denoising of log-pT/log-E**, not
as a claim to reproduce OmniLearned's joint generative/classification pretraining.
Only training families optimize the encoder; validation chooses the checkpoint.
Use separate run directories for each objective/initialization:

```bash
.venv/bin/python omni-wflow/workflow.py generate --config omni-wflow/configs/run-smoke.json --output omni-wflow/runs/domain-pretrain
.venv/bin/python omni-wflow/workflow.py pretrain --config omni-wflow/configs/run-smoke.json --output omni-wflow/runs/domain-pretrain
.venv/bin/python omni-wflow/workflow.py generate --config omni-wflow/configs/run-smoke.json --output omni-wflow/runs/domain-classifier
.venv/bin/python omni-wflow/workflow.py train --config omni-wflow/configs/run-smoke.json --output omni-wflow/runs/domain-classifier --init omni-wflow/runs/domain-pretrain/seed-101/best.pt
.venv/bin/python omni-wflow/workflow.py evaluate --config omni-wflow/configs/run-smoke.json --output omni-wflow/runs/domain-classifier
```

Supply a local official jet checkpoint via `--init /path/to/checkpoint.pt`, with a
matching official-size/four-feature config. The native encoder must match shapes
and keys; missing/incompatible backbone weights fail. Pretraining time embeddings
and task-specific heads are explicitly excluded in `transfer.json`. Our angular
adapter is restored from a workflow checkpoint when available, otherwise starts
fresh. `--freeze-encoder` freezes the backbone and angular adapter, leaving the
classifier head trainable. A new classifier head is always initialized for transfer.
No pretrained weights or multi-terabyte jet datasets are downloaded automatically.
Official jet transfer is supported but has not been measured without a supplied
compatible weight file. Local denoising and frozen-transfer paths were smoke-tested.

## Metrics and controls

Report per setting: ROC AUC with paired parent-family bootstrap 95% intervals,
average precision at 50% injected-event prevalence, and test signal efficiency at
10%/1% background acceptance. The pilot's FPR resolution is 1/600. These empirical
ROC operating points do not constitute independently validated deployment cuts.

Compare PET2 with gradient-boosted multiplicity, ideal proton/antiproton/net-proton
counts, and an ideal-PID summary baseline including pT sums and angular harmonics.
All baseline fitting uses training families only. Training-only smoothed pooled
multiplicity weights give a reweighted-AUC diagnostic; this is not an exact
per-setting count match. AUC conditional on at least one accepted source particle
is reported separately. These two diagnostics currently give point estimates.

Shared-boost signal is scanned at all 144 settings. Mixed, phi-scrambled and
mean-matched Poisson controls are evaluated at six predeclared settings. The
synthetic mixed control redraws independent daughter directions. The Poisson
control Poissonizes the emitted count and uses independent donors with the same
one-particle law, so its accepted mean matches the fixed-source ensemble.
Phi scrambling preserves accepted pT, eta and count, allowing a correlation check.
The browser honestly reports slices where a control was not evaluated.

For fixed scores, the mixture display uses the exact identity
`AUC_f = 0.5 + f*(AUC_injected - 0.5)`. A pure-background positive component dilutes
AUC; even perfect injected-event classification gives at most .505 for f=.01.
Changing class prevalence alone is a different operation and does not alter
population ROC AUC. The mixture identity is unit-tested by explicit weighted
mixture distributions. Mixture plots do not retrain the network.

## Independent generation seed and inference

```bash
.venv/bin/python omni-wflow/workflow.py generate --config omni-wflow/configs/run-independent.json --output omni-wflow/runs/independent-data
.venv/bin/python omni-wflow/workflow.py evaluate --config omni-wflow/configs/run-pilot.json --output omni-wflow/runs/pilot-periodic --external-data omni-wflow/runs/independent-data/particles.h5 --seed 101

.venv/bin/python omni-wflow/infer.py --checkpoint omni-wflow/runs/pilot-periodic/seed-101/best.pt --input omni-wflow/runs/independent-data/particles.h5 --output omni-wflow/results/scores.npz --rotations 2
```

External evaluation requires a different data seed and identical declared physics,
and writes `evaluation-external-<fingerprint>/` rather than replacing the primary
test results. External pool has 1,000 parents, of which 150 test families are used;
its intervals are therefore broader. `infer.py` scores arbitrary compatible ragged
clouds without labels; caller must apply the training acceptance and feature
conventions. The output is a signal-ranking score, not a calibrated posterior for
an arbitrary deployment prevalence.

## Verify and inspect artifacts

```bash
.venv/bin/python -m unittest discover -s omni-wflow/tests -v
node omni-wflow/tests/check_browser.mjs omni-wflow/runs/pilot-periodic/seed-101/evaluation/index.html
```

Checks cover family disjointness, acceptance/scrambling closure, explicit masks,
real zero features, small/empty clouds, padding/order invariance, periodic local
and global embeddings, gradient flow, reload consistency, tied-score bootstrap,
and mixture-AUC closure. Browser tests execute the embedded JavaScript with a
minimal DOM and exercise model/control/axis/fraction selections. Background-half
and permuted-test-label AUC null diagnostics are saved with results. Hyperparameters
and checkpoint choices use validation only. Test AUC maps are exploratory and
correlated; a new final test pool is needed after using them for model development.

Each seed directory stores `identity.json`, `history.json`, `best.pt`, baseline
models and `complete.json`. Evaluation stores parent-indexed scores/accepted source
counts, per-setting checkpoints, `results.json`, figures and `index.html`. The run
root has `seed-summary.json` (training-seed variation is distinct from test-sample
uncertainty). Generated HDF5 data, weights and results are Git-ignored. No existing
toy/explorer files are modified. This is a synthetic generator study; detector
response, calibration to BES data and exact global conservation remain absent.
