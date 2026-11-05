# First executed pilot

Completed on the local CPU with the compact OmniLearned PET2 adapter:

- 4,000 independent 0–5% central BES toy background parents.
- 2,800 training / 600 validation / 600 test families; no family overlap.
- Three training seeds (101, 202, 303); at most six epochs, validation-only early stopping.
- 144 signal parameter settings, 18 additional control/settings, four classifiers:
  648 evaluated records per model seed, 1,944 across the three seeds.
- Ideal-kinematics input, |eta|<1, no extra pT cut, full azimuth.
- Paired 400-replicate event-family bootstrap; two common rotations averaged at inference.

Open [the run selector](runs/pilot-periodic/index.html) or
[seed 101's interactive AUC browser](runs/pilot-periodic/seed-101/evaluation/index.html).
The browser selects parameter axes, panels, source/control, classifier and mixture
fraction. Training-seed variation is saved separately in seed-summary.json.

## Representative result

One isotropic clump, n=8, T=100 MeV, beta=.4, injection probability 1:

| Model / training seed | ROC AUC | Paired test 95% interval |
| --- | ---: | --- |
| PET2 / 101 | .907 | [.894, .920] |
| PET2 / 202 | .891 | [.878, .905] |
| PET2 / 303 | .852 | [.838, .867] |
| Ideal proton-count baseline / 101 | .916 | [.904, .931] |
| Multiplicity baseline / 101 | .683 | [.673, .695] |

The test intervals condition on each trained model; they do not include training
variation. Seed 303 stopped at epoch 1, while seeds 101/202 selected epoch 6.
This pilot does not establish convergence or a universal advantage over counts.
At several settings, much of the separation is explained by ideal proton counts.
At others, momentum information contributes: parameter-held-out setting 113
(T=180 MeV, beta=.4, n=2, k=1, isotropic) gives PET2 seed-101 AUC .711 versus
ideal proton counts .644. A matched statistical comparison and more training
statistics are needed before claiming a significant architecture advantage.

The strongest large-count additions approach AUC 1 for both PET2 and the count
baselines. That is not evidence that the network uniquely detects correlations.
Compare the scrambled controls and the multiplicity-reweighted diagnostic, and
repeat with track-like inputs before inferring detector-level sensitivity.

## Independent generated background check

The seed-101 model was evaluated on a separately generated pool (data seed 412194,
1,000 parents, 150 held-out test families), at all 144 settings and the same control
subset. At the representative n=8 setting, AUC is .903 [.879,.925], consistent with
the original test pool. External results are stored in
`runs/pilot-periodic/seed-101/evaluation-external-e07ca89ceb/`.

The original pool's background-half-versus-half AUC is .480 (approximate null SE
.024); permuted-test-label AUC is .488. External values are .511 and .506.
These are consistent with chance. Family overlap is false in all saved audits.

## Implementation checks and remaining experiments

Six automated scientific/integration tests pass, including a complete temporary
train/evaluate/restart workflow. Offline browser JavaScript checks pass for all
three model seeds. Actual inference exported 400 unlabelled background scores.
Local toy-denoising pretraining and frozen-encoder classifier transfer were also
smoke-tested; those tiny runs are API/implementation checks, not physics results.
Static PNG/SVG AUC maps were visually inspected. Live browser rendering was not
verified; browser access was unavailable and the native-app preview tool stalled.

The 20,000-parent configuration is implemented but has not been executed. Official
jet-pretrained transfer needs a compatible local checkpoint; no weights or jet
corpus were downloaded. Track-like and explicit-PID configurations, selected
specialists (by narrowing the allowed training settings), and larger final test
samples are follow-up experiments, not completed measurements in this report.
