# Installed environment audit

Inspected 2026-11-04 in this repository's .venv. These are observations of the
installed release, not assumptions based on the latest upstream main branch.

- omnilearned 0.2.0; torch 2.15.0; sklearn and h5py are importable/discoverable.
- CUDA unavailable; MPS built into this torch, but MPS availability returned false
  in this execution environment. Reprobe when launching real training; do not
  promise Metal acceleration based on an arm64 machine alone.
- `omnilearned.network.PET2` supports classifier, generator and pretrain modes.
  Its classifier output is `forward(...)["y_pred"]`; `y` remains a required
  forward argument even in classifier mode. Ensure labels do not enter the
  classifier forward computation; the installed classifier branch uses x/optional
  conditions and does not pass y into its classifier head.
- Native kinematics are [delta_eta, delta_phi, log_pT, log_E], not the HICAD
  explorer's existing five encoded features. Do not pass those five features
  directly while enabling interactions that assume the native four.
- Optional PID input is distinct from the HDF5 dataset called `pid` that stores
  event class labels in the custom-data convention. Avoid ambiguous naming in
  the adapter and keep labels separate from particle species IDs.
- The official custom-data convention uses train/val/test directories and HDF5
  `data`, `pid` and optional `global`. We can export this format, while using a
  local ragged Dataset/collator to avoid unnecessary padding and duplication.
- PET2 body infers the mask from x[...,2] != 0. The adapter must preserve valid
  particles and padding through all transformations; generic standardization
  cannot be applied without accounting for this convention.
- Local neighborhoods use ordinary Euclidean distances on the supplied coordinate
  prefix, not automatically a periodic whole-event phi metric. Adapt/test this.
- The installed collator computes a truncation bound but then overwrites it with
  the full stored sequence length. Implement and test our own length handling.
- `train.run` calls DDP initialization and follows CPU/CUDA paths; the installed
  utilities do not provide a ready MPS training path. Use the package's model
  with a local training loop, explicit device selection and no mandatory DDP.
- Official “small” has eight transformer body blocks, dimension 128 and eight
  heads. First benchmark a compact PET2 configuration on complete event clouds;
  distinguish compact scratch results from checkpoint-compatible official models.
- Pretrain mode has both generation and classification heads. Loading jet weights
  is a transfer-learning experiment, not equivalent to training from scratch or
  performing toy-domain generative pretraining.

No package files were modified. No pretrained weights or datasets were downloaded.
