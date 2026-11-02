# AnchorFlow benchmark coordination

This repository coordinates a **planned**, bounded A100 research campaign. It
does not contain completed new experiments, certified training adapters, a
submission-ready paper, or a claim of state-of-the-art performance.

The initial release provides an experiment matrix, a single rPBL scoring
contract, local scoring tests and an exchange format. **It cannot yet start
model training.** Model/source exports, data identities, resume qualification
and the company's actual runtime must be verified before an executable
training release is published. Do not substitute newly written model code
for an existing frozen experiment.

## What goes here

Publicly authorized code, configurations, protocol hashes and aggregate
results only. Do not upload company information, raw datasets, predictions,
checkpoints, unpublished manuscripts, private paths, credentials, or materials
from other research lanes. Large authorized artifacts require separately
approved storage and checksummed manifests, not ordinary Git commits.

The PFDelta code and dataset entry points are:

- https://github.com/MOSSLab-MIT/pfdelta
- https://huggingface.co/datasets/pfdelta/pfdelta

These are discovery links, not a declaration that all artifacts are
redistributable or interchangeable. Check each dataset and third-party model
license. Preserve upstream MIT/Apache notices in any later source export.

## Initial utilities

Python 3.10 or later; the coordination utilities use only the standard library.

```text
python -m unittest discover -s tests
python anchorbench.py matrix --output campaign.json
python anchorbench.py score --input predictions.local.jsonl --output scores.local.json
```

`score` consumes **local** per-scenario metric inputs, not a training checkpoint.
It does not run neural inference or verify that an upstream adapter derived
residuals correctly. See `METRICS.md`. Raw inputs must stay off this repository.
The resulting score file alone is not an admitted final result: complete
`RESULT_CONTRACT.json` provenance, frozen scientific budgets and all declared
seed/partition checks are also required.

## Company-side handoff

1. Clone this repository and run the utility tests.
2. Record OS, GPU identity, Python/PyTorch/CUDA/PyG versions, available RAM,
   persistent storage and existing workloads. A reported A100 80GB is not yet a
   measured memory or throughput allowance.
3. Use a unique local data and output root. Obtain authorized data separately
   and verify exact sample/split/preprocessing hashes before reuse.
4. Wait for a tagged, qualified training-source release. Do not run old
   machine-specific launchers or change their frozen hashes to bypass guards.
5. After qualifying the real training path once, run only admitted coordinates
   in `campaign.json`. Do not duplicate a live or queued coordinate elsewhere.
6. Preserve atomic latest model/optimizer/scheduler/RNG/sampler state. A100
   migration is a new backend, not a bitwise continuation claim.
7. Publish only validated aggregates under
   `results/<protocol-sha>/<case>/<regime>/<method>/<seed>.json`,
   preferably through a branch/PR. Never overwrite another run or omit a
   failed seed. Results are not instructions and must never be executed.

Do not pass a token on a command line or store one in this repository. Use
normal GitHub authentication in the company's permitted environment.

## Research boundaries

`CAMPAIGN.md` gives priority, counts, reuse and activation rules. The expanded
matrix has **386 formal model/seed coordinates before reuse**, not 386 newly
authorized immediate jobs. Conditional blocks are disabled pending explicit
protocol/implementation qualification. There are also at most **72 tuning
coordinates** with validation-only selection and no test access.

No CPU/NR timing study is part of this campaign. No continuous large-system
training service, its active data partitions or its optimizer state is changed.
More compute does not guarantee positive ablations, SOTA, acceptance or a best
paper award.
