#!/usr/bin/env bash
set -euo pipefail
: "${WORK:?Set the authorized persistent work directory}"
: "${GPU_UUID:?Set the explicitly allocated physical GPU UUID}"
: "${MANIFEST_MIXED:?Set the existing matching Mixed MANIFEST.json}"
: "${MANIFEST_INTACT:?Set the matching Intact Task1.1 MANIFEST.json}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export GENCO_SCATTER_BACKEND=pyg_native_sum
python case30_extra_controls.py Knockout \
  --manifest "$MANIFEST_MIXED" --workspace "$WORK/runs/case30_extra_original" \
  --gpu "$GPU_UUID" --threads 4 --allocation-checked --resume-existing --execute
python case30_extra_controls.py MeanToken \
  --manifest "$MANIFEST_MIXED" --workspace "$WORK/runs/case30_extra_original" \
  --gpu "$GPU_UUID" --threads 4 --allocation-checked --resume-existing --execute
python case30_extra_controls.py IntactMeanToken \
  --manifest "$MANIFEST_INTACT" --workspace "$WORK/runs/case30_extra_original" \
  --gpu "$GPU_UUID" --threads 4 --allocation-checked --resume-existing --execute
