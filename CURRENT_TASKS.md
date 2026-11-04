# Current assignment: Mixed Case30 controls only

CURRENT AUTHOR SCOPE -- 2026-11-04

This supersedes earlier task lists and automatic launch scripts. The paper now
requests exactly the existing main comparison, Case30 Mixed-only mechanism
controls, and a Case118 result figure. No new experiment matrix is authorized.

Company A100: retain the existing exclusive Mixed Unit, Mixed Knockout and
Mixed MeanToken assignments, seeds 0/1/2 for each. Check the original live,
completed and failed coordinates before any action. Continue original full
current states; do not duplicate them or restart completed endpoints.
Laboratory: original Mixed Deferred remains laboratory-owned. The two missing
Case500 Mixed CANOS seeds 1/2 are laboratory-owned and being placed in parallel.
All existing Full, GENCO and completed main-table Intact results are reused.

Cancel future Case30 Intact MeanToken seeds 0/1/2. If any is already active,
retain its exact full current model/optimizer/scheduler/RNG/order/validation
state and the true consumed cost before an owned controlled stop. Do not
silently discard its latest state or overwrite its failures. Return whether
each coordinate is not started, completed, or stopped with a retained state.
There is no remote operator acknowledgment in this publication itself.
Laboratory Intact Deferred/Unit/Knockout are also retired from this paper.
Intact training in the main comparison is NOT cancelled.

Do not invoke the old run_case30_extra.sh: it also launches Intact MeanToken.
Do not pull into a live training checkout or overwrite its editable package.
Use the existing original immutable qualified training checkout, including its
original qualification bytes. This documentation update is not a changed
numerical training release and does not replace that frozen checkout.

For already-running Mixed jobs, continue their existing supervisor/checkout.
Only for allocated missing or paused coordinates, the original commands are:

  cd "$ORIGINAL_CONTROL_CHECKOUT"
  python case30_extra_controls.py Knockout \
    --manifest "$MANIFEST_MIXED" --workspace "$WORK/runs/case30_extra_original" \
    --gpu "$GPU_UUID" --threads 4 --allocation-checked --resume-existing --execute
  python case30_extra_controls.py MeanToken \
    --manifest "$MANIFEST_MIXED" --workspace "$WORK/runs/case30_extra_original" \
    --gpu "$GPU_UUID" --threads 4 --allocation-checked --resume-existing --execute

Preserve the previous Mixed Full matching protocol: initial tensors and RNG,
ordered TRAIN membership/permutations, full-TRAIN scale, original DEV450,
AdamW, layered objective, schedule and fixed H200/152000-update endpoint.
No new seed, validation rule, hyperparameter search, clipping or early stopping.
The company CUDA backend difference remains disclosed relative to CPU Full.
These controls still make zero TEST calls. Return full current/final states,
original histories, coordinate identities, anonymous content hashes and costs
through the existing authorized artifact handoff. No weights, data, predictions,
private paths, credentials or manuscripts belong in the public repository.
Complete three-run cohorts and the original common evaluation are required
before the paper table can be filled; partial seed means are not substitutes.

Former Tables III/IV and their diagnostic collection are no longer requested
for this paper. Preserve any completed original evidence; do not delete results.
The withdrawn baseline must not be resumed. Do not alter the independent
continuous training service or another project. SOTA remains an objective,
not a conclusion certified by this handoff.
