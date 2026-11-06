# 2026-11-06 deadline update: CANOS takeover priority and completed Mixed controls

Read [DEADLINE_COORDINATION_20261106.md](DEADLINE_COORDINATION_20261106.md) first. The original CANOS pair is not nearly finished; prepare parallel A100 full-state takeover now, but A5000 remains owner until actual private state receipt and seed-specific release. No duplicate or guessed runner. Preserve the company's already-running Deferred CUDA cohort separately from the laboratory CPU cohort. Return the nine existing Unit/Knockout/MeanToken TEST artifacts; do not retrain. Intact Case30 is optional only after these priorities; FlowCAN remains withdrawn.

---

# Case500 Mixed CANOS: A100 takeover of missing seeds 1 and 2

Updated: 2026-11-05. This adds two existing missing baseline coordinates.
It does not replace the company's nine original Case30 Mixed controls.

## Assignment and ownership

| Coordinate | Current owner | A100 action |
|---|---|---|
| Case500 / task1.3 / CANOS / seed1 | A5000, still active | Claim readiness, then take over only after explicit release |
| Case500 / task1.3 / CANOS / seed2 | A5000, still active | Claim readiness, then take over only after explicit release |
| Case500 / task1.3 / CANOS / seed3 | Already completed | Reuse; do not train again |

Use the separate seed issues for claims. Reply with the seed, operator,
available GPU slot, environment readiness, and requested handoff time. Preparing
the runtime and verifying existing input caches can proceed immediately.
**A claim or this document is not a release of the active A5000 worker.**
One owner per seed: do not start a fresh copy or race two hosts to a result.
Taking just one seed is useful; the other may remain on A5000.

The company retains its existing exclusive Mixed Unit, Knockout and MeanToken
coordinates (three seeds each). Do not duplicate those nine jobs, launch
retired Intact controls, resume the withdrawn baseline, or alter a live checkout.
Check actual capacity and existing owners before staging or allocating a GPU.

## Scientific identity: keep the original native CANOS recipe

- Dataset: PFDelta Case500, task1.3 Mixed training, not task1.1 intact-only.
  Exactly48,600 TRAIN and5,400 VAL graphs; preserve ordered membership,
  cached preprocessing, masks, original statistics and per-epoch permutation.
- Native CANOS-PF: hidden dimension128,20 interaction steps, sent messages
  enabled. Use the original native core, not AnchorFlow, GENCO or a new decoder.
  The published `native_reference/core/models/canos_pf.py` is a reference;
  its presence alone does not certify the full execution-source closure.
- Native primary objective: CANOS voltage/generation MSE plus0.1 times its
  original constraint loss. Native universal power balance is also recorded.
  This is not the AnchorFlow0.9S+0.1P objective.
- Batch64, including the final24-graph TRAIN batch:760 updates per epoch.
  Adam, learning rate0.0005, weight decay0. No new clipping, AMP, TF32,
  batch-size changes, loss modifications or hyperparameter search.
- Original SequentialLR: LinearLR start_factor0.01/end_factor1.0, milestone5,
  then StepLR step_size4/gamma0.9. Restore the actual scheduler state.
- Maximum200 epochs/152,000 updates per seed, preserving native early stopping:
  report_every2, decrease_by0.01, decrease_for20, needs_best_every30.
  This is a cap with the original stopping rule, not a new fixed-H200 ablation.
- Select by the leading native composite loss on all5,400 VAL graphs; retain
  the original `<=` tie rule. Never select by TEST, a preferred seed, or PBL
  alone. Preserve the existing best-model artifact and complete VAL history.
- Frozen scientific-scope SHA256:
  `ee671a60603b4ff8642bf2c6b8afc719ff7900506ff7321ebb5c725ae03e0a8b`.
  The CANOS recipe is retained; cancelled coordinates in the historical family
  are not revived by this handoff.
- Original finite science deadline: Unix1794679419.4563515. This assignment
  does not reset its clock. Report a missed deadline explicitly.

Runtime reference: Python3.10.21, Torch2.1.2+cu118, NumPy1.26.4, PyG2.6.1;
preserve the actual original native dependencies and deterministic settings.
Record the A100 hardware/runtime difference. Do not claim cross-device
bitwise equivalence from matching package versions.

## Required transfer package: private authorized channel only

The handoff coordinator supplies the **current** complete package after
ownership release, not the state observed when this document was written.
Do not upload packages, data, weights, predictions, manuscripts, credentials,
private network locators or private filesystem paths to this public repository.
Anonymous hashes and non-sensitive task-state receipts are sufficient here.

Required package closure:

1. Frozen runner and dependency manifest, original protocol/configuration,
   native model/loss/loader code and exact dataset identities.
2. Latest complete model parameters and buffers, named Adam state and parameter
   groups, scheduler, Python/NumPy/Torch CPU/CUDA RNG states.
3. Exact epoch/within-epoch cursor, current permutation, pending next action,
   training/validation counters and original logs, selected model and VAL history.
4. Packet member manifest, complete-checkpoint hash, packet hash, original
   failures and consumed scientific/engineering costs.
5. Explicit A5000 release receipt binding seed, complete state, source identity,
   last returned/saved update and any not-saved suffix. No unreported rollback.

Retained input SHA256 identities (locators are transferred privately):

| Input | SHA256 |
|---|---|
| TRAIN | `ab185fc84d05c478f170308dd509a528c4b08a916a35bbd619ffc2a33baebe84` |
| VAL | `6fb602eaa7c80f09fd657b6e6c0d6c587ff262704088b6c74650a5912a2da224` |
| Original common TEST cache | `bd92c76a4b9844037080be1ef395dbf725c4f754e83cc32d14861d22887f6510` |

The existing company Case30 manifest and launch scripts are **not** CANOS500
entry points. No new runnable CANOS500 release or transfer of these private
payloads is asserted by this documentation. Obtain the actual handoff package
and its explicit seed entry point before execution; do not invent a CLI or
reconstruct a fresh trajectory from weights alone.

## Execution and acceptance

1. A100 operator claims readiness for one or both seed issues; coordinator
   confirms which seed is being released. A5000 continues until that agreement.
2. At a complete checkpoint boundary, preserve the current full state and
   verify its original independent E/F copies and exact ACK. Coordinator
   verifies the owned A5000 numerical process has stopped and publishes release.
   Never terminate another workload or stop both seeds merely to prepare one.
3. Receive the bound package; verify every member/source/input hash. Admit
   actual A100 memory, scratch/output/checkpoint storage and protected workload
   growth, including the project's5GiB write margin. Use a unique run directory.
4. Qualify this genuinely changed execution path using the current full state:
   perform and replay the next update from disk within the A100 runtime, using
   the original next batch; check full model/Adam/scheduler/RNG/cursor state.
   Record qualification cost separately and do not advance the scientific
   parent by an unreported engineering update. No extra generic initial-state
   smoke or repeat of already verified unchanged paths is required.
5. Continue from that current state with no overlapping A5000 owner. Preserve
   complete atomic resume states and the original full-packet independent-copy
   ACK semantics. An alternative durable endpoint requires an explicit frozen
   amendment; do not silently weaken two-copy retention to make training faster.
6. Apply original selection/early-stop rules. Restore the selected model and
   reproduce the full native VAL score with original rtol1e-6/atol1e-8 before
   TEST. A failed gate leaves TEST unopened; preserve the failure.
7. Original common TEST cache has6,600 graphs. Evaluate only the original
   first6,000 feasible graphs: N/N-1/N-2,2,000 each. Do not substitute near-
   infeasible data, rebuild a split, or tune on the resulting TEST metrics.
   Use the original completed-output common MSE and PBL evaluator, not the
   native training loss as a substitute.
8. Return selected/final/full current states, per-scenario/per-seed results,
   source/data/checkpoint hashes, selection epoch, qualification, all failures,
   exact counts and costs through the authorized private artifact channel.
   The coordinator verifies results before combining seeds1/2 with completed3.
   No partial multi-seed means or best-seed substitution fills the paper table.

Keep the issue updated with readiness, released-state hash, qualified/resumed,
actual progress, terminal success or failure, and artifact-return confirmation.
An issue being posted/claimed is not evidence that an A100 run has started.
