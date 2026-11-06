# 2026-11-06 deadline coordination: Mixed ablations and missing CANOS seeds

This dated update supersedes stale execution-status statements, not numerical
recipes or a live checkout. Target: the author's November 10 deadline.

## Immediate allocation

1. **CANOS500 seed1 and seed2: prepare A100 takeover in parallel now.**
   The laboratory's original workers are only at approximately25 and30 of the
   maximum200 epochs, not close to completion. Their exact snapshot at the
   November6 morning inspection was saved18620/22800 and exact E/F ACK18430/22610.
   Both numerical processes were alive, without reported scientific failure.
   Local I/O processes had exited overnight and were restored without restarting
   either numerical process. Subsequent full transfers suffered SSH timeouts.
   A measured1MiB actual-packet prefix took43.28s on this shared connection;
   this is a link diagnostic, not a projected A100 training-speedup benchmark.
   Do not wait on the premise that A5000 is nearly done.
2. **Do not stop or duplicate either original seed until a real receiving A100
   is ready and the seed's latest full-state transfer is complete.** A5000 is
   still the owner. Issues1/2 require a readiness claim, not another speculative
   search for missing authority. Publication is not a release or a running job.
3. **Mixed Deferred:** company reports an already-started CUDA cohort at80/200;
   retain its full latest state and continue it, rather than restart. The
   laboratory's original CPU cohort is a separate preserved trajectory
   (approximately108870/143260/124070 saved updates at this inspection).
   Do not average a mixture of CPU/company seeds or combine partial runs into
   one row. On return, use a complete, prespecified three-seed cohort and
   disclose its backend; preserve the other cohort as additional evidence.
4. **Unit/Knockout/MeanToken:** the user reports all nine completed TEST runs.
   Do not retrain or rerun TEST just to regenerate already-saved aggregates.
   Return the existing artifacts listed below. Full is reused, not retrained.
5. **Intact Case30 is optional and lowest priority.** No new launch may delay
   the missing CANOS pair, complete Mixed Deferred cohort or their existing
   readouts/return. FlowCAN remains withdrawn.

## A100 operator: one readiness reply per issue

Post this aggregate-only object as an issue comment after actual local checks:

```json
{
  "seed": 1,
  "ready_for_original_state_takeover": true,
  "parallel_pair_capacity_checked": true,
  "python": "3.10.21",
  "torch": "2.1.2+cu118",
  "numpy": "1.26.4",
  "torch_geometric": "2.6.1",
  "original_case500_train_val_test_hashes_present": false,
  "private_receiving_channel_ready": false,
  "new_training_started": false
}
```

Replace versions/booleans with observed values; use seed2 for issue2.
Send the private receiving location through the established private channel,
not the public issue. The coordinator cannot invent an A100 SSH endpoint,
upload a private state to a public repo, or truthfully call a job runnable
without both its state and source/data closure.

The recipe/data hashes, full-state fields, selection/tie rules, scientific
deadline and readout conditions are already in
[CASE500_CANOS_A100_TASKS.md](CASE500_CANOS_A100_TASKS.md). Use that CANOS
contract, **not** the Case30 fixed-H200/AdamW path. The completed original
CANOS run is seed3, not seed0.

## Coordinator release/receipt required for each seed

No receiver should keep asking the author for optimizer or budget choices.
The existing contract fixes them. The coordinator must supply:

- Frozen runnable native source closure, original environment and dependency
  hashes, paired data/scale/initialization identities and the scientific plan.
- The **latest** full model, Adam, scheduler, Python/NumPy/CPU/CUDA RNG,
  permutation/cursor/within-epoch state, selected model and complete VAL
  history, with failures and cumulative costs. A best-only checkpoint or an
  older ACK while the worker has a newer saved state is insufficient.
- Exact packet/member hashes, latest step/epoch, E/F full copies and matching
  remote ACK. After a controlled saved boundary, check original owner exit and
  publish a seed-specific release tied to this state, not a generic promise.
- A receiving A100 import/current-state next-update and disk-replay
  qualification. A backend change is not promised to be bitwise equivalent.
- The exact resume command from the verified runnable bundle; **no guessed
  CLI based on an unrelated case30 runner**. Continue the original optimizer,
  scheduler, data order, early stop and maximum200 epochs. Retain all outcomes.

On successful claim+private receipt, transfer the two original seeds to
independent A100 run roots and use available capacity concurrently. Until then,
restore A5000 reception and keep both existing workers rather than halt useful
work. Do not copy private weights, raw predictions, data, hostnames or credentials
into GitHub.

## Return the nine already-completed ablation evaluations

The user relayed18 rounded means from
`case30-mixed-ablation-TEST-20261106.csv`. They have been inserted into the
author's working table as **operator-reported results**, with a local provenance
record. The primary CSV and per-seed readout bindings are not yet present on
this repository's main branch at the November6 check.

Publish only the non-sensitive aggregate CSV/JSON; keep saved predictions and
full checkpoints private. Return the existing per-seed rows (27 class rows:
three variants x three seeds x three classes), not just the nine mean rows:

```text
variant,seed,topology,cases,MSE,PBL_pu,scientific_scope_sha256,code_sha256,train_semantic_sha256,val_semantic_sha256,test_semantic_sha256,initial_sha256,checkpoint_sha256,checkpoint_selection,actual_updates,readout_sha256,backend
```

Variants are `Unit weights`, `Knockout`, `Mean token`; seeds0/1/2; topology
N/N-1/N-2; cases2000. MSE/PBL must be finite, nonnegative and from the same
saved prediction/readout. Include each model's actual fixed-H200/VAL selection
receipt and the completed readout/call manifest, not an assertion that the
coordinator already reproduced them. Report actual mismatches; do not relabel
a different objective, data population or selection rule.

An unweighted mean of the three per-seed class means must reproduce the
reported table precision. Do not pool topologies or silently replace Full.
Full is the existing original CPU cohort; these controls are CUDA cohorts.
Current descriptive result: Full is lower in all six columns than each of
these three controls; Knockout raises displayed MSE by11.8--22.0 times.
No statistical significance or completed Deferred comparison follows.

Return Deferred separately only after all three chosen cohort endpoints and
their original readout gates pass. Preserve the still-running other cohort's
identity and results; do not launch a new duplicate.
