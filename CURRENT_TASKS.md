# Current assignment: finish original Case30 controls, never repeat the main matrix

This is the current cross-machine allocation, not an experiment proposal.
Do not ask again for per-seed approval or classify these coordinates as
unassigned. Check the company's own completed/live/queued records first.
An existing company run must finish or resume its own original checkout;
it must not be duplicated in a second workspace.

## Exact owners and remaining work

| Coordinates (seeds 0,1,2 throughout) | Exclusive owner | Current action |
|---|---|---|
| Mixed Unit | Company A100, original `CASE30_ALLOCATION.json` | Continue existing original Unit batch; if never started, use its assigned matched runner. |
| Mixed Knockout | Company A100, new `CASE30_EXTRA_ALLOCATION.json` | Execute this new finite batch. |
| Mixed MeanToken | Company A100, new allocation | Execute after Mixed Knockout, or on another explicitly allocated GPU. |
| Intact MeanToken | Company A100, new allocation | Execute with the separate Intact reference and Task1.1 manifest. |
| Mixed Deferred | Laboratory A5000 original queue | Already assigned; company must not start duplicates. |
| Intact Deferred | Laboratory A5000 original queue | Continue the original current state and final readout. |
| Intact Unit and coarse Knockout | Laboratory A5000 original six-job queue | Reuse existing jobs. Frozen `local_only` computes the coarse branch and multiplies its increment by zero; it is the Knockout, not a missing extra method. |
| Case500 missing native baselines | Laboratory original eight-job queue | FlowCAN Intact seeds1/2/3, CANOS Mixed seeds1/2, FlowCAN Mixed seeds1/2/3. No company replacement training. |
| All Full, GENCO and completed main baselines | Existing original owners | Reuse their original results; no new training. |

The local RTX3050 Mixed recovery uses **full VAL5400**. It is not the
original Mixed CPU Full/DEV450 cohort and cannot replace or be pooled with
these Mixed controls. Main-table reporting uses the existing three declared
runs; the original seed labels remain unchanged. The new company work is
exactly nine controls, in addition to the previously assigned three Units.

## Preserve the exact comparison

Mixed reference: `case30_reference/FULL_REFERENCE.json`. Intact reference:
`case30_intact_reference/FULL_REFERENCE.json`. They include exact original
Full random tensors/RNG, ordered cache-content identities, full-TRAIN Bpu
and evaluation membership. The PT attachments contain **never-trained
initializations**, not learned weights, data or labels.

Both use all48600 TRAIN graphs, B64 with a24-graph tail, the original
epoch permutations, AdamW5e-4/weight-decay.01, original layered0.9S+0.1P,
the original plateau schedule, fixed H200/152000 updates and no early stop.
Mixed uses its original stratified DEV450 for all201 evaluations;
Intact uses its original full VAL5400. No new scale fitting or checkpoint
selection is allowed. CUDA RNG from the original Intact initializations is
restored along with CPU/Python/NumPy RNG.

MeanToken replaces harmonic interpolation with one arithmetic-mean token
per connected component and constant broadcast, retaining the local trunk,
coarse neural modules,12 layers and25 physical readouts. It does not solve
a harmonic system. Knockout retains the harmonic setup and coarse neural
branch but zeros its voltage increment. Its unused coarse gradients are
zero by design; do not claim matched effective parameter participation or
FLOPs. All modes preserve known controls and the original physical operator.

Mixed's original Full ran on CPU; the company CUDA controls are not a
bitwise-backend or fully isolated timing comparison. Intact's Full ran on
CUDA, but a different GPU/build still does not establish bitwise equality
or an isolated wall-clock benchmark. Record the actual backend.

## Commands: use the working environment, do not repeat setup

Never pull into a live Unit checkout. Keep that checkout and its editable
installation untouched. Create a separate immutable checkout of this release
for the nine new controls. Its runner prepends its own `training/src` and
therefore does not require replacing the old environment's editable package.
Use the already working Python3.10/Torch2.1.2/NumPy1.26.4/PyG2.6.1
environment; install nothing when it already works.

Keep all data/current states outside source on an authorized persistent data
volume. Set actual `WORK`, `GPU_UUID`, `MANIFEST_MIXED`, `MANIFEST_INTACT`.
Use an explicitly allocated physical GPU UUID and check competing work,
RAM, disk and duplicate coordinates once at launch. One trainer per GPU;
do not bypass this guard or interrupt TCPA/another workload.

Reuse the existing Mixed manifest. Only if Intact Task1.1 has not been
prepared, reuse the same public immutable data revision and run:

```sh
python -m circuitflow prepare-data --data-root "$WORK/data" --case case30 --task 1.1 --revision "$REVISION"
```

Set `MANIFEST_INTACT` to the printed manifest. This is data preparation,
not new Full training. The matched runner uses the supplied original scale,
not a newly fitted scale. It refuses incompatible cache-content identities.

Finish/continue the previously assigned Unit batch from its own checkout.
For never-started Units only, the original command remains:

```sh
python case30_matched.py --manifest "$MANIFEST_MIXED" --workspace "$WORK/runs/case30_unit_original" --gpu "$GPU_UUID" --threads 4 --allocation-checked --execute
```

From the **new** checkout, execute the nine extra controls with:

```sh
bash run_case30_extra.sh
```

That script checks the four required environment variables and runs exactly
Mixed Knockout0/1/2, Mixed MeanToken0/1/2 and Intact MeanToken0/1/2,
sequentially on the allocated GPU. It stops on an actual error or controlled
pause; it never invents a successor. `--resume-existing` restores only the
matching current full model/Adam/scheduler/RNG/order/within-VAL state.
Completed matching endpoints are verified and reused. A recorded FAILED
state requires diagnosis, not automatic retry or a fresh seed.

Read-only plans:

```sh
python case30_extra_controls.py Knockout
python case30_extra_controls.py MeanToken
python case30_extra_controls.py IntactMeanToken
```

`CASE30_EXTRA_QUALIFICATION.json` binds the changed sources and reference
bytes. Seventeen focused CPU tests passed, including exact ownership, original Full tensor
loading, real two-update Adam/RNG replay, inference identity, zero/nonzero
coarse gradients and both450/5400 validation tails. This is engineering
qualification, not actual company GPU training or accuracy evidence.
Do not run a redundant canary per unchanged seed.

## Return status and evidence

After launch, verify one actual positive optimizer update and the owned
PID/birth/current-state pointer. Keep commands, private paths and machine
details local. Public status may contain exact coordinate, step, elapsed
time, completion and anonymous artifact hashes. Publication of this source
does not prove a company worker started.

Retain the complete run directories, both rolling current slots, final
`fixed_H200.pt`, `inference.pt`, full DEV/VAL histories and all failures.
These controls intentionally make **zero TEST calls**. They are not inputs
to the historical five-seed exchange/export path. Return sanitized
per-coordinate progress/completion and arrange the existing authorized
private artifact handoff with the coordinator. Do not upload states,
predictions, data, private paths or manuscripts to public GitHub.
The coordinator will qualify the original endpoints and perform only the
missing common evaluation, then fill the Case30 paper table.

Only MSE/PBL, their N/N-1/N-2 stratification, retained-prediction tails and
tolerance rates are requested. Do not add All/rPBL paper outputs, NR/timing,
118/500 ablations, extra seeds, a new main matrix or continuous-service
recipe changes. Old `COMPANY_QUEUE.json`, `company_h200.json`,
`REVIEW_EXPERIMENTS.md` and five-seed export examples are historical
capability records, not launch instructions.
