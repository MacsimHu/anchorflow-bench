# Current tasks: reuse first, three runs, Case30 ablations only

**This supersedes the earlier five-seed/386-coordinate company plan.**
`ACTIVE_SCOPE.json` is the current scope. Historical source capabilities,
queue entries and examples are not authorization to rerun completed science.

## Immediate assignment: three Mixed Unit seeds, not six new runs

The coordinator has resolved the previously missing cross-machine ownership:
`CASE30_ALLOCATION.json` assigns Mixed Deferred seeds0/1/2 to the **already
running original A5000 queue**, and Mixed Unit seeds0/1/2 exclusively to
**company-A100**. The previously staged lab Mixed Unit queue must not be
dispatched. Do not confuse an Intact Unit worker with this Mixed allocation.
This replaces "ownership unknown" for these six coordinates; do not ask for
another per-seed authorization or skip all six again.

Use **`case30_matched.py`**, not `anchortrain.py`, for these Unit runs.
The original Full used fixed stratified **DEV450**, not full VAL5400, for its
learning-rate schedule and secondary checkpoint selection. The new runner
retains the original450 indices, all48600 TRAIN graphs, actual Full initial
tensors/RNG, Bpu, batch order and fixed H200/152000 updates. It changes only
electrical-to-unit coarse weighting in the architecture. CUDA execution is
explicitly a different backend from the retained CPU Full, not a bitwise
continuation or an isolated proof that every effect comes from the mechanism.

The complete Full configuration, seed identities, all450 indices, full-TRAIN
scale and ordered cache-content hashes are in
`case30_reference/FULL_REFERENCE.json`. The three accompanying PT files
contain **never-trained random initialization tensors and RNG only**; they
are not trained checkpoints, data or labels. Loading them avoids assuming
that a repeated seed produces exactly identical tensors on another build.
Obtain public PFDelta data locally or reuse the already prepared manifest;
do not download again, refit the scale, or prepare another case for this batch.

With the already working isolated reproduction environment and actual
`WORK`, `MANIFEST`, and allocated `GPU_UUID` from the company machine:

```sh
python case30_matched.py --manifest "$MANIFEST" --workspace "$WORK/runs/case30_unit_original" \
  --gpu "$GPU_UUID" --threads 4 --allocation-checked --execute
```

This executes exactly Unit seeds0,1,2 sequentially on the one allocated A100,
without waiting for a later authorization and without launching Deferred,
Full, GENCO or another experiment. Use an immutable new source checkout;
do not pull changes into a running scientific checkout. The default plan
command `python case30_matched.py` prints the three assigned jobs only.
For an identified interrupted **new Unit** batch, add `--resume-existing`;
only its current full model/optimizer/scheduler/RNG/order is resumed.
Completed matching endpoints are verified and reused, not retrained.

Start with this real batch, then report one actual command/PID/first positive
optimizer update and current-state pointer. Do not loop preflight or perform
another smoke per seed. Keep command/PID/private paths local; public status
can contain coordinate, step, elapsed time and completion, but not raw
predictions or trained weights. Retain the three full states, inference
exports and DEV histories on the authorized company volume for final intake.
This runner does not launch TEST or reinterpret the old five-seed exchange.
An actual exception must include its traceback and exact failed dependency;
"ownership unknown" is no longer an applicable blocker for these six jobs.

## What the company should do

1. Recover and inventory existing original results. Keep their true seed IDs,
   training recipe, data partition and checkpoint rule. Do not turn a missing
   locally imported file into a request to repeat training.
2. The only presumptive main-training gap is **FlowCAN, case500, Task1.3**.
   Check for completed/assigned original runs first. If already running,
   finish that owned original work rather than starting a company duplicate.
   If completed elsewhere, return its existing result packet.
3. Finish only necessary **Case30** matched controls: Deferred, Unit weights,
   coarse knockout and a simple mean-token global comparator. Reuse the
   existing Full parent/reference and the original three declared seeds.
   Check the already running controls before allocating anything new.
4. Rescore retained predictions for the one canonical rPBL, tails and equation
   pass fractions. These are evaluation tasks, not another full training matrix.

Existing CANOS500 Mixed or FlowCAN500 Intact rows may need their remaining
original seed results imported. Their currently visible partial coverage is
an **intake verification issue**, not permission to rerun those methods.
Do not silently pool a single run with incompatible runs just to display3.

## Do not launch

- No new AnchorFlow/GENCO/CANOS/FlowCAN main matrix.
- No five-seed main comparison, broad retuning or new Full baseline wave.
- No default Case118/500 mechanism replication.
- No default factorial, depth, anchor, supervision,2000-bus or extra-baseline sweep.
- No CPU/NR timing study or change to the active continuous-training service.

The old `company_h200.json` and `COMPANY_QUEUE.json` remain source/protocol
records, not the current launch queue. `run_company.py --execute` now refuses
the withdrawn five-seed waves before any model import or training.
Its read-only plan remains useful for identifying historical specifications.
Do not bypass this by copying old direct-train examples.

## Files and data

The source handoff is already present in this checkout. PFDelta public data
can be obtained at the company machine using the official preparation path
when an exact missing task actually requires it. Do not download all cases
or retransmit large old caches by default.

Historical five-seed500 reports remain intact for completeness. The main
three-run reporting set must follow the actual original declared identities,
never a best-three selection. Returning a verified existing prediction or
metric packet is preferable to a fresh run.

If an old company wave has already started, report its exact coordinate and
current state. Preserve its checkpoint; do not automatically kill it or launch
a replacement. This repository has no evidence that company training has or
has not already started.
