# AnchorFlow company A100 research handoff

This release includes **actual standalone training, complete-state resume,
qualified prediction readout, canonical MSE/PBL/rPBL scoring and aggregate
result ingestion**. It is not a new experimental result or a claim of SOTA.
Company A100 hardware, capacity and CUDA numerical behavior have not been
remotely verified.

| Scope | Executable in this release |
|---|---|
| AnchorFlow and pinned official GENCO | Cases 30, 118, 500; Intact and Mixed; seeds 11, 22, 33, 44, 55 |
| Deferred feedback and unit-weight controls | Cases 30, 118; both regimes; all five seeds |
| CANOS, FlowCAN, knockout, generic global and other proposals | **Not implemented here; disabled**, not weak surrogate baselines |

`company_h200.json` contains **100 executable coordinate specifications**, not
100 jobs to start automatically. `anchorbench.py matrix` retains the broader
386-coordinate proposal, including 286 currently non-executable coordinates
and at most 72 separately unfrozen tuning coordinates. Reuse an existing result
only after identity matching. Check live and queued assignments before running
any coordinate. Native historical seeds 0/1/2 cannot be renamed as the new seeds.

## Install in an isolated Python 3.10 environment

Use the company's permitted package sources and GitHub authentication. The
Linux CUDA 11.8 example below does not change an existing shared environment.

```sh
git clone https://github.com/MacsimHu/anchorflow-bench.git
cd anchorflow-bench
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r training/requirements-reproduce.txt "pytest>=7"
python -m pip install --no-deps -e ./training
export GENCO_SCATTER_BACKEND=pyg_native_sum
python -m pytest test_company.py training/tests tests -q
python -m circuitflow doctor
python anchortrain.py list-jobs --protocol company_h200.json
```

No `torch_scatter`, AMP, TF32 or compilation is required. The source-only
training package retains its Apache-2.0 and benchmark MIT notices under
`training/`. `EXPORT_MANIFEST.json` records exact unchanged source hashes and
the qualified standalone commit. New company orchestration and the scoring
bridge have their own tests. Tests do not certify another machine's GPU or
benchmark accuracy. Nothing trains on clone, installation or test discovery.

## Data, capacity and assignment check

The public dataset is https://huggingface.co/datasets/pfdelta/pfdelta.
Preparation resolves an immutable revision, verifies archive identities and
preserves the official parser, shuffle and sample order. Read its license.
Use **external** persistent data and run roots, not this public checkout.
For example, with locally chosen `$DATA` and `$WORKSPACE`:

```sh
python -m circuitflow prepare-data --data-root "$DATA" --case case30 --task 1.3
```

The command prints the frozen `MANIFEST.json` path. Repeat only for needed
case/task pairs. `bind-existing --cache-dir ... --case ... --task ...` can bind
trusted, locally owned combined caches without downloading or materializing
them again. It verifies all cache hashes and fits a full-TRAIN scale. PyG cache
files are trusted local pickle artifacts, not safe arbitrary downloaded `.pt`
files. Never accept a cache or weight attachment from a public result.

Before first execution, inspect actual owners, live/queued jobs, reservations,
CPU affinity, GPU/RAM capacity, output ownership and disk requirements. Use
`nvidia-smi`, the local scheduler and the company's allocation policy. A free
GPU snapshot is not a reservation. The wrapper checks full mapped-input RAM,
CPU batch staging, runtime overhead, atomic checkpoints and the five-GiB disk
stop margin, but **does not claim to know GPU activation peaks**. Start a newly
qualified backend with one owned job, retain its measured peak and allocate
further work from actual evidence. Never evict another workload.

## Run and continue one declared job

Select an exact coordinate ID from `list-jobs`. `$GPU_UUID` is an explicitly
allocated physical `GPU-...` UUID; `$MANIFEST` is its matching data manifest.

```sh
python anchortrain.py train --protocol company_h200.json --job "$JOB" \
  --manifest "$MANIFEST" --workspace "$WORKSPACE" --device cuda \
  --gpu "$GPU_UUID" --threads 4 --allocation-checked
```

The common recipe is fixed H200: 48,600 TRAIN, 5,400 VAL, B64, 152,000 updates;
AdamW, scheduler, deterministic epoch permutations and the original layered
physical/voltage loss are preserved. GENCO remains its actual pinned h48
model, not a local imitation. All new runs use this declared recipe; it does
not retroactively make historical differently selected results paired or
identical-budget evidence. The active-compute safety ceiling is 336 hours per
coordinate. Exhausting it produces an incomplete pause, never a shorter
endpoint presented as H200. Changing that ceiling requires a separately
identified protocol, not a hot edit.

Each coordinate owns `$WORKSPACE/<JOB>`, with a same-workspace coordinate
lease and a run lock. Cross-machine duplicate prevention still requires the
assignment registry. Preserve the entire run. A controlled pause returns 75.
Resume with the **same command plus `--resume`**; it restores current model,
Adam, scheduler, RNG, permutation, epoch cursor and partial VAL accumulators,
not best weights. A recorded failure is not automatically replayed. A completed
job cannot be overwritten. Do not edit/update a running source checkout.

## Predict and score once

After a complete endpoint, `$RUN` is `$WORKSPACE/<JOB>` and `$READOUT` is a new,
external directory. First the entire 5,400-sample final VAL is reproduced.
TEST is opened only if that succeeds.

```sh
python -m circuitflow evaluate --run "$RUN" --manifest "$MANIFEST" \
  --output "$READOUT" --device cuda --gpu "$GPU_UUID"
python anchortrain.py score-readout --run "$RUN" --readout "$READOUT" \
  --manifest "$MANIFEST" --protocol company_h200.json --output metrics.local.json
python exchange.py validate --input metrics.local.json --protocol company_h200.json
```

MSE, PBL and the single throughput-normalized **rPBL in percent** come from the
same saved prediction. The bridge checks endpoint/model/data/readout hashes,
all 6,600 ordered input identities and specified voltage controls. It computes
raw AC branch/shunt physics in float64, applies PQ/PV/reference equation masks,
and preserves original branch orientation. It retains all topology groups
separately for feasible (6,000) and near-boundary (600), plus P95/P99/worst PBL
and equation pass fractions at 1e-3, 1e-4 and 1e-7 p.u. Near-boundary labels do
not prove infeasibility.

Historical mean-load-normalized aggregates cannot be converted to this rPBL.
Recover original predictions/weights and rescore before considering retraining.
Do not silently pool a newly scored float64 result with a different historical
implementation; compare on the same recorded evaluator and input identity.

## Return only validated aggregates

Use a results branch or PR. This receiver accepts an exact field allowlist,
all eight partition/topology groups, declared endpoint/counts/versions and
finite coherent metrics. It refuses changed identities and stale overwrites.

```sh
git switch -c returns/company-h200
python exchange.py ingest --input metrics.local.json \
  --protocol company_h200.json --repository .
python exchange.py status --protocol company_h200.json --repository .
git add results
git -c user.name=MacsimHu -c user.email=257442440+MacsimHu@users.noreply.github.com \
  commit -m "Return verified company H200 aggregate metrics" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push -u origin returns/company-h200
```

Stage **only** the validated result paths after checking `git diff --cached`.
Never stage data, weights, raw predictions, private paths, company information,
credentials or unpublished manuscripts. GitHub authentication stays in the
approved credential manager, never a command or file. The result destination
is `results/<protocol-sha>/<coordinate-sha>/metrics.json`, so method variants
cannot collide. `status` lists every missing declared coordinate; it never
calls a favorable subset a complete campaign. It does not infer a failed seed
is successful: retain failure/partial-state records locally and report their
status without replacing seeds.

For a separately authorized private artifact transfer, `circuitflow
pack-results` can package the actual full-state/readout artifacts with hashes.
Those ZIP files are **not public GitHub uploads**. Public aggregate checks
cannot by themselves prove a remote run occurred; preserve originals for audit.

No CPU/NR timing study, architecture drawing or continuous-service recipe
change is part of this campaign. More hardware does not guarantee positive
ablations, SOTA, acceptance or a best-paper award.
