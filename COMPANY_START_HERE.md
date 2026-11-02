# Company runner: start here

**Start the implemented jobs now. Do not wait for all optional research branches.**
Read `COMPANY_QUEUE.json` for exact job IDs. This file supersedes any instruction
to wait for a future training-source release. Nothing here authorizes changing
a live run or the separate continuous-training service.

## 1. Put all large writes on the data volume

Choose an existing **authorized writable directory** on the data volume as
`WORK`. A volume's free capacity does not grant permission to its root. Locate
the current account's work directory from the mounted-volume/user directory
layout. Do not recursively scan other users, chmod/chown the volume, or use
sudo to bypass permissions. If none exists, report that exact storage-permission
blocker rather than installing into a nearly full system volume.

Clone this small source repository using normal approved GitHub authentication.
Do not post the machine inventory or private paths to GitHub.
For a Linux shell:

```sh
export WORK="/actual/authorized/data-volume/work-directory"
mkdir -p "$WORK/tmp" "$WORK/cache/pip" "$WORK/cache/huggingface" \
  "$WORK/cache/torch" "$WORK/cache/xdg" "$WORK/data" "$WORK/runs" "$WORK/envs"
export TMPDIR="$WORK/tmp"
export PIP_CACHE_DIR="$WORK/cache/pip"
export HF_HOME="$WORK/cache/huggingface"
export TORCH_HOME="$WORK/cache/torch"
export XDG_CACHE_HOME="$WORK/cache/xdg"
```

Use the existing **Python 3.10 interpreter** to create a new isolated environment
on this volume. Do not upgrade or downgrade either existing environment.

```sh
/actual/path/to/python3.10 -m venv "$WORK/envs/anchorflow-h200"
. "$WORK/envs/anchorflow-h200/bin/activate"
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r training/requirements-reproduce.txt "pytest>=7"
python -m pip install --no-deps -e ./training
export GENCO_SCATTER_BACKEND=pyg_native_sum
```

The pinned environment is intentional. Torch 2.6/2.13 are not interchangeable
formal backends. There is no need to repair an unrelated slow-import environment
first. Do not switch AMP, TF32, model size, optimizer or data population for speed.

## 2. One bounded environment check, then useful work

Obtain the explicitly allocated physical GPU UUID with `nvidia-smi -L`.

```sh
export GPU_UUID="GPU-the-allocated-physical-uuid"
python preflight.py --work-root "$WORK" --python "$(command -v python)" \
  --probe-cuda --gpu "$GPU_UUID"
```

This checks a real local write, installation margin, package metadata and one
tiny CUDA calculation with a timeout. It does not initialize Torch during its
metadata-only check and does not install, train, benchmark or certify training
memory peaks. A versions-only report exits 2 and is not GPU qualification.
Keep its machine report local. Reuse relevant existing engineering checks;
do not rerun redundant CPU/GPU canaries for each unchanged seed.

Prepare only the first needed cell, **case30 Mixed**, using the frozen public
source revision consistently across later case/task cells:

```sh
python -m circuitflow prepare-data --data-root "$WORK/data" --case case30 --task 1.3
```

Use the printed `MANIFEST.json` path as `MANIFEST`. Retain its exact public
revision and archive hashes. Reuse that same revision for subsequent preparations
with `--revision`; do not let different cells silently follow different HEADs.
Never redownload or refit an already bound cache. Check for matching completed
results, live/queued jobs and assignments before starting.

## 3. First executable wave

The first four entries in `COMPANY_QUEUE.json` are one predeclared seed block:
AnchorFlow Full, Deferred, Unit weights and GENCO, **case30 Mixed seed11**.
This directly advances the mechanism evidence and a strong baseline. It is not
a tuning sweep; do not choose later seeds or arms according to test outcomes.

```sh
python run_company.py --case case30 --regime Mixed --seeds 11
python run_company.py --case case30 --regime Mixed --seeds 11 \
  --manifest "$MANIFEST" --workspace "$WORK/runs" --gpu "$GPU_UUID" \
  --threads 4 --allocation-checked --execute
```

The first command prints the four exact jobs without launching anything. The
second runs this finite wave sequentially through training, full-VAL-qualified
TEST, canonical scoring and validated aggregate preparation. It never pushes
raw artifacts or silently overwrites a partial/failed run. Use
`--resume-existing` only to explicitly continue an identified existing full
latest state. When this wave ends it does not invent a successor experiment.

To start just one selected job instead:

```sh
export JOB="copy-an-exact-ready-job-id-from-COMPANY_QUEUE.json"
python anchortrain.py train --protocol company_h200.json --job "$JOB" \
  --manifest "$MANIFEST" --workspace "$WORK/runs" --device cuda \
  --gpu "$GPU_UUID" --threads 4 --allocation-checked
```

Verify that this exact owned process advances actual updates and writes its
current-state checkpoint. Observe full training memory/capacity, then schedule
the independent ready work using available headroom. There is no need to wait
for an entire seed to finish before planning capacity, and no utilization target
justifies an artificial experiment. Avoid oversubscribing CPU affinity/RAM.

The current CLI selects a GPU with **no existing compute process at launch**.
It therefore supports one active trainer per physical GPU. Do not bypass this
guard or promise concurrent same-GPU trainers; run the queue sequentially on a
single A100 until a separately reviewed sharing implementation exists.
CPU data preparation and aggregate processing can overlap within actual resources.

Subsequent waves include every declared seed and both regimes. Prioritize
case30 mechanism completion, case118 confirmation, then the remaining main
case500 cells. The missing native FlowCAN500 Mixed result stays high priority
but is **blocked on its portable native adapter**, not replaced by AnchorFlow.
Actual MIT native baseline source/reference objectives are included under
`native_reference/`, with byte hashes and the exact native CANOS recipe.
These are source materials, not a secretly enabled training adapter. Its
learned PV reactive-power output must be retained for MSE; voltage-only
reconstruction is not an equivalent native readout.
Existing case500 predictions/checkpoints should fill canonical rPBL by rescoring
before any avoidable retraining.

For resume, readout, canonical scoring and aggregate-only return, use the exact
commands in the main README. Keep full current optimizer/RNG/sampler states,
all unfavorable results and failures. Do not silently restart from best weights,
replace a failed seed or report a partial endpoint as H200.

## 4. Complete breadth, finite boundaries

`CAMPAIGN.md` specifies all 386 proposed formal coordinates and the separate
maximum 72 validation-only tuning coordinates. `COMPANY_QUEUE.json` distinguishes
100 currently executable specifications from 286 blocked proposals. It lists
every blocked coordinate and its activation requirements, not fictitious jobs.
`TUNING_PLAN.json` fixes the three-rate grid and VAL-only selection. It remains
disabled until its configurable trainer is implemented and does not delay the
first native-recipe wave.

| Research question | Work and activation boundary |
|---|---|
| Main accuracy | Four methods, 30/118/500, Intact/Mixed, five seeds. Reuse exact identities; native CANOS/FlowCAN ports remain required. |
| Core mechanism | Full plus Deferred, Unit, knockout and mean-token global controls on30/118, both regimes, five seeds. Preserve local core/controls/loss and declare parameter differences. |
| Interaction | Deferred+Unit completes the2x2 feedback/weighting comparison, with matched Full/Deferred/Unit reuse. |
| Larger-grid mechanism | Repeat the four controls on500, first three declared seeds, after small-system evidence and implementation qualification. |
| Fair-budget sensitivity | CANOS/FlowCAN challenge with frozen actual optimizer/loss/selection and allowed inputs. Do not call unmatched objectives identical recipes. |
| Depth/anchors/physical loss | Depth6/24; random/degree/electrical-distance anchors; voltage-only/final-physics-only losses. Freeze exact definitions before activation. |
| Scale/additional baselines | Independent2000-bus population and real PowerFlowNet/hierarchical-GNN sources. No copied population assumptions or invented weak surrogates. |
| Evaluation breadth | Same-readout MSE/PBL/one rPBL, all topology groups, separate near-boundary set, P95/P99/worst residual and tolerance fractions. Keep adverse outcomes. |

The independent unit is a training run, not each test scenario. Primary
contrasts and seed inclusion are fixed before TEST. Report per-system/per-regime
results and seed uncertainty without promising statistical significance.
Case30 is the primary ablation system; case118 provides confirmation, and500
is a bounded extension, not a reason to postpone the first useful results.

No CPU/NR timing study, architecture drawing or continuous-service modification
is requested. SOTA and best-paper recognition are ambitions, never admission
criteria or guaranteed outcomes.
