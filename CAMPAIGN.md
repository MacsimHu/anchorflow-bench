# Bounded A100 campaign, proposal 2026-11-02

## Decision and primary evidence

Determine whether harmonic voltage correction with immediate AC feedback
improves held-out AC power-flow accuracy beyond strong neural baselines and
generic global communication. SOTA is a goal, not an outcome assumed by the
protocol. Mechanism claims need matched controls, not just a larger mean gap.

Primary endpoints are completed-output MSE and PBL on the feasible All
partition, separately by system and training regime. Canonical rPBL, topology
subsets, tails, tolerance fractions and the separate near-boundary partition
are secondary, always reported even when unfavorable. Preserve original
unstable seeds and distinguish retrospective repairs.

Use the historical fixed partitions and test sample identities. Intact uses
48,600 N scenarios. Mixed uses16,200 each of N/N-1/N-2. Both test on6,000
feasible scenarios,2,000 per class; separate near-boundary tests have600
scenarios,200 per class where verified. Never add loading-intermediate
training samples while calling it the same Mixed task.

The public matrix is a **proposal**, not a frozen launch authority. Seeds
11/22/33/44/55 define the new proposed five-run campaign; reduced conditional
blocks use the first three, not the three best results. Historical seeds0/1/2
or another recipe remain separate unless the exact reuse contract matches.
Matching a seed number alone does not establish a paired initialization,
identical data order, training history or backend.

## Execution order and maximum formal coordinates

| Priority | Block | Cases / regimes | Maximum coordinates | Question |
|---|---|---|---:|---|
| P0 | Unified scoring and recovery | all existing results | 0 training by default | Can existing predictions/checkpoints fill the canonical metric and missing readouts? |
| P1 | Four-method native strong comparison |30/118/500, Intact/Mixed,5 seeds |120 | Is the predictive advantage reproducible with strong native baselines? |
| P1 | Core matched mechanisms |30/118, Intact/Mixed,5 seeds,4 extra arms |80 | Deferred feedback, unit weights, coarse knockout and simple global correction versus Full |
| P2 | Common-budget challenge |30/118/500, Intact/Mixed,3 seeds, CANOS/FlowCAN |36 | Do gains survive a separately frozen common training/selection budget? |
| P2 | Feedback x weighting interaction |30/118, Intact/Mixed,3 seeds |12 | Add deferred+unit to complete the2x2 factorial with reused matched arms |
| P2 | Depth sensitivity |30/118, Mixed,3 seeds,2 extra depths |12 | Is the effect only an iteration-depth choice? |
| P2 | Anchor placement |30/118, Mixed,3 seeds,3 extra choices |18 | Does placement matter beyond anchor count? |
| P2 | Intermediate physical supervision |30/118, Mixed,3 seeds,2 extra losses |12 | Distinguish feedback input from layered loss supervision |
| P3 | Large-system mechanisms |500, Intact/Mixed,3 seeds,4 extra arms |24 | Does the small-system mechanism transfer? |
| P3 | Independent scale extension |2000, Mixed,3 seeds,4 methods |12 | Does the method scale on an independently frozen eligible dataset? |
| P3 | Additional strong baselines |30/118/500, Intact/Mixed,5 seeds,2 methods |60 | Reproducible PowerFlowNet/hierarchical-GNN comparisons, subject to source/license/interface verification |

Total upper bound: **386 formal coordinates before reuse**. They are not
386 fresh mandatory trainings and are not all runnable in this initial
release. Do not restart completed or live historical work merely to satisfy
the new display matrix.

P1 Full uses the same matched recipe as its controls. An AnchorFlow-vs-GENCO
common-budget view may reuse P1 only if all actual budget/selection fields
match. Native CANOS/FlowCAN comparisons remain a separate practical comparison,
not an architecture-only causal experiment.

### Fixed choices required before activating conditional blocks

- **Simple global control:** same local core, controls, layer count, feedback
  placement and loss. A mean-pooled token predicts a voltage correction
  broadcast to all buses. Declare its actual parameter count; do not call it
  exactly capacity-matched unless verified. It answers a limited generic
  global-reach question, not equivalence to every global mechanism.
- **Depth:**6 and24 versus the12-layer reference. Preserve shared recurrent
  modules where applicable. Log all parameter differences.
- **Anchors:** random, degree-selected and electrical-distance-selected
  placements with fixed count and the reference included. Anchor placement
  does not alter physical PV/reference controls. Freeze selection seeds.
- **Loss:** final voltage supervision only, and final-layer rather than
  layered physics supervision. Keep feedback inputs unchanged. Freeze exact
  coefficients and normalization before running.
- **Additional baselines:** proposed PowerFlowNet and hierarchical GENCO.
  Verify public source/version/license and allowed inputs. Do not invent an
  implementation from a paper summary or present a failed port as a weak
  published baseline.
- **2000 buses:** official dataset has special sample-count limitations.
  Freeze its actual eligible partitions instead of copying the48,600 recipe.
  Use an independent research directory, never the active continuous service.

## Fairness, tuning and selection

Freeze model source, complete-output adapter, data/split/scaling hashes,
initialization, batch schedule, optimizer, loss, scheduler, stopping,
checkpoint rule, precision and allowed inputs. A100 migration requires a
new runtime identity. Do not silently use AMP/TF32 to increase throughput.

Allocate the same declared **search budget**, not a conveniently weak
configuration, to native baselines: at most3 validation-only configurations
per method/case/regime, one separate tuning seed101. That is at most72 tuning
coordinates for the four-method main matrix. Exact configurations must be
frozen from native recipe evidence before launch; unspecified candidates
are not runnable. Do not tune on test results or add seeds after a reversal.

Formal primary comparisons use a fixed declared endpoint. Also retain a
separately labeled validation-selected readout if frozen beforehand.
The historical fixed-endpoint H200 recipe must not be relabeled as
validation-selected simply because another baseline uses early stopping.

## Evaluation-only breadth

Reuse verified final checkpoints for:

- N/N-1/N-2 and feasible/near-boundary breakdowns.
- PBL p95/p99/worst, specified-equation infinity norm and pass fractions.
- Per-seed and paired-method differences; bootstrap seed blocks, not flattened
  scenario rows. Small seed counts limit inferential claims. Use multiplicity
  control if testing multiple mechanism contrasts.
- Error by loading bin, outage distance and connectivity characteristics,
  using input-defined bins frozen without looking at outcomes.
- Initial versus final and per-layer residual trajectories, control
  preservation, nonfinite outputs and all failures.
- Topology/operating-point OOD and reduced training-data experiments only after
  exact new split protocols are frozen. These are not free reinterpretations
  of the main tasks and have no automatic training authorization here.

Do not select near-boundary reporting based on favorable rankings. Neither
small mean PBL nor numerical nonconvergence proves physical feasibility or
infeasibility. No speed/NR timing study is added.

## Resource and stopping rules

Before launching, verify actual machine access, owner, existing jobs, available
GPU/RAM, persistent disk and duplicate coordinates across machines. Choose
concurrency from measured full training peaks, not the80GB label. Reuse an
unchanged path's qualification rather than running redundant canaries.

Start P0 and P1 before conditional breadth. Keep a minimum5GiB safety margin
plus the actual future input/checkpoint/output reservation. Preserve current
optimizer/RNG/sampler state and failure records. Requalify a changed backend
or path once, without pretending that an initial-parent smoke verifies resume
from the actual latest state.

Each admitted block has a finite job list, update/epoch budget and deadline.
On storage/resource loss, preserve a complete current checkpoint where possible,
stop only owned work, and record the precise trigger. Never change tolerances,
seed inclusion, success criteria or datasets after observing test results.

An80GB A100 is useful capacity, not a promise that this entire campaign fits
the submission window. Record real progress/cost before promising completion.
