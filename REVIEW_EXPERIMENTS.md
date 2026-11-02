# Review-driven experiment and manuscript map, 2 November 2026

**SUPERSEDED SCHEDULING.** The broad five-seed expansion below is not active.
Current scope is original three-run main evidence, exact reuse before any
training, case500 FlowCAN Task1.3 gap verification and Case30-only mechanisms.
Read `CURRENT_TASKS.md` / `ACTIVE_SCOPE.json`. More resources do not justify
repeating completed work. Old case118/500 ablations and sensitivity sweeps are
not default tasks.

This is the execution/evidence map, not a claim that the experiments have finished.
Use `COMPANY_START_HERE.md` to start the implemented finite wave immediately.
Do not wait for optional baselines or repeat completed historical work by default.
The existing 386-coordinate ceiling and separate 72-coordinate tuning ceiling
remain unchanged. Available capacity does not justify scientific duplicates.

## What actually closes the review

| Review concern | Required evidence | Priority and experiment block | Author manuscript destination |
|---|---|---|---|
| Larger-grid accuracy | Complete native CANOS/FlowCAN coverage, retain all existing results, rescore canonical500 rPBL from original predictions where possible | P0 recovery and P1 main comparison | Table I and Figs.1-3 |
| Isolating the mechanism | Full, Deferred, Unit, coarse knockout and mean-token global correction with identical local core, controls, objective and endpoint | P1 core on30/118, P3 confirmation on500 | Table II and reserved mechanism panel |
| Feedback/weight interaction | Deferred+Unit completes the2x2 with Full, Deferred and Unit; report the interaction, not only favorable single contrasts | P2 interaction | Table II and mechanism panel |
| Optimization fairness | Preserve native recipes as one view; run the independent common-budget challenge and equal validation-only search budget as another | P2 fairness and bounded tuning | Reserved common-budget table |
| Solver-quality reporting | Same-output P95/P99/worst PBL, maximum specified-equation mismatch and pass fractions at1e-3/1e-4/1e-7 pu; reference-label floor and all failures | P0 evaluation before unnecessary training | Reserved tails/tolerance table and residual panel |
| Boundary/generalization | Keep feasible and near-boundary separate; include all methods and all unfavorable regimes, not only wins | P0 evaluation on every eligible checkpoint | Numerical-results text and retained complete evidence |
| Novelty | Harmonic physical-voltage extension versus a simpler global token, not just another neural baseline | P1 generic-global, P2 placement/depth/loss | Introduction, method and mechanism result |
| Reproducibility | Exact source, data/input roles, runtime, initialization/order, optimizer/loss/scheduler, checkpoint rule, harmonic implementation and failed runs | Every block | Problem/method and experiment protocol |
| Computational claims | No CPU/NR timing study or runtime-superiority claim. Report actual basis construction and reuse, without treating parameter count as a speed demonstration | Reporting boundary, no timing jobs | Method and limitations |

The supplied review is useful author feedback, not an independent review
performed by this repository. A weak-reject recommendation is not experimental
evidence. Neither resource abundance nor the desired award changes success criteria.

## Finite scheduling, without blocking useful work

1. Recover existing predictions/checkpoints and complete missing readouts first.
   Canonical rPBL, tails, pass fractions and boundary metrics normally need
   evaluation, not another200-epoch training.
2. Execute the current four-job case30 Mixed seed11 wave, then all declared
   seeds. Complete both regimes and case118 under the same frozen comparison.
   Existing assigned/live runs keep their original owners and protocols.
3. Implement and qualify the missing native CANOS/FlowCAN adapters, preserving
   independent learned PV-Q outputs. Complete missing500 baselines in parallel
   with model-control engineering where separate actual resources permit.
4. Implement coarse knockout and the mean-token global correction, then their
   fixed-budget controls. Their absence is a real mechanism gap, not something
   a placeholder or large headline reduction can replace.
5. Add the weighting/feedback factorial, equal-search fairness, depth6/24,
   fixed-count random/degree/electrical-distance anchors and the two physical
   supervision controls. Use the already declared first three seeds.
6. Confirm the four mechanisms on500. Independently qualify the2000-bus
   dataset and real PowerFlowNet/hierarchical-GNN sources for the final
   conditional blocks. Do not copy the48600 training count to2000 or touch
   the active continuous-training service.

Case30 is the primary ablation system, not the only evidence now worth obtaining.
Case118 checks whether the mechanism generalizes; case500 is a bounded scale
confirmation. Complete essential comparisons before breadth that cannot affect
the paper's central claims. Never choose which controls to finish based on TEST wins.

## Result return and direct paper ingestion

Retain raw predictions and full recovery checkpoints in the private work volume.
Return only the already validated `results/<protocol>/<coordinate>/metrics.json`.
After every declared seed of a cohort has returned:

```sh
python paper_export.py --repository . --protocol company_h200.json \
  --output "$WORK/paper_cohorts_completed.json"
```

The exporter validates each input using the existing receiver, checks common
population identities and numerical runtime, and exports **no partial-cohort
means**. Incomplete cohorts have `groups: null`, not zero or best-seed estimates.
Means/SD/ranges retain ordered seeds11/22/33/44/55. P99 is a mean of per-run
quantiles; the worst residual spans all cases and runs. These are different
estimands and must not be interchanged.

The author workspace can place this aggregate export at
`review/COMPANY_COHORTS.json` and rebuild its inline table slots. This is a new,
separately labeled fixed-H200 comparison, never automatic replacement/pooling of
historical0/1/2 runs, retrospective repairs or native-selected checkpoints.
Matching seed labels alone does not certify paired shared initialization.
Exact original initialization/order artifacts remain required for a causal
ablation claim. Missing adapters/protocols still block their own result slots.

Current readiness is100 implemented coordinate specifications and286 blocked
proposals. Tuning is specified but its configurable launcher is not implemented.
These statements concern source readiness, not a count of completed A100 runs.
