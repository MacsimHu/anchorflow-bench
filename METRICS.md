# One rPBL contract

All newly admitted results use exactly one rPBL:
`pfdelta_recorded_orientation_throughput_percent_v1`.

For scenario m and bus i, let r_i = (rP_i, rQ_i) be the mismatch of the
specified equations in per-unit on a 100 MVA base. At PV buses rQ is zero;
both components are zero at the reference bus. Powers and mismatches come
from the same completed prediction, with input controls restored.

Using each branch's **recorded** from/to orientation and predicted terminal
powers, define

```text
T_i = max(sum_{e: from(e)=i} |S_from(e)|,
          sum_{e: to(e)=i}   |S_to(e)|)
PBL_m = mean_i sqrt(rP_i^2 + rQ_i^2)
rPBL_m (%) = (100/n) sum_{i: T_i > 1e-10 pu} ||r_i||_2 / T_i
MSE_m = MSE_PQ + MSE_PV + MSE_reference + MSE_branch
```

Terms with T_i <= 1e-10 contribute zero, retaining the existing benchmark
contract. Do not replace the two separate recorded-orientation sums with the
sum over all incident terminals. Report the exact definition, including this
threshold behavior, rather than describing it as an orientation-invariant
electrical feasibility certificate.

Final means average the **per-scenario** metrics, then summarize independent
training runs. `All` pools only its own feasible or near-boundary partition,
never both. A dataset's `near_infeasible` label is not proof of no solution.
rPBL values are percent; multiplying an already-percent result by100 again is
an error. Never divide the mean PBL by a mean denominator to emulate the
mean of per-bus/per-scenario ratios.

## Historical migration

Some supplied 500-bus numbers divide scenario PBL by mean nodal load magnitude.
Those are a different statistic. Preserve their original bytes and provenance,
but **do not publish them in the canonical rPBL column or rank them against
canonical rPBL**. There is no constant conversion factor.

Recover the corresponding prediction/controls/branch outputs and rescore, or
rerun inference from a verified checkpoint on exactly the same sample IDs.
If neither is available, mark canonical rPBL `pending_rescore`, not zero and
not the old number. Do not retrain solely to change the metric if verified
predictions or checkpoints already suffice.

## Local JSONL scorer inputs

Each local input line contains:

```text
sample_id: nonempty identifier, unique across the file
partition: feasible | near_infeasible
topology: N | N-1 | N-2
group_mse: {PQ: number, PV: number, reference: number, branch: number}
residual: [[rP_0, rQ_0], ..., [rP_(n-1), rQ_(n-1)]]
from_bus, to_bus: equally sized arrays of zero-based integer bus indices
abs_s_from_pu, abs_s_to_pu: matching nonnegative terminal-power magnitudes
```

All numbers must be finite. Residual signs may be positive or negative.
`group_mse` is computed by the common output-completion adapter, not by this
exchange utility. Keep these files local; they contain predictions.

The scorer emits per-partition/per-topology and All summaries, PBL p95/p99,
worst PBL, worst specified-equation component and passing fractions at
1e-3, 1e-4 and1e-7 pu. These are scenario-level diagnostics for one run.
Seed intervals and paired comparisons are computed later without pretending
that thousands of test cases are thousands of independent training runs.
