# One checkout is the source handoff

Pull this repository once, including `training/` and `native_reference/`.
No model source, license or dependency file is intentionally held back for a
later transfer of the currently executable100-coordinate release.
The GitHub commit ZIP is an alternative to Git clone.

| Material | Included or obtained |
|---|---|
| AnchorFlow and official pinned GENCO implementation | `training/src/`, including vendored dependencies and original licenses |
| Exact installation dependencies | `training/requirements-reproduce.txt`, `training/pyproject.toml` |
| TRAIN/VAL preparation, public revision and hashes | `python -m circuitflow prepare-data`, documented in `COMPANY_START_HERE.md` |
| Fixed training recipe, finite jobs and full-state recovery | `company_h200.json`, `anchortrain.py`, `run_company.py`, `training/src/circuitflow/` |
| Same-output canonical metrics, tails and tolerances | `anchorbench.py`, `anchor_physics.py`, strict `exchange.py` receiver |
| Complete priority map and conditions | `COMPANY_QUEUE.json`, `CAMPAIGN.md`, `TUNING_PLAN.json`, `REVIEW_EXPERIMENTS.md` |
| Native CANOS/FlowCAN source and objectives | `native_reference/`, hashes and native-recipe provenance included; adapters still to be implemented |
| Aggregate result return and author-table export | `exchange.py`, `paper_export.py` |
| Focused checks | `tests/`, `test_company.py`, `training/tests/` |

## Large public data: download at the company machine

Do **not** transfer an old local cache merely because it already exists.
Use the supplied official preparation command for PFDelta's permitted public
data. The first preparation records the exact source revision and archive
hashes; pass that same revision for every later case/regime. Reuse an existing
verified company cache rather than downloading it again.

Prepare the first actual needed case/task before the rest of the matrix.
Preserve48,600 TRAIN,5,400 VAL and the separate6,000 feasible/600 near-boundary
TEST populations where the frozen release requires them. The2000-bus proposal
needs its own eligible-population protocol and is not admitted by copying
these counts. Do not mix active continuous-service data into this benchmark.

No pretrained weights are needed to start the declared fresh initializations.
Historical weights/predictions are useful only for exact reuse or rescoring,
not as a substitute for the declared initial state. They are not included in
this public handoff. Their licensing, original identities and restoration must
be verified before any separate transfer; do not upload private checkpoints,
raw samples, machine reports, credentials or manuscripts to this repository.

## Do not wait, but do not mislabel readiness

Start the four-job case30 Mixed seed11 wave in `COMPANY_START_HERE.md`.
All currently executable training source is present.
The other286 proposals are planning/source-readiness gaps, not missing secret
data files that a second bulk transfer will magically solve. They explicitly
need adapter/control implementation or independent dataset/source qualification.
The72-coordinate tuning grid is specified but its launcher remains disabled.

Return validated aggregate result files through the documented Git workflow.
Keep full latest optimizer/RNG/order/checkpoint state on the private data volume.
This checkout never automatically changes a running job, creates successor
experiments or pushes raw scientific artifacts.
