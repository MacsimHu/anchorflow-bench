"""Bounded campaign enumeration and one-contract local prediction scoring."""

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


METRIC_SCHEMA = "pfdelta_recorded_orientation_throughput_percent_v1"
SEEDS = (11, 22, 33, 44, 55)
METHODS = ("AnchorFlow", "GENCO", "CANOS", "FlowCAN")
REGIMES = ("Intact", "Mixed")
TOPOLOGIES = ("N", "N-1", "N-2")


def campaign():
    blocks = [
        ("main_native", "P1", (30, 118, 500), REGIMES, METHODS, ("Full",), SEEDS),
        ("mechanism_core", "P1", (30, 118), REGIMES, ("AnchorFlow",),
         ("Deferred", "Unit weights", "Coarse knockout", "Generic global"), SEEDS),
        ("common_budget", "P2", (30, 118, 500), REGIMES,
         ("CANOS", "FlowCAN"), ("Full",), SEEDS[:3]),
        ("feedback_weight_interaction", "P2", (30, 118), REGIMES,
         ("AnchorFlow",), ("Deferred + unit",), SEEDS[:3]),
        ("depth", "P2", (30, 118), ("Mixed",),
         ("AnchorFlow",), ("Depth6", "Depth24"), SEEDS[:3]),
        ("anchor_placement", "P2", (30, 118), ("Mixed",), ("AnchorFlow",),
         ("Random anchors", "Degree anchors", "Electrical distance anchors"), SEEDS[:3]),
        ("physical_supervision", "P2", (30, 118), ("Mixed",),
         ("AnchorFlow",), ("Voltage only", "Final physics only"), SEEDS[:3]),
        ("mechanism_500", "P3", (500,), REGIMES, ("AnchorFlow",),
         ("Deferred", "Unit weights", "Coarse knockout", "Generic global"), SEEDS[:3]),
        ("scale_2000", "P3", (2000,), ("Mixed",), METHODS, ("Full",), SEEDS[:3]),
        ("additional_baselines", "P3", (30, 118, 500), REGIMES,
         ("PowerFlowNet", "Hierarchical GENCO"), ("Full",), SEEDS),
    ]
    jobs = []
    counts = {}
    for block, priority, cases, regimes, methods, variants, seeds in blocks:
        start = len(jobs)
        for case, regime, method, variant, seed in itertools.product(
                cases, regimes, methods, variants, seeds):
            job = dict(block=block, priority=priority, case=case, regime=regime,
                       method=method, variant=variant, seed=seed,
                       status="planned_not_admitted", training_enabled=False)
            job["coordinate_sha256"] = hashlib.sha256(
                json.dumps(job, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            jobs.append(job)
        counts[block] = len(jobs) - start
    if len(jobs) != 386 or len({j["coordinate_sha256"] for j in jobs}) != len(jobs):
        raise ValueError("Campaign enumeration is inconsistent.")
    return dict(
        schema="anchorflow-a100-proposal-v1", frozen_launch_authority=False,
        canonical_metric_schema=METRIC_SCHEMA, formal_coordinates_before_reuse=len(jobs),
        tuning_coordinate_cap=72, tuning_seed=101, tuning_TEST_access=False,
        tuning_configurations_frozen=False, blocks=counts, jobs=jobs,
        scoring_first=True, reuse_requires_full_identity=True,
        continuous_training_service_changed=False, timing_study=False)


def finite_number(value, name, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, not a boolean.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return float(value)


def score_sample(record):
    if not isinstance(record, dict):
        raise ValueError("Each prediction record must be an object.")
    required = ("sample_id", "partition", "topology", "group_mse", "residual",
                "from_bus", "to_bus", "abs_s_from_pu", "abs_s_to_pu")
    if any(field not in record for field in required):
        raise ValueError("Prediction record lacks required metric inputs.")
    if not isinstance(record["sample_id"], str) or not record["sample_id"].strip():
        raise ValueError("A nonempty sample identity is required.")
    if record["partition"] not in ("feasible", "near_infeasible"):
        raise ValueError("Unknown test partition.")
    if record["topology"] not in TOPOLOGIES:
        raise ValueError("Unknown topology category.")
    groups = record["group_mse"]
    if not isinstance(groups, dict) or set(groups) != {"PQ", "PV", "reference", "branch"}:
        raise ValueError("MSE requires all four completed-output group means.")
    mse = math.fsum(finite_number(value, "group MSE", nonnegative=True)
                    for value in groups.values())
    residual = record["residual"]
    if not isinstance(residual, list) or not residual:
        raise ValueError("A nonempty nodal residual array is required.")
    components, magnitudes = [], []
    for row in residual:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("Each bus needs two masked residual components.")
        values = [finite_number(v, "residual component") for v in row]
        components.extend(abs(v) for v in values)
        magnitudes.append(math.hypot(*values))
    n = len(residual)
    fields = [record[k] for k in ("from_bus", "to_bus", "abs_s_from_pu", "abs_s_to_pu")]
    if any(not isinstance(value, list) for value in fields) or len({len(v) for v in fields}) != 1:
        raise ValueError("Branch index and terminal arrays must have matching lengths.")
    from_sums, to_sums = [0.] * n, [0.] * n
    for source, target, sf, st in zip(*fields):
        if any(type(i) is not int or not 0 <= i < n for i in (source, target)):
            raise ValueError("Branch endpoints must be valid integer bus indices.")
        from_sums[source] += finite_number(sf, "from terminal magnitude", nonnegative=True)
        to_sums[target] += finite_number(st, "to terminal magnitude", nonnegative=True)
    denominators = [max(a, b) for a, b in zip(from_sums, to_sums)]
    relative = math.fsum(r / t for r, t in zip(magnitudes, denominators) if t > 1e-10)
    result = dict(sample_id=record["sample_id"], partition=record["partition"],
                  topology=record["topology"], MSE=mse, PBL=math.fsum(magnitudes) / n,
                  rPBL_percent=100 * relative / n,
                  equation_max_abs_pu=max(components))
    if any(not math.isfinite(result[k]) for k in
           ("MSE", "PBL", "rPBL_percent", "equation_max_abs_pu")):
        raise ValueError("Metric arithmetic overflowed.")
    return result


def quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(samples):
    groups = defaultdict(list)
    identities = set()
    for record in samples:
        row = score_sample(record)
        if row["sample_id"] in identities:
            raise ValueError("Duplicate sample identity across the input file.")
        identities.add(row["sample_id"])
        for topology in (row["topology"], "All"):
            groups[(row["partition"], topology)].append(row)
    if not identities:
        raise ValueError("An empty file is not a successful evaluation.")
    output = []
    for (partition, topology), rows in sorted(groups.items()):
        pbl = [r["PBL"] for r in rows]
        equation = [r["equation_max_abs_pu"] for r in rows]
        output.append(dict(
            partition=partition, topology=topology, cases=len(rows),
            **{metric: statistics.mean(r[metric] for r in rows)
               for metric in ("MSE", "PBL", "rPBL_percent")},
            PBL_p95_pu=quantile(pbl, .95), PBL_p99_pu=quantile(pbl, .99),
            PBL_worst_pu=max(pbl), equation_worst_pu=max(equation),
            equation_pass_fraction={f"{tolerance:g}": sum(v <= tolerance for v in equation) / len(rows)
                                    for tolerance in (1e-3, 1e-4, 1e-7)}))
    return dict(
        schema="anchorflow-local-score-v1", metric_schema=METRIC_SCHEMA,
        status="scored_not_admitted", upstream_physics_adapter_verified=False,
        PBL_units="pu_on_100MVA", rPBL_units="percent",
        sample_count=len(identities), partitions_pooled=False, groups=output)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank prediction record on line{line_no}.")
            yield json.loads(line)


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "matrix":
        result = campaign()
    else:
        result = summarize(read_jsonl(args.input))
        digest = hashlib.sha256()
        with args.input.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result["prediction_input_sha256"] = digest.hexdigest()
        result["scorer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_new(args.output, result)


if __name__ == "__main__":
    main()
