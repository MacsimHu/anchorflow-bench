"""Export complete declared cohorts, never partial-seed means, for author tables."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from anchorbench import METRIC_SCHEMA, write_new
from anchortrain import digest, read_protocol
from exchange import validate


def export(repository, protocol_path):
    plan = read_protocol(protocol_path)
    protocol_sha = digest(protocol_path)
    declared = {}
    for job in plan["jobs"]:
        key = (job["case"], job["regime"], job["method"], job["variant"])
        declared.setdefault(key, []).append(job)
    actual, sources, populations = {}, {}, {}
    prefix = Path(repository) / "results" / protocol_sha
    for source in sorted(prefix.glob("*/metrics.json")):
        value = validate(json.loads(source.read_text(encoding="utf-8")), protocol_path)
        identifier = value["coordinate_sha256"]
        if source.parent.name != identifier or identifier in actual:
            raise ValueError("Duplicate or misfiled admitted coordinate")
        key = (value["case"], value["regime"])
        population = (value["data_manifest_sha256"], value["test_sha256"],
                      json.dumps(value["runtime"], sort_keys=True))
        if key in populations and populations[key] != population:
            raise ValueError("Methods in one comparison must use identical input populations and runtime")
        populations[key] = population
        actual[identifier] = value
        sources[identifier] = hashlib.sha256(source.read_bytes()).hexdigest()
    cohorts = []
    for key, jobs in sorted(declared.items()):
        jobs.sort(key=lambda job: job["seed"])
        found = [actual[job["id"]] for job in jobs if job["id"] in actual]
        runtime = {json.dumps(value["runtime"], sort_keys=True) for value in found}
        if len(runtime) > 1:
            raise ValueError("Do not pool different numerical runtimes within one cohort")
        complete = len(found) == len(jobs)
        cohort = dict(zip(("case", "regime", "method", "variant"), key))
        cohort.update(
            status="complete" if complete else "pending",
            declared_seeds=[job["seed"] for job in jobs],
            observed_seeds=[value["seed"] for value in found],
            missing_coordinates=[job["id"] for job in jobs if job["id"] not in actual],
            groups=None, architecture_only_causal_claim=False)
        if complete:
            groups = []
            for partition in ("feasible", "near_infeasible"):
                for topology in ("All", "N", "N-1", "N-2"):
                    selected = [next(group for group in value["groups"]
                                     if (group["partition"], group["topology"])
                                     == (partition, topology)) for value in found]
                    metrics = {}
                    for field in ("MSE", "PBL", "rPBL_percent", "PBL_p95_pu", "PBL_p99_pu"):
                        values = [group[field] for group in selected]
                        metrics[field] = dict(mean=statistics.mean(values),
                                              sample_SD=statistics.stdev(values),
                                              minimum=min(values), maximum=max(values),
                                              seed_values=values)
                    for tolerance in ("0.001", "0.0001", "1e-07"):
                        values = [group["equation_pass_fraction"][tolerance] for group in selected]
                        metrics["pass_" + tolerance] = dict(
                            mean=statistics.mean(values), sample_SD=statistics.stdev(values),
                            minimum=min(values), maximum=max(values), seed_values=values)
                    groups.append(dict(
                        partition=partition, topology=topology, cases_per_run=selected[0]["cases"],
                        metrics=metrics, PBL_global_worst_pu=max(g["PBL_worst_pu"] for g in selected),
                        equation_global_worst_pu=max(g["equation_worst_pu"] for g in selected)))
            cohort["groups"] = groups
        cohorts.append(cohort)
    return dict(
        schema="anchorflow-paper-cohorts-v1", protocol_sha256=protocol_sha,
        code_sha256=plan["code_sha256"], metric_schema=METRIC_SCHEMA,
        checkpoint_selection_rule="fixed_H200", actual_updates=152000,
        source_result_sha256=sources, cohorts=cohorts,
        quantile_rule="mean of per-run quantiles, not a pooled case quantile",
        worst_rule="maximum across every declared seed and case",
        uncertainty_rule="sample SD and observed range over declared training runs",
        partitions_pooled=False, partial_means_exported=False,
        paired_initialization_requires_separate_artifact_check=True,
        historical_results_merged=False, SOTA_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.repository, args.protocol)
    write_new(args.output, result)
    print(json.dumps(dict(output=str(args.output),
                          complete=sum(c["status"] == "complete" for c in result["cohorts"]),
                          pending=sum(c["status"] == "pending" for c in result["cohorts"]))))


if __name__ == "__main__":
    main()
