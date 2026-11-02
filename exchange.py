"""Strict aggregate-only result admission; never publish weights or raw samples."""
import argparse
import json
import math
from pathlib import Path
import re

from anchorbench import METRIC_SCHEMA, write_new
from anchortrain import digest, read_protocol, selected_job


FIELDS = {
    "schema", "status", "protocol_sha256", "code_sha256", "data_manifest_sha256",
    "checkpoint_sha256", "test_sha256", "readout_sha256", "case", "regime", "method",
    "variant", "seed", "coordinate_sha256", "actual_updates", "checkpoint_selection_rule",
    "metric_schema", "runtime", "same_inference_for_all_metrics", "partitions_pooled",
    "groups", "scientific_claim"}
GROUP_FIELDS = {
    "partition", "topology", "cases", "MSE", "PBL", "rPBL_percent",
    "PBL_p95_pu", "PBL_p99_pu", "PBL_worst_pu", "equation_worst_pu",
    "equation_pass_fraction"}
RUNTIME_FIELDS = {"python", "torch", "numpy", "torch_geometric", "device", "threads"}
CLAIM = "one run, not complete campaign or SOTA"


def number(value, lower=0., upper=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < lower or (upper is not None and value > upper)):
        raise ValueError("A metric/rate must be finite and in its declared range")


def validate(value, protocol_path):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("Unexpected or missing fields; free text/raw artifacts are not public results")
    for key in ("protocol_sha256", "code_sha256", "data_manifest_sha256",
                "checkpoint_sha256", "test_sha256", "readout_sha256", "coordinate_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch("[0-9a-f]{64}", value[key]):
            raise ValueError("A bound result needs full SHA256 identities")
    job = selected_job(protocol_path, value["coordinate_sha256"])
    if (value["protocol_sha256"] != digest(protocol_path)
            or value["code_sha256"] != job["company_code_sha256"]
            or any(value[key] != job[key] for key in ("case", "regime", "method", "variant", "seed"))
            or type(value["seed"]) is not int
            or value["schema"] != "anchorflow-public-result-v1" or value["status"] != "complete"
            or value["actual_updates"] != 152000 or type(value["actual_updates"]) is not int
            or value["checkpoint_selection_rule"] != "fixed_H200"
            or value["metric_schema"] != METRIC_SCHEMA
            or value["same_inference_for_all_metrics"] is not True
            or value["partitions_pooled"] is not False or value["scientific_claim"] != CLAIM):
        raise ValueError("Result identity, endpoint or metric convention differs from the frozen protocol")
    runtime = value["runtime"]
    if (not isinstance(runtime, dict) or set(runtime) != RUNTIME_FIELDS
            or runtime["device"] not in ("cpu", "cuda")
            or type(runtime["threads"]) is not int or runtime["threads"] < 1):
        raise ValueError("A public runtime contains only versions, backend and thread count")
    for key in ("python", "torch", "numpy", "torch_geometric"):
        if not isinstance(runtime[key], str) or not re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,60}", runtime[key]):
            raise ValueError("Invalid public version; do not include paths, owners or host information")
    if not isinstance(value["groups"], list) or len(value["groups"]) != 8:
        raise ValueError("Both partitions and all topology categories must be reported")
    groups = {}
    for group in value["groups"]:
        if not isinstance(group, dict) or set(group) != GROUP_FIELDS:
            raise ValueError("Invalid aggregate metric fields")
        key = (group["partition"], group["topology"])
        if (key in groups or key[0] not in ("feasible", "near_infeasible")
                or key[1] not in ("All", "N", "N-1", "N-2")):
            raise ValueError("Duplicate or unknown partition/topology")
        expected = 2000 if key[0] == "feasible" else 200
        expected *= 3 if key[1] == "All" else 1
        if type(group["cases"]) is not int or group["cases"] != expected:
            raise ValueError("Unexpected population; seeds and partitions cannot be pooled")
        for field in ("MSE", "PBL", "rPBL_percent", "PBL_p95_pu", "PBL_p99_pu",
                      "PBL_worst_pu", "equation_worst_pu"):
            number(group[field])
        if not (group["PBL_p95_pu"] <= group["PBL_p99_pu"] <= group["PBL_worst_pu"]
                and group["PBL"] <= group["PBL_worst_pu"]):
            raise ValueError("Tail and mean summaries are inconsistent")
        passes = group["equation_pass_fraction"]
        if not isinstance(passes, dict) or set(passes) != {"0.001", "0.0001", "1e-07"}:
            raise ValueError("Report all prespecified equation tolerances")
        for fraction in passes.values():
            number(fraction, upper=1.)
            if not math.isclose(fraction * expected, round(fraction * expected), abs_tol=1e-7):
                raise ValueError("Pass fractions do not correspond to integer sample counts")
        if not passes["1e-07"] <= passes["0.0001"] <= passes["0.001"]:
            raise ValueError("Tolerance pass fractions must be monotone")
        groups[key] = group
    for partition in ("feasible", "near_infeasible"):
        for field in ("MSE", "PBL", "rPBL_percent"):
            mean = sum(groups[partition, topology][field] for topology in ("N", "N-1", "N-2")) / 3
            if not math.isclose(mean, groups[partition, "All"][field], rel_tol=1e-10, abs_tol=1e-14):
                raise ValueError("All is the sample-weighted summary within one partition only")
    return value


def destination(value):
    # All variable path segments have already been checked against the generated protocol.
    return Path("results") / value["protocol_sha256"] / value["coordinate_sha256"] / "metrics.json"


def ingest(source, protocol_path, repository):
    value = validate(json.loads(Path(source).read_text(encoding="utf-8")), protocol_path)
    target = Path(repository).resolve() / destination(value)
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != value:
            raise FileExistsError("Never overwrite another result for an admitted coordinate")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    write_new(target, value)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "ingest"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--protocol", type=Path, required=True)
        if name == "ingest":
            command.add_argument("--repository", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--protocol", type=Path, required=True)
    status.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "status":
        plan = read_protocol(args.protocol)
        prefix = args.repository / "results" / digest(args.protocol)
        completed = {}
        for source in sorted(prefix.glob("*/metrics.json")):
            value = validate(json.loads(source.read_text(encoding="utf-8")), args.protocol)
            if source.parent.name != value["coordinate_sha256"]:
                raise ValueError("A result is filed under a different coordinate")
            completed[value["coordinate_sha256"]] = digest(source)
        print(json.dumps(dict(completed=len(completed), declared=len(plan["jobs"]),
                              missing=[j["id"] for j in plan["jobs"] if j["id"] not in completed],
                              complete=len(completed) == len(plan["jobs"]),
                              scientific_SOTA_claim=False), indent=2))
    elif args.command == "validate":
        value = validate(json.loads(args.input.read_text(encoding="utf-8")), args.protocol)
        print(json.dumps(dict(valid=True, coordinate=value["coordinate_sha256"])))
    else:
        print(ingest(args.input, args.protocol, args.repository))


if __name__ == "__main__":
    main()
