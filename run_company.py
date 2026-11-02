"""Run one explicitly selected, finite company wave through training and return."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

from anchorbench import write_new
from anchortrain import ROOT, digest, read_protocol, selected_job
from exchange import ingest, validate


def authorize_current_scope():
    scope = json.loads((ROOT / "ACTIVE_SCOPE.json").read_text(encoding="utf-8"))
    if (scope.get("schema") != "anchorflow-reuse-first-three-run-scope-v1"
            or scope.get("old_five_seed_driver_execute_authorized") is not True):
        raise RuntimeError(
            "The five-seed company waves are withdrawn. Reuse original three-run "
            "results; check only the actual case500 FlowCAN Task1.3 gap and existing "
            "Case30 controls. Read CURRENT_TASKS.md; do not bypass this with old train examples.")


def select_wave(queue_path, protocol_path, case, regime, seeds):
    queue = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    plan = read_protocol(protocol_path)
    declared = {job["id"]: job for job in plan["jobs"]}
    if (queue["execution_protocol_sha256"] != digest(protocol_path)
            or len(queue["ready_order"]) != len(declared)
            or {job["id"] for job in queue["ready_order"]} != set(declared)
            or any(declared[job["id"]] != job for job in queue["ready_order"])):
        raise ValueError("The priority queue must contain the exact frozen implemented jobs")
    if len(set(seeds)) != len(seeds) or not set(seeds).issubset({11, 22, 33, 44, 55}):
        raise ValueError("Select distinct declared seed IDs; do not rename a historical seed")
    jobs = [job for job in queue["ready_order"] if
            job["case"] == case and job["regime"] == regime and job["seed"] in seeds]
    if not jobs:
        raise ValueError("No implemented job matches this wave")
    return jobs


def complete_endpoint(run, job, manifest):
    value = json.loads((run / "COMPLETE.json").read_text(encoding="utf-8"))
    if (value["complete"] is not True or value["identity"]["job"] != job
            or value["identity"]["data_manifest_sha256"] != digest(manifest)
            or value["progress"]["step"] != 152000
            or value["inference_sha256"] != digest(run / "inference.pt")
            or value["fixed_H200_sha256"] != digest(run / "fixed_H200.pt")
            or value["fixed_H200_VAL_sha256"] != digest(run / "fixed_H200_VAL.npy")):
        raise ValueError("An existing endpoint differs; preserve it and do not silently reuse/retrain")
    return value


def command(*args):
    subprocess.run([sys.executable, "-B", *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT / "COMPANY_QUEUE.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "company_h200.json")
    parser.add_argument("--case", choices=("case30", "case118", "case500"), required=True)
    parser.add_argument("--regime", choices=("Intact", "Mixed"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allocation-checked", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    if args.execute:
        authorize_current_scope()
    jobs = select_wave(args.queue, args.protocol, args.case, args.regime, args.seeds)
    print(json.dumps(dict(jobs=len(jobs), historical_specifications_only=True,
                          current_scope="CURRENT_TASKS.md and ACTIVE_SCOPE.json",
                          training_requested=args.execute,
                          coordinate_ids=[job["id"] for job in jobs]), indent=2), flush=True)
    if not args.execute:
        return 0
    if not (args.allocation_checked and args.manifest and args.workspace and args.gpu):
        parser.error("--execute requires --allocation-checked, --manifest, --workspace and --gpu")
    workspace = args.workspace.resolve()
    if workspace.is_relative_to(ROOT):
        raise ValueError("Training/checkpoints/readouts must stay outside the source checkout")
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    if data["case"] != args.case or data["task"] != ("1.1" if args.regime == "Intact" else "1.3"):
        raise ValueError("The manifest belongs to a different cell")
    from filelock import FileLock
    workspace.mkdir(parents=True, exist_ok=True)
    with FileLock(str(workspace / "COMPANY_DRIVER.lock"), timeout=0):
        record_dir = workspace / ("driver_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        record_dir.mkdir()
        write_new(record_dir / "WAVE.json", dict(
            protocol_sha256=digest(args.protocol), manifest_sha256=digest(args.manifest),
            declared_jobs=[job["id"] for job in jobs], no_automatic_successor=True,
            actual_case=args.case, regime=args.regime, execution_seeds=args.seeds))
        for coordinate in jobs:
            job = selected_job(args.protocol, coordinate["id"])
            run = workspace / job["id"]
            if not (run / "COMPLETE.json").exists():
                if (run / "FAILED.json").exists() or (run / "SETUP_FAILED.json").exists():
                    raise RuntimeError("Preserve and review the recorded failed run; never replay it automatically")
                if run.exists() and not args.resume_existing:
                    raise RuntimeError("Existing partial run requires explicit --resume-existing from its full latest state")
                arguments = [
                    ROOT / "anchortrain.py", "train", "--protocol", args.protocol,
                    "--job", job["id"], "--manifest", args.manifest, "--workspace", workspace,
                    "--device", "cuda", "--gpu", args.gpu, "--threads", args.threads,
                    "--allocation-checked"]
                if run.exists():
                    arguments.append("--resume")
                command(*arguments)
            endpoint = complete_endpoint(run, job, args.manifest)
            readout = run / "readout"
            if not (readout / "COMPLETE.json").exists():
                if readout.exists():
                    raise RuntimeError("Preserve the partial/failed readout; do not overwrite or automatically repeat TEST")
                command("-m", "circuitflow", "evaluate", "--run", run, "--manifest", args.manifest,
                        "--output", readout, "--device", "cuda", "--gpu", args.gpu)
            evaluated = json.loads((readout / "COMPLETE.json").read_text(encoding="utf-8"))
            if (evaluated["complete"] is not True or evaluated["TEST_graphs"] != 6600
                    or evaluated["model_sha256"] != endpoint["inference_sha256"]
                    or evaluated["TEST_sha256"] != digest(readout / "TEST.npy")):
                raise ValueError("Existing readout differs from this exact fixed endpoint")
            metrics = run / "metrics.local.json"
            if not metrics.exists():
                command(ROOT / "anchortrain.py", "score-readout", "--run", run, "--readout", readout,
                        "--manifest", args.manifest, "--protocol", args.protocol, "--output", metrics)
            value = validate(json.loads(metrics.read_text(encoding="utf-8")), args.protocol)
            if (value["coordinate_sha256"] != job["id"]
                    or value["checkpoint_sha256"] != endpoint["inference_sha256"]
                    or value["data_manifest_sha256"] != digest(args.manifest)
                    or value["readout_sha256"] != evaluated["TEST_sha256"]):
                raise ValueError("Existing metric output does not bind this run and prediction")
            target = ingest(metrics, args.protocol, ROOT)
            write_new(record_dir / (job["id"] + ".json"),
                      dict(status="aggregate_return_prepared", source=metrics.name,
                           public_relative_path=target.relative_to(ROOT).as_posix()))
        command(ROOT / "exchange.py", "status", "--protocol", args.protocol, "--repository", ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
