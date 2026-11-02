"""Portable company H200 entry point; no launch on import, install or clone."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from anchorbench import campaign, write_new


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training" / "src"))
SCHEMA = "anchorflow-company-h200-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity():
    paths = [ROOT / name for name in
             ("anchortrain.py", "anchorbench.py", "anchor_physics.py", "exchange.py")]
    paths += sorted((ROOT / "training" / "src").rglob("*.py"))
    value = hashlib.sha256()
    for path in sorted(paths):
        value.update(path.relative_to(ROOT).as_posix().encode())
        value.update(bytes.fromhex(digest(path)))
    return value.hexdigest()


def protocol():
    jobs = []
    for coordinate in campaign()["jobs"]:
        full = (coordinate["block"] == "main_native"
                and coordinate["method"] in ("AnchorFlow", "GENCO"))
        control = (coordinate["block"] == "mechanism_core"
                   and coordinate["variant"] in ("Deferred", "Unit weights"))
        if not (full or control):
            continue
        job = dict(coordinate)
        job.update(
            id=coordinate["coordinate_sha256"],
            case=f"case{coordinate['case']}",
            task="1.1" if coordinate["regime"] == "Intact" else "1.3",
            maximum_active_hours=336,
            model_spec=dict(name="genco_h48") if coordinate["method"] == "GENCO" else dict(
                name="circuitflow",
                coupling="deferred" if coordinate["variant"] == "Deferred" else "coupled",
                coarse_weighting="unit" if coordinate["variant"] == "Unit weights" else "electrical",
                max_component_buses=512),
            status="requires_current_assignment_check", training_enabled=True)
        jobs.append(job)
    return dict(
        schema=SCHEMA, code_sha256=code_identity(), jobs=jobs,
        scientific_endpoint="fixed H200, 152000 updates; secondary VAL selection separate",
        actual_runtime_must_be_recorded=True, automatic_successor=False,
        existing_assignment_and_reuse_check_required=True,
        nonimplemented_proposal_coordinates_remain_disabled=True)


def read_protocol(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan != protocol():
        raise ValueError("Protocol or source differs from this release; do not hot-edit a frozen run")
    return plan


def selected_job(path, identifier):
    plan = read_protocol(path)
    jobs = [job for job in plan["jobs"] if job["id"] == identifier]
    if len(jobs) != 1:
        raise ValueError("Job is not an implemented coordinate in the frozen company protocol")
    return dict(jobs[0], company_protocol_sha256=digest(path),
                company_code_sha256=plan["code_sha256"])


def resource_gate(job, manifest_path, workspace, threads):
    import psutil
    import shutil
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["case"] != job["case"] or manifest["task"] != job["task"]:
        raise ValueError("Check allocation against the selected grid and training population")
    cache_bytes = sum(manifest["splits"][split]["bytes"] for split in ("train", "val"))
    # Full mapped input residency, three CPU batch copies and measured runtime overhead.
    batch_bytes = 64 * manifest["splits"]["train"]["bytes"] / 48600
    ram = cache_bytes + 3 * batch_bytes + 4 * 1024**3
    if psutil.virtual_memory().available < ram:
        raise MemoryError(f"Training RAM estimate {int(ram)} bytes exceeds actual available RAM")
    affinity = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), "cpu_affinity") else os.cpu_count()
    if threads > affinity:
        raise ValueError("The thread allocation exceeds this process's actual CPU affinity")
    parameters = 20069187 if job["method"] == "GENCO" else 1500000
    state_bytes = 3 * parameters * 4 + 16 * 1024**2
    output_bytes = (5400 + 6600) * int(job["case"][4:]) * 2 * 4
    required_disk = 5 * 1024**3 + 3 * state_bytes + 2 * output_bytes + 1024**3
    parent = workspace
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < required_disk:
        raise OSError("Insufficient input-independent output/checkpoint and five-GiB stop reservation")
    return dict(training_RAM_estimate_bytes=int(ram), output_reservation_bytes=required_disk,
                estimate_not_measured_GPU_peak=True, CPU_affinity_count=affinity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("protocol")
    generate.add_argument("--output", type=Path, required=True)
    listing = commands.add_parser("list-jobs")
    listing.add_argument("--protocol", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--protocol", type=Path, required=True)
    train.add_argument("--job", required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--workspace", type=Path, required=True)
    train.add_argument("--allocation-checked", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--gpu")
    train.add_argument("--threads", type=int, default=4)
    score = commands.add_parser("score-readout")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--readout", type=Path, required=True)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "protocol":
        write_new(args.output, protocol())
    elif args.command == "list-jobs":
        for job in read_protocol(args.protocol)["jobs"]:
            print(job["id"], job["case"], job["regime"], job["method"], job["variant"], job["seed"])
    elif args.command == "score-readout":
        read_protocol(args.protocol)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from anchor_physics import score_readout
        write_new(args.output, score_readout(
            args.run, args.readout, args.manifest, args.protocol))
    else:
        if not args.allocation_checked or args.threads < 1:
            raise ValueError("Check existing assignments, jobs, reservations, RAM/disk and duplicates first")
        job = selected_job(args.protocol, args.job)
        from circuitflow.cli import numerical_environment
        allocation = numerical_environment(args.device, args.gpu)
        from filelock import FileLock
        from circuitflow.training import train_job
        workspace = args.workspace.resolve()
        if workspace.is_relative_to(ROOT):
            raise ValueError("Keep writable scientific runs outside the public source checkout")
        resources = resource_gate(job, args.manifest, workspace, args.threads)
        (workspace / "leases").mkdir(parents=True, exist_ok=True)
        with FileLock(str(workspace / "leases" / (job["id"] + ".lock")), timeout=0):
            print(json.dumps(dict(job=job["id"], allocation=allocation,
                                  resources=resources, automatic_successor=False)), flush=True)
            return train_job(job, args.manifest, workspace / job["id"], resume=args.resume,
                             device=args.device, threads=args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
