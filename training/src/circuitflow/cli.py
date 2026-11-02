import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess

from .io import read_json


def numerical_environment(device, gpu=None):
    if os.environ.get("GENCO_SCATTER_BACKEND", "pyg_native_sum") != "pyg_native_sum":
        raise ValueError("The handoff declares the native PyG sum backend, not an implicit alternative")
    os.environ["GENCO_SCATTER_BACKEND"] = "pyg_native_sum"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return dict(device="cpu", GPU_reserved=False)
    if not gpu or not gpu.startswith("GPU-"):
        raise ValueError("Select one allocated physical GPU explicitly with --gpu GPU-<uuid>")
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    rows = [[value.strip() for value in row] for row in csv.reader(inventory.stdout.splitlines())]
    selected, = [row for row in rows if row[0] == gpu]
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    busy = [row for row in csv.reader(processes.stdout.splitlines()) if row and row[0].strip() == gpu]
    if busy:
        raise RuntimeError("The selected GPU has another compute process; do not interrupt or duplicate it")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return dict(device="cuda", gpu_uuid=gpu, gpu_name=selected[1],
                total_MiB=int(selected[2]), currently_free_MiB=int(selected[3]),
                free_snapshot_is_not_an_allocation_reservation=True)


def job_from(protocol, identifier):
    plan = read_json(protocol)
    if plan["schema"] != "circuitflow-collaboration-v1":
        raise ValueError("Use a declared collaboration protocol")
    matches = [job for job in plan["jobs"] if job["id"] == identifier]
    if len(matches) != 1:
        raise ValueError("The scientific job ID must be unique and explicitly declared")
    job = matches[0]
    if (type(job["seed"]) is not int or job["seed"] not in (1, 2)
            or job["case"] != "case500" or job["task"] not in ("1.1", "1.3")):
        raise ValueError("This handoff must not duplicate existing seed0 or case118 work")
    return job


def main():
    parser = argparse.ArgumentParser(prog="circuitflow")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--weights", type=Path)
    jobs = commands.add_parser("list-jobs")
    jobs.add_argument("--protocol", type=Path, required=True)
    data = commands.add_parser("prepare-data")
    data.add_argument("--data-root", type=Path, required=True)
    data.add_argument("--case", choices=("case30", "case118", "case500"), required=True)
    data.add_argument("--task", choices=("1.1", "1.3"), required=True)
    data.add_argument("--revision")
    existing = commands.add_parser("bind-existing")
    existing.add_argument("--cache-dir", type=Path, required=True)
    existing.add_argument("--case", choices=("case30", "case118", "case500"), required=True)
    existing.add_argument("--task", choices=("1.1", "1.3"), required=True)
    existing.add_argument("--scale", type=Path)
    train = commands.add_parser("train")
    train.add_argument("--protocol", type=Path, required=True)
    train.add_argument("--job", required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--gpu")
    train.add_argument("--threads", type=int, default=4)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate.add_argument("--gpu")
    pack = commands.add_parser("pack-results")
    pack.add_argument("--run", type=Path, required=True)
    pack.add_argument("--readout", type=Path)
    pack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list-jobs":
        for job in read_json(args.protocol)["jobs"]:
            print(f"{job['priority']}\t{job['id']}")
        return 0
    if args.command == "prepare-data":
        numerical_environment("cpu")
        from .data import prepare
        print(prepare(args.data_root, args.case, args.task, revision=args.revision))
        return 0
    if args.command == "bind-existing":
        numerical_environment("cpu")
        from .data import bind_existing
        print(bind_existing(args.cache_dir, args.case, args.task, scale_file=args.scale))
        return 0
    if args.command == "doctor":
        numerical_environment("cpu")
        import torch
        from .api import ModelSpec, construct, forward, load_inference
        from .fixtures import four_bus
        from .genco.adapter import PFAdapter
        torch.set_num_threads(1)
        if args.weights:
            runtime, adapter, provenance = load_inference(args.weights)
        else:
            runtime = construct(ModelSpec(max_component_buses=128))
            runtime.model.eval()
            adapter, provenance = PFAdapter(power_scale_pu=2.5, generator_layout="aggregate_pf"), {}
        graph, _ = four_bus()
        with torch.no_grad():
            output, _ = forward(runtime, adapter, graph, "cpu", need_loss=False)
        print(json.dumps(dict(
            platform=platform.system(), python=platform.python_version(), torch=torch.__version__,
            parameters=sum(parameter.numel() for parameter in runtime.model.parameters()),
            output_shape=list(output["bus"].shape), CPU_only=not torch.cuda.is_initialized(),
            synthetic_API_example_not_benchmark_accuracy=True, weight_provenance=provenance), indent=2))
        return 0
    if args.command == "pack-results":
        from .evaluation import pack_results
        print(json.dumps(pack_results(args.run, args.readout, args.output), indent=2))
        return 0
    allocation = numerical_environment(args.device, args.gpu)
    if args.command == "train":
        if args.threads < 1:
            raise ValueError("Specify a positive CPU thread allocation")
        from .training import train_job
        job = job_from(args.protocol, args.job)
        print(json.dumps(dict(job=job["id"], allocation=allocation, automatic_successor=False)), flush=True)
        return train_job(job, args.manifest, args.output, resume=args.resume,
                         device=args.device, threads=args.threads)
    from .evaluation import evaluate
    evaluate(args.run, args.manifest, args.output, device=args.device)
    return 0
