"""Lightweight local setup check. Reports stay local; nothing installs or trains."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


EXPECTED = {"torch": "2.1.2", "numpy": "1.26.4", "torch-geometric": "2.6.1"}
GIB = 1024**3
ENVIRONMENT_PROBE = """
import importlib.metadata as m, json, sys
versions = {}
for name in ('torch', 'numpy', 'torch-geometric'):
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        versions[name] = None
print(json.dumps(dict(python=list(sys.version_info[:3]), packages=versions)))
"""


def check_work_root(path, *, install_bytes=8 * GIB):
    import shutil
    path = Path(path).resolve(strict=True)
    if not path.is_dir() or type(install_bytes) is not int or install_bytes < 0:
        raise ValueError("Choose an existing authorized writable directory and a nonnegative install estimate")
    required = install_bytes + 5 * GIB
    available = shutil.disk_usage(path).free
    if available < required:
        raise OSError(f"Need {required} bytes for the install estimate plus stop margin; only {available} available")
    with tempfile.TemporaryFile(dir=path) as stream:
        stream.write(b"anchorflow-write-check")
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=str(path), writable_verified=True, available_bytes=available,
                install_estimate_bytes=install_bytes, stop_margin_bytes=5 * GIB,
                training_and_dataset_capacity_checked=False)


def inspect_environment(executable):
    result = subprocess.run([str(executable), "-I", "-c", ENVIRONMENT_PROBE],
                            capture_output=True, text=True, timeout=15, check=True)
    record = json.loads(result.stdout)
    record["formal_versions_match"] = (
        record["python"][:2] == [3, 10] and all(
            isinstance(record["packages"][name], str)
            and record["packages"][name].split("+")[0] == version
            for name, version in EXPECTED.items()))
    record["torch_import_attempted"] = False
    record["CUDA_verified"] = False
    return record


def probe_cuda(executable, gpu, timeout):
    if not gpu.startswith("GPU-") or not 1 <= timeout <= 600:
        raise ValueError("Choose the allocated physical GPU UUID and a bounded timeout")
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=15, check=True)
    if gpu not in inventory.stdout.splitlines():
        raise ValueError("The selected GPU UUID is not in the live inventory")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True)
    if any(row and row[0].strip() == gpu for row in csv.reader(processes.stdout.splitlines())):
        raise RuntimeError("The selected GPU has a compute owner; check the assignment, do not evict it")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    result = subprocess.run(
        [str(executable), "-I", "-c",
         "import json,torch; "
         "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
         "x=torch.ones((32,32),device='cuda'); y=x@x; "
         "assert bool(torch.isfinite(y).all()) and float(y[0,0])==32.; "
         "print(json.dumps(dict(CUDA_verified=True,device=torch.cuda.get_device_name(0),"
         "training_peak_measured=False)))"],
        env=environment, capture_output=True, text=True, timeout=timeout, check=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--probe-cuda", action="store_true")
    parser.add_argument("--gpu")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    report = dict(
        workspace=check_work_root(args.work_root), environment=inspect_environment(args.python),
        no_install_or_training_performed=True,
        report_is_local_private_machine_information_not_public_result=True)
    if args.probe_cuda:
        if not args.gpu:
            parser.error("--probe-cuda requires --gpu GPU-<allocated UUID>")
        report["CUDA"] = probe_cuda(args.python, args.gpu, args.timeout)
    report["formal_environment_ready"] = (
        report["environment"]["formal_versions_match"]
        and report.get("CUDA", {}).get("CUDA_verified", False))
    print(json.dumps(report, indent=2))
    return 0 if report["formal_environment_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
