"""One fixed-endpoint validation qualification followed by the complete common TEST."""
import json
import hashlib
import os
from pathlib import Path
import time
import zipfile

from filelock import FileLock
import numpy as np
import torch
from torch_geometric.data import Batch

from .api import forward, load_inference
from .data import input_identity, load_pool
from .io import read_json, sha256, write_json
from .training import numerical_runtime, source_identity, synchronized


def predict_split(runtime, adapter, pool, device, output, name, buses):
    path = output/(name+".npy")
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(pool), buses, 2))
    values[:] = np.nan
    values.flush()
    with (output/(name+"_INPUTS.jsonl")).open("x", encoding="utf-8") as identities:
        for first in range(0, len(pool), 64):
            indices = list(range(first, min(first+64, len(pool))))
            graphs = [pool.get(index) for index in indices]
            for index, graph in zip(indices, graphs):
                identities.write(json.dumps(dict(index=index, input_sha256=input_identity(graph)))+"\n")
            identities.flush()
            with (output/"CALL_LEDGER.jsonl").open("a", encoding="utf-8") as calls:
                calls.write(json.dumps(dict(split=name, first=first, count=len(indices),
                                            status="attempt_recorded_before_forward"))+"\n")
            raw = Batch.from_data_list(graphs)
            with torch.no_grad():
                predictions, _ = forward(runtime, adapter, raw, device, need_loss=False)
            values[first:first+len(indices)] = predictions["bus"].detach().cpu().numpy().reshape(
                len(indices), buses, 2)
            values.flush()
            write_json(output/"PROGRESS.json", dict(split=name, returned=first+len(indices)))
    synchronized(device)
    del values
    return path


def evaluate(run, manifest_path, output, *, device="cpu"):
    run, output = Path(run), Path(output)
    complete = read_json(run/"COMPLETE.json")
    if (not complete["complete"] or complete["progress"]["step"] != 152000
            or sha256(run/"inference.pt") != complete["inference_sha256"]
            or sha256(run/"fixed_H200_VAL.npy") != complete["fixed_H200_VAL_sha256"]
            or sha256(manifest_path) != complete["identity"]["data_manifest_sha256"]
            or source_identity() != complete["identity"]["source_sha256"]):
        raise ValueError("Bind the complete fixed endpoint, original data, source and VAL arrays first")
    runtime_record = complete["identity"]["runtime"]
    torch.set_num_threads(runtime_record["threads"])
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if numerical_runtime(device, runtime_record["threads"]) != runtime_record:
        raise ValueError("Use the declared training runtime and device; a backend change needs a separate protocol")
    if os.environ.get("GENCO_SCATTER_BACKEND") != "pyg_native_sum":
        raise ValueError("The declared scatter backend is required")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = read_json(manifest_path)
    buses = int(manifest["case"][4:])
    runtime, adapter, _ = load_inference(run/"inference.pt", device=device)
    try:
        val = load_pool(manifest_path, "val")
        actual_path = predict_split(runtime, adapter, val, device, output, "VAL_qualification", buses)
        actual = np.load(actual_path, allow_pickle=False)
        reference = np.load(run/"fixed_H200_VAL.npy", allow_pickle=False)
        close = np.isclose(actual, reference, atol=1e-7, rtol=1e-6)
        qualification = dict(
            passed=bool(close.all()), graphs=len(val), atol=1e-7, rtol=1e-6,
            mismatched_elements=int((~close).sum()),
            max_abs_difference=float(np.max(np.abs(actual.astype(np.float64)-reference))),
            reference_sha256=complete["fixed_H200_VAL_sha256"], actual_sha256=sha256(actual_path))
        write_json(output/"QUALIFICATION.json", qualification)
        if not qualification["passed"]:
            raise ValueError("VAL readout qualification failed; no TEST is opened or rescored")
        del val, actual, reference
        test = load_pool(manifest_path, "test")
        predictions = predict_split(runtime, adapter, test, device, output, "TEST", buses)
        write_json(output/"COMPLETE.json", dict(
            complete=True, TEST_graphs=len(test), VAL_qualification_graphs=5400,
            model_sha256=complete["inference_sha256"], TEST_sha256=sha256(predictions),
            original_units=["angle_radians", "magnitude_pu"], numerical_repairs=0,
            input_identity_schema="circuitflow.data.input_identity-v1",
            output_shape=[len(test), buses, 2], no_case_omissions=True,
            central_common_scoring_pending=True, wall_seconds=time.monotonic()-started))
    except Exception as error:
        write_json(output/"FAILED.json", dict(type=type(error).__name__, message=str(error),
                                             actual_attempts="CALL_LEDGER.jsonl", no_seed_replacement=True))
        raise


def pack_results(run, readout, destination):
    run, destination = Path(run), Path(destination)
    readout = Path(readout) if readout is not None else None
    lease = FileLock(str(run/"RUN.lock"), timeout=0)
    with lease:
        return _pack_closed_files(run, readout, destination)


def _pack_closed_files(run, readout, destination):
    if not (run/"RUN_PLAN.json").is_file() and not (run/"SETUP_FAILED.json").is_file():
        raise ValueError("Only an identified scientific run or recorded setup failure may be packaged")
    training_complete = (run/"COMPLETE.json").is_file() and read_json(run/"COMPLETE.json")["complete"]
    readout_complete = bool(readout is not None and (readout/"COMPLETE.json").is_file()
                            and read_json(readout/"COMPLETE.json")["complete"])
    if readout is not None and not readout_complete and not (readout/"FAILED.json").is_file():
        raise ValueError("Do not package a readout while its writer may still be active")
    if training_complete and any(not (run/name).is_file()
                                 for name in ("fixed_H200.pt", "inference.pt", "fixed_H200_VAL.npy")):
        raise ValueError("A claimed complete endpoint is missing required artifacts")
    if readout_complete and not (readout/"TEST.npy").is_file():
        raise ValueError("The complete readout lacks its actual TEST predictions")
    status = "complete" if training_complete and readout_complete else "incomplete_or_failed"
    names = ("RUN_PLAN.json", "COMPLETE.json", "LATEST.json", "VALIDATION.json",
             "fixed_H200.pt", "inference.pt", "best.pt", "fixed_H200_VAL.npy",
             "TRAIN.jsonl", "CALL_LEDGER.jsonl", "CHECKPOINT_WRITES.jsonl",
             "FAILED.json", "SETUP_FAILED.json", "PAUSED.json")
    if not training_complete:
        names += ("latest_a.pt", "latest_b.pt")
    members = [(run/name, "training/"+name) for name in names if (run/name).is_file()]
    members += [(path, "training/"+path.name) for path in sorted(run.glob("ATTEMPT_*.json"))]
    readout_names = ("COMPLETE.json", "FAILED.json", "QUALIFICATION.json", "PROGRESS.json",
                     "CALL_LEDGER.jsonl", "VAL_qualification.npy", "VAL_qualification_INPUTS.jsonl",
                     "TEST.npy", "TEST_INPUTS.jsonl")
    if readout is not None:
        members += [(readout/name, "readout/"+name) for name in readout_names if (readout/name).is_file()]
    inventory = [dict(member=name, bytes=path.stat().st_size, sha256=sha256(path)) for path, name in members]
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path, name in members:
            archive.write(path, name)
        archive.writestr("MANIFEST.json", json.dumps(
            dict(schema="circuitflow-return-v1", status=status, files=inventory), indent=2))
    with zipfile.ZipFile(destination) as archive:
        for item in inventory:
            digest = hashlib.sha256()
            with archive.open(item["member"]) as stream:
                for chunk in iter(lambda: stream.read(4*1024**2), b""):
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError("A returned archive member differs from its recorded source")
    return dict(file=destination.name, bytes=destination.stat().st_size,
                sha256=sha256(destination), members=len(members), status=status,
                raw_datasets_or_credentials_included=False)
