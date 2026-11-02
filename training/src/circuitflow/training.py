"""Finite full-TRAIN/full-VAL H200 jobs; no automatic TEST, tuning or successor jobs."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time

from filelock import FileLock
import numpy as np
import psutil
import torch
import torch_geometric
from torch_geometric.data import Batch

from . import checkpoint
from .api import ModelSpec, construct, forward, inference_payload
from .data import COUNTS, load_pool
from .genco.adapter import PFAdapter
from .genco.normalization import PowerScale
from .io import read_json, sha256, write_json


RECIPE = dict(epochs=200, batch_size=64, train_count=48600, val_count=5400,
              schedule_seed=202610161509, learning_rate=0.0005, weight_decay=0.01,
              betas=[0.9, 0.999], eps=1e-8, checkpoint_every_updates=190,
              objective="0.9 masked voltage MSE + 0.1 layered weighted physics norm",
              scheduler="ReduceLROnPlateau(full-VAL native loss, factor=.7, patience=5)",
              selection="fixed complete H200 primary; full-VAL physics secondary only",
              AMP=False, TF32=False, compile=False, early_stopping=False,
              gradient_clipping=None, deterministic_algorithms=True)


class ControlledPause(RuntimeError):
    pass


def permutation(epoch, count=48600):
    if type(epoch) is not int or epoch < 1:
        raise ValueError("Epochs begin at one")
    indices = list(range(count))
    random.Random(RECIPE["schedule_seed"]+epoch).shuffle(indices)
    return indices


def source_identity():
    root = Path(__file__).resolve().parent
    value = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        value.update(path.relative_to(root).as_posix().encode())
        value.update(bytes.fromhex(sha256(path)))
    return value.hexdigest()


def synchronized(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def numerical_runtime(device, threads):
    return dict(
        python=platform.python_version(), torch=str(torch.__version__),
        numpy=np.__version__, torch_geometric=torch_geometric.__version__,
        device=str(device), threads=threads,
        gpu_name=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None)


class Trainer:
    def __init__(self, root, job, manifest_path, device, threads):
        self.root, self.device = Path(root), device
        self.job = job
        manifest_path = Path(manifest_path).resolve()
        manifest = read_json(manifest_path)
        if (manifest["case"] != job["case"] or manifest["task"] != job["task"]
                or any(manifest["splits"][split]["count"] != COUNTS[split] for split in ("train", "val"))):
            raise ValueError("A job cannot change grid, task or full split population")
        self.train = load_pool(manifest_path, "train")
        self.val = load_pool(manifest_path, "val")
        scale_path = manifest_path.parent/manifest["power_scale"]["file"]
        if sha256(scale_path) != manifest["power_scale"]["sha256"]:
            raise ValueError("The frozen TRAIN-only power scale changed")
        self.scale = PowerScale.from_dict(read_json(scale_path))
        if (self.scale.mode != "train_fitted" or len(self.scale.indices) != COUNTS["train"]
                or self.scale.source_id != manifest["splits"]["train"]["sha256"]):
            raise ValueError("Use the declared full-TRAIN scale, not per-sample or TEST normalization")
        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        self.spec = ModelSpec(**job["model_spec"])
        self.runtime = construct(self.spec, device)
        self.adapter = PFAdapter(power_scale_pu=self.scale, generator_layout="aggregate_pf")
        self.optimizer = torch.optim.AdamW(
            self.runtime.model.parameters(), lr=0.0005, betas=(.9, .999), eps=1e-8, weight_decay=.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=.7, patience=5, threshold=1e-4,
            threshold_mode="rel", cooldown=0, min_lr=0., eps=1e-8)
        self.identity = dict(
            job=job, recipe=RECIPE, source_sha256=source_identity(),
            data_manifest_sha256=sha256(manifest_path), power_scale_sha256=self.scale.fingerprint,
            runtime=numerical_runtime(device, threads))
        self.progress = dict(
            phase="initial_val", epoch=1, next_batch=0, step=0, train_graphs=0, val_graphs=0,
            attempted_train_graphs=0, attempted_val_graphs=0,
            val_cursor=0, val_loss_sum=0., val_physics_sum=0., val_invalid=0,
            observations=[], best=None, pending_best=None, train_seconds=0., val_seconds=0.)
        self.order = permutation(1)
        model_bytes = sum(value.numel()*value.element_size() for value in
                          self.runtime.model.state_dict().values() if torch.is_tensor(value))
        self.maximum_state_bytes = 3*model_bytes+16*1024**2
        self.disk_margin = 5*1024**3
        self.last_committed_step = 0

    def guard(self):
        if (self.root/"STOP").exists():
            raise ControlledPause("Operator STOP requested; preserve current complete state")
        elapsed = self.progress["train_seconds"]+self.progress["val_seconds"]
        if elapsed >= self.job["maximum_active_hours"]*3600:
            raise ControlledPause("Declared active-compute budget exhausted; no automatic successor")
        if shutil.disk_usage(self.root).free < self.disk_margin+self.maximum_state_bytes:
            raise ControlledPause("Insufficient reserved checkpoint/write-stop space")
        if psutil.virtual_memory().available < 2*1024**3:
            raise ControlledPause("Host available RAM is below the declared stop margin")

    def current_payload(self):
        return checkpoint.capture(self.runtime, self.optimizer, self.scheduler, self.progress,
                                  self.order, self.identity, self.device)

    def record_attempt(self, split, first, count):
        with (self.root/"CALL_LEDGER.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(
                split=split, epoch=self.progress["epoch"], phase=self.progress["phase"],
                first=first, count=count, committed_step=self.last_committed_step,
                entered_forward=False, status="attempt_recorded_before_forward"))+"\n")

    def save_current(self):
        if shutil.disk_usage(self.root).free < self.disk_margin+self.maximum_state_bytes:
            raise ControlledPause("Insufficient space for another atomic full-state write")
        previous = read_json(self.root/"LATEST.json") if (self.root/"LATEST.json").exists() else {}
        name = "latest_b.pt" if previous.get("file") == "latest_a.pt" else "latest_a.pt"
        started = time.monotonic()
        record = checkpoint.save(self.root/name, self.current_payload(), self.maximum_state_bytes)
        record.update(step=self.progress["step"], phase=self.progress["phase"],
                      epoch=self.progress["epoch"], saved_unix=time.time())
        write_json(self.root/"LATEST.json", record)
        self.last_committed_step = self.progress["step"]
        with (self.root/"CHECKPOINT_WRITES.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(file=name, step=self.last_committed_step,
                                         seconds=time.monotonic()-started, bytes=record["bytes"]))+"\n")

    def resume(self):
        record = read_json(self.root/"LATEST.json")
        path = self.root/record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError("The current full checkpoint differs; never substitute best weights")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.progress, self.order = checkpoint.restore(
            saved, self.runtime, self.optimizer, self.scheduler, self.identity, self.device)
        if self.order != permutation(self.progress["epoch"]):
            raise ValueError("The actual saved epoch permutation differs")
        if self.progress["step"] != record["step"]:
            raise ValueError("The full state and atomic recovery pointer disagree")
        self.last_committed_step = record["step"]

    def train_step(self):
        progress = self.progress
        offset = progress["next_batch"]*64
        indices = self.order[offset:offset+64]
        if not indices:
            raise ValueError("The saved TRAIN cursor is beyond the complete epoch")
        synchronized(self.device)
        started = time.monotonic()
        raw = Batch.from_data_list([self.train.get(index) for index in indices])
        self.runtime.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        progress["attempted_train_graphs"] += len(indices)
        self.record_attempt("train", offset, len(indices))
        _, loss = forward(self.runtime, self.adapter, raw, self.device)
        loss.backward()
        checks = [torch.isfinite(parameter.grad).all() for parameter in
                  self.runtime.model.parameters() if parameter.grad is not None]
        if checks and not torch.stack(checks).all():
            raise FloatingPointError("A nonfinite gradient is a failure, not a skipped example")
        self.optimizer.step()
        synchronized(self.device)
        progress["train_seconds"] += time.monotonic()-started
        progress["step"] += 1
        progress["train_graphs"] += len(indices)
        progress["next_batch"] += 1
        if progress["next_batch"]*64 >= len(self.order):
            progress["phase"] = "val"
        if progress["step"] % 10 == 0:
            with (self.root/"TRAIN.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(
                    step=progress["step"], epoch=progress["epoch"], native_loss=float(loss.detach()),
                    train_graphs=progress["train_graphs"], train_seconds=progress["train_seconds"]))+"\n")

    def validation_step(self):
        progress = self.progress
        first = progress["val_cursor"]
        stop = min(first+64, len(self.val))
        if first >= stop:
            raise ValueError("The saved VAL cursor is outside the complete population")
        synchronized(self.device)
        started = time.monotonic()
        raw = Batch.from_data_list([self.val.get(index) for index in range(first, stop)])
        self.runtime.model.eval()
        progress["attempted_val_graphs"] += stop-first
        self.record_attempt("val", first, stop-first)
        with torch.no_grad():
            outputs, loss = forward(self.runtime, self.adapter, raw, self.device)
        n = int(self.job["case"][4:])
        voltages = outputs["bus"][:, 1].reshape(stop-first, n)
        if progress["phase"] == "val" and progress["epoch"] == 200:
            path = self.root/"fixed_H200_VAL.npy"
            if path.exists():
                predictions = np.load(path, mmap_mode="r+", allow_pickle=False)
                if predictions.shape != (5400, n, 2) or predictions.dtype != np.float32:
                    raise ValueError("The fixed-epoch VAL prediction buffer differs")
            else:
                predictions = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float32, shape=(5400, n, 2))
                predictions[:] = np.nan
            predictions[first:stop] = outputs["bus"].detach().cpu().numpy().reshape(stop-first, n, 2)
            predictions.flush()
            del predictions
        progress["val_invalid"] += int((voltages <= 0).any(dim=1).sum().item())
        physics = float(self.runtime.model.layer_residuals[11].detach())
        progress["val_loss_sum"] += float(loss)*(stop-first)
        progress["val_physics_sum"] += physics*(stop-first)
        progress["val_graphs"] += stop-first
        progress["val_cursor"] = stop
        synchronized(self.device)
        progress["val_seconds"] += time.monotonic()-started
        if stop < len(self.val):
            return False
        initial = progress["phase"] == "initial_val"
        epoch = 0 if initial else progress["epoch"]
        observation = dict(epoch=epoch, count=len(self.val),
                           native_loss=progress["val_loss_sum"]/len(self.val),
                           final_layer_physics=progress["val_physics_sum"]/len(self.val),
                           invalid_count=progress["val_invalid"])
        if not all(math.isfinite(observation[key]) for key in ("native_loss", "final_layer_physics")):
            raise FloatingPointError("A nonfinite full-VAL observation cannot select a model")
        progress["observations"].append(observation)
        selected = not initial and observation["invalid_count"] == 0 and (
            progress["best"] is None or
            observation["final_layer_physics"] < progress["best"]["final_layer_physics"])
        if selected:
            progress["pending_best"] = dict(observation)
        if not initial:
            self.scheduler.step(observation["native_loss"])
        progress.update(val_cursor=0, val_loss_sum=0., val_physics_sum=0., val_invalid=0, next_batch=0)
        if epoch == 200:
            progress["phase"] = "done"
        else:
            progress["phase"] = "train"
            progress["epoch"] = 1 if initial else epoch+1
            self.order = permutation(progress["epoch"])
        write_json(self.root/"VALIDATION.json", progress["observations"])
        return True

    def finish_selection(self):
        proposed = self.progress["pending_best"]
        if proposed is None:
            return
        if shutil.disk_usage(self.root).free < self.disk_margin+self.maximum_state_bytes:
            raise ControlledPause("Insufficient space for a selected-state checkpoint")
        payload = self.current_payload()
        payload["progress"].update(best=proposed, pending_best=None)
        checkpoint.save(self.root/"best.pt", payload, self.maximum_state_bytes)
        self.progress.update(best=proposed, pending_best=None)
        self.save_current()

    def run(self):
        while self.progress["phase"] != "done":
            self.guard()
            self.finish_selection()
            if self.progress["phase"] == "train":
                self.train_step()
                if (self.progress["step"] == 1 or self.progress["step"] % 190 == 0
                        or self.progress["phase"] == "val"):
                    self.save_current()
            else:
                finished = self.validation_step()
                if finished or self.progress["val_cursor"] % (32*64) == 0:
                    self.save_current()
            write_json(self.root/"PROGRESS.json", {
                key: self.progress[key] for key in
                ("phase", "epoch", "next_batch", "step", "train_graphs", "val_graphs",
                 "val_cursor", "train_seconds", "val_seconds")})
        self.finish_selection()
        if (self.progress["step"] != 152000 or self.progress["train_graphs"] != 9720000
                or self.progress["val_graphs"] != 201*5400
                or [row["epoch"] for row in self.progress["observations"]] != list(range(201))):
            raise ValueError("An incomplete trajectory cannot be labeled H200")
        latest = read_json(self.root/"LATEST.json")
        final = self.root/"fixed_H200.pt"
        if final.exists():
            if sha256(final) != latest["sha256"]:
                raise ValueError("The existing fixed endpoint differs from the complete current state")
        else:
            if shutil.disk_usage(self.root).free < self.disk_margin+self.maximum_state_bytes:
                raise ControlledPause("Insufficient space to publish the fixed endpoint")
            pending = final.with_name(final.name+".pending")
            with (self.root/latest["file"]).open("rb") as source, pending.open("xb") as target:
                shutil.copyfileobj(source, target, 4*1024**2)
                target.flush()
                os.fsync(target.fileno())
            if sha256(pending) != latest["sha256"]:
                raise ValueError("The fixed endpoint copy differs")
            os.replace(pending, final)
        export = inference_payload(
            self.runtime, self.spec, self.scale,
            dict(job=self.job, data_manifest_sha256=self.identity["data_manifest_sha256"],
                 endpoint="fixed_H200", updates=152000, from_scratch=True,
                 pretrained_case30_example_not_used_for_training=True))
        if shutil.disk_usage(self.root).free < self.disk_margin+self.maximum_state_bytes:
            raise ControlledPause("Insufficient space for the inference export")
        checkpoint.save(self.root/"inference.pt", export, self.maximum_state_bytes)
        write_json(self.root/"COMPLETE.json", dict(
            complete=True, fixed_H200_sha256=sha256(final), inference_sha256=sha256(self.root/"inference.pt"),
            fixed_H200_VAL_sha256=sha256(self.root/"fixed_H200_VAL.npy"),
            identity=self.identity, progress=self.progress, TEST_calls=0, numerical_repairs=0))


def train_job(job, manifest_path, root, *, resume=False, device="cpu", threads=1):
    if (platform.python_version_tuple()[:2] != ("3", "10")
            or torch.__version__.split("+")[0] != "2.1.2"
            or np.__version__ != "1.26.4" or torch_geometric.__version__ != "2.6.1"):
        raise RuntimeError("Formal jobs require the published reproduction environment")
    if os.environ.get("GENCO_SCATTER_BACKEND") != "pyg_native_sum":
        raise RuntimeError("Set GENCO_SCATTER_BACKEND=pyg_native_sum before importing the model")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    root = Path(root).resolve()
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise FileNotFoundError("A continuation requires its existing run directory")
    with FileLock(str(root/"RUN.lock"), timeout=0):
        if (root/"COMPLETE.json").exists():
            raise FileExistsError("This scientific job is complete; do not launch it again")
        if (root/"FAILED.json").exists():
            raise RuntimeError("Preserve and review the failed scope; do not silently replay a failure")
        attempt = len(list(root.glob("ATTEMPT_*_START.json")))+1
        started = time.monotonic()
        write_json(root/f"ATTEMPT_{attempt:04d}_START.json", dict(
            unix=time.time(), resume=resume, job_id=job["id"], PID=os.getpid()))
        try:
            trainer = Trainer(root, job, manifest_path, device, threads)
        except Exception as error:
            write_json(root/"SETUP_FAILED.json", dict(
                type=type(error).__name__, message=str(error), optimizer_updates=0,
                TEST_calls=0, attempt_wall_seconds=time.monotonic()-started))
            raise
        if resume:
            trainer.resume()
        else:
            write_json(root/"RUN_PLAN.json", trainer.identity)
            trainer.save_current()
        try:
            trainer.run()
        except ControlledPause as error:
            can_save = shutil.disk_usage(root).free >= trainer.disk_margin+trainer.maximum_state_bytes
            if can_save:
                trainer.save_current()
            record = dict(
                reason=str(error), current_step=trainer.progress["step"],
                committed_step=trainer.last_committed_step, current_full_state_saved=can_save,
                uncommitted_updates=trainer.progress["step"]-trainer.last_committed_step,
                attempted_train_graphs=trainer.progress["attempted_train_graphs"],
                attempted_val_graphs=trainer.progress["attempted_val_graphs"],
                attempt_wall_seconds=time.monotonic()-started)
            write_json(root/f"ATTEMPT_{attempt:04d}_PAUSED.json", record)
            write_json(root/"PAUSED.json", record)
            return 75
        except Exception as error:
            write_json(root/"FAILED.json", dict(
                type=type(error).__name__, message=str(error),
                current_step=trainer.progress["step"], committed_step=trainer.last_committed_step,
                attempted_train_graphs=trainer.progress["attempted_train_graphs"],
                attempted_val_graphs=trainer.progress["attempted_val_graphs"],
                attempt_wall_seconds=time.monotonic()-started, seed_not_replaced=True, TEST_calls=0))
            raise
    return 0
