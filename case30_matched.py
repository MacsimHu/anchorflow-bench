"""Run only the three assigned Unit controls against the retained Mixed Full."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from case30_contract import authorize, cache_identity, tensor_identity


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "case30_reference" / "FULL_REFERENCE.json"
ALLOCATION = ROOT / "CASE30_ALLOCATION.json"


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trainer_type(*, variant="Unit", reference_path=REFERENCE, allocation_path=ALLOCATION):
    import random
    import shutil
    import time

    import numpy as np
    import torch
    from torch_geometric.data import Batch

    from circuitflow import checkpoint
    from circuitflow.api import ModelSpec, construct, forward, inference_payload
    from circuitflow.data import load_pool
    from circuitflow.genco.adapter import PFAdapter
    from circuitflow.genco.normalization import PowerScale
    from circuitflow.io import write_json
    from circuitflow.training import Trainer, numerical_runtime, permutation, source_identity, synchronized
    weighting = {"Unit": "unit", "Knockout": "knockout", "MeanToken": "mean_token"}[variant]

    class MatchedUnit(Trainer):
        def __init__(self, root, job, manifest_path, device, threads):
            reference = read(reference_path)
            self.root, self.device, self.job = Path(root), device, job
            if (job["case"] != "case30" or job["task"] != reference["task"]
                    or job["model_spec"] != dict(reference["model_spec"], coarse_weighting=weighting)
                    or type(job["seed"]) is not int or job["seed"] not in (0, 1, 2)):
                raise ValueError("The assigned job differs from the original three paired Unit controls")
            manifest_path = Path(manifest_path).resolve()
            manifest = read(manifest_path)
            if manifest["case"] != "case30" or manifest["task"] != reference["task"]:
                raise ValueError("The manifest task differs from the original Full reference")
            self.train = load_pool(manifest_path, "train")
            original_val = load_pool(manifest_path, "val")
            for name, pool in (("train", self.train), ("val", original_val)):
                expected = reference["data"][name]
                if (len(pool) != expected["count"]
                        or cache_identity(pool._data, pool.slices) != expected["semantic_sha256"]):
                    raise ValueError("The original ordered graph/label population differs: " + name)
            indices = reference["evaluation_indices"]
            if reference["task"] == "1.3":
                if (len(indices) != 450 or len(set(indices)) != 450
                        or any(sum(index // 1800 == cohort for index in indices) != 150
                               for cohort in range(3))):
                    raise ValueError("The original stratified DEV450 membership changed")
            elif reference["task"] != "1.1" or indices != list(range(5400)):
                raise ValueError("The original Intact full-VAL5400 membership changed")
            self.validation_population = "original_DEV450" if len(indices) == 450 else "original_VAL5400"
            self.val = [original_val.get(index) for index in indices]
            self.scale = PowerScale.from_dict(reference["power_scale"])
            self.spec = ModelSpec(**job["model_spec"])
            seed = job["seed"]
            initial_info = reference["initials"][str(seed)]
            initial_path = reference_path.parent / initial_info["file"]
            if digest(initial_path) != initial_info["sha256"]:
                raise ValueError("A published untrained Full initialization changed")
            initial = torch.load(initial_path, map_location="cpu", weights_only=True)
            if (initial["schema"] != "anchorflow-public-untrained-initial-v1"
                    or initial["seed"] != seed
                    or tensor_identity(initial["model"]) != initial_info["tensor_identity"]):
                raise ValueError("The original seed initialization identity differs")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.runtime = construct(self.spec, "cpu")
            configuration = self.runtime.model.get_extra_state()
            expected_configuration = dict(initial["model"]["_extra_state"], coarse_weighting=weighting)
            if configuration != expected_configuration:
                raise ValueError("The architecture differs from the declared single control")
            state = dict(initial["model"], _extra_state=configuration)
            self.runtime.model.load_state_dict(state, strict=True)
            if tensor_identity(self.runtime.model.state_dict()) != initial_info["tensor_identity"]:
                raise ValueError("Loading changed an original Full initial tensor")
            self.runtime.model.to(device)
            random.setstate(initial["python"])
            numpy_state = initial["numpy"]
            np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32),
                                 *numpy_state[2:]))
            torch.set_rng_state(initial["torch"])
            if torch.device(device).type == "cuda":
                if initial.get("cuda"):
                    if len(initial["cuda"]) != torch.cuda.device_count():
                        raise ValueError("The original CUDA RNG requires the same visible device count")
                    torch.cuda.set_rng_state_all(initial["cuda"])
                else:
                    torch.cuda.manual_seed(seed)
            self.adapter = PFAdapter(power_scale_pu=self.scale, generator_layout="aggregate_pf")
            self.optimizer = torch.optim.AdamW(
                self.runtime.model.parameters(), lr=.0005, betas=(.9, .999),
                eps=1e-8, weight_decay=.01)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=.7, patience=5, threshold=1e-4,
                threshold_mode="rel", cooldown=0, min_lr=0., eps=1e-8)
            self.identity = dict(
                job=job, recipe=reference["original_recipe"], source_sha256=source_identity(),
                runner_sha256=digest(Path(__file__)), contract_helper_sha256=digest(ROOT / "case30_contract.py"),
                full_reference_sha256=digest(reference_path), allocation_sha256=digest(allocation_path),
                data_manifest_sha256=digest(manifest_path), power_scale_sha256=self.scale.fingerprint,
                runtime=numerical_runtime(device, threads),
                scheduler_monitor=reference["scheduler_monitor"],
                original_initial_sha256=initial_info["original_state_sha256"],
                backend_comparability=reference["company_backend"])
            self.progress = dict(
                phase="initial_val", epoch=1, next_batch=0, step=0, train_graphs=0, val_graphs=0,
                attempted_train_graphs=0, attempted_val_graphs=0, val_cursor=0, val_loss_sum=0.,
                val_physics_sum=0., val_invalid=0, observations=[], best=None, pending_best=None,
                train_seconds=0., val_seconds=0.)
            self.order = permutation(1)
            model_bytes = sum(value.numel() * value.element_size()
                              for value in self.runtime.model.state_dict().values()
                              if torch.is_tensor(value))
            self.maximum_state_bytes = 3 * model_bytes + 16 * 1024**2
            self.disk_margin = 5 * 1024**3
            self.last_committed_step = 0

        @property
        def validation_count(self):
            return len(self.val)

        def validation_step(self):
            progress = self.progress
            first = progress["val_cursor"]
            stop = min(first + 64, self.validation_count)
            if not 0 <= first < stop:
                raise ValueError("The DEV450 recovery cursor is outside its original population")
            synchronized(self.device)
            started = time.monotonic()
            raw = Batch.from_data_list(self.val[first:stop])
            self.runtime.model.eval()
            progress["attempted_val_graphs"] += stop - first
            self.record_attempt("original_DEV450", first, stop - first)
            with torch.no_grad():
                outputs, loss = forward(self.runtime, self.adapter, raw, self.device)
            if (outputs["bus"][:, 1] <= 0).any():
                raise FloatingPointError("A nonpositive voltage is retained as a failed control")
            current_prediction = self.root / "DEV_CURRENT.npy"
            if first == 0:
                predictions = np.lib.format.open_memmap(
                    current_prediction, mode="w+", dtype=np.float32, shape=(self.validation_count, 30, 2))
                predictions[:] = np.nan
            else:
                predictions = np.load(current_prediction, mmap_mode="r+", allow_pickle=False)
                if (predictions.shape != (self.validation_count, 30, 2) or predictions.dtype != np.float32
                        or not np.isfinite(predictions[:first]).all()):
                    raise ValueError("The retained DEV prefix differs from its recovery cursor")
            predictions[first:stop] = outputs["bus"].detach().cpu().numpy().reshape(stop-first, 30, 2)
            predictions.flush()
            del predictions
            physics = float(self.runtime.model.layer_residuals[11].detach())
            if not np.isfinite(physics):
                raise FloatingPointError("The DEV450 physics score is nonfinite")
            progress["val_loss_sum"] += float(loss) * (stop - first)
            progress["val_physics_sum"] += physics * (stop - first)
            progress["val_graphs"] += stop - first
            progress["val_cursor"] = stop
            synchronized(self.device)
            progress["val_seconds"] += time.monotonic() - started
            if stop < self.validation_count:
                return False
            initial = progress["phase"] == "initial_val"
            epoch = 0 if initial else progress["epoch"]
            observation = dict(epoch=epoch, count=self.validation_count,
                               native_loss=progress["val_loss_sum"] / self.validation_count,
                               final_layer_physics=progress["val_physics_sum"] / self.validation_count,
                               invalid_count=0)

            def retain_prediction(name):
                target = self.root / name
                pending = target.with_name(target.name + ".pending")
                with current_prediction.open("rb") as source, pending.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                os.replace(pending, target)
                return dict(file=name, sha256=digest(target))

            if epoch in (0, 200):
                observation["fixed_prediction"] = retain_prediction(f"dev_{epoch:04d}.npy")
            progress["observations"].append(observation)
            if not initial:
                if (progress["best"] is None
                        or observation["final_layer_physics"] < progress["best"]["final_layer_physics"]):
                    previous = (progress["best"] or {}).get("selected_prediction", {}).get("file")
                    name = "selected_dev_b.npy" if previous == "selected_dev_a.npy" else "selected_dev_a.npy"
                    progress["pending_best"] = dict(
                        observation, selected_prediction=retain_prediction(name))
                self.scheduler.step(observation["native_loss"])
            progress.update(val_cursor=0, val_loss_sum=0., val_physics_sum=0.,
                            val_invalid=0, next_batch=0)
            if epoch == 200:
                progress["phase"] = "done"
            else:
                progress["phase"] = "train"
                progress["epoch"] = 1 if initial else epoch + 1
                self.order = permutation(progress["epoch"])
            write_json(self.root / "VALIDATION.json", progress["observations"])
            return True

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
                    if self.validation_step():
                        self.save_current()
                write_json(self.root / "PROGRESS.json", self.progress)
            self.finish_selection()
            progress = self.progress
            if (progress["step"] != 152000 or progress["train_graphs"] != 9720000
                    or progress["val_graphs"] != 201 * self.validation_count
                    or [row["epoch"] for row in progress["observations"]] != list(range(201))):
                raise ValueError("The paired control has not completed its original fixed H200")
            latest = read(self.root / "LATEST.json")
            current = self.root / latest["file"]
            if digest(current) != latest["sha256"]:
                raise ValueError("The final full state differs from its recovery pointer")
            final = self.root / "fixed_H200.pt"
            if final.exists():
                if digest(final) != latest["sha256"]:
                    raise FileExistsError("Never overwrite a different fixed endpoint")
            else:
                if shutil.disk_usage(self.root).free < self.disk_margin + self.maximum_state_bytes:
                    raise RuntimeError("Insufficient space to retain the fixed full state")
                pending = final.with_name(final.name + ".pending")
                with current.open("rb") as source, pending.open("xb") as target:
                    shutil.copyfileobj(source, target, 4 * 1024**2)
                    target.flush()
                    os.fsync(target.fileno())
                if digest(pending) != latest["sha256"]:
                    raise ValueError("The fixed endpoint copy differs")
                os.replace(pending, final)
            payload = inference_payload(self.runtime, self.spec, self.scale, self.identity)
            checkpoint.save(self.root / "inference.pt", payload, self.maximum_state_bytes)
            write_json(self.root / "COMPLETE.json", dict(
                complete=True, identity=self.identity, progress=progress,
                fixed_H200_sha256=digest(final), inference_sha256=digest(self.root / "inference.pt"),
                validation_population=self.validation_population, TEST_calls=0,
                backend_matched_to_original_CPU=False))

    return MatchedUnit


def main(argv=None, *, allocation_path=ALLOCATION, reference_path=REFERENCE, variant="Unit"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allocation-checked", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args(argv)
    allocation = read(allocation_path)
    seeds = [args.seed] if args.seed is not None else [0, 1, 2]
    reference = read(reference_path)
    regime = "Intact" if reference["task"] == "1.1" else "Mixed"
    jobs = [authorize(allocation, seed, variant, regime) for seed in seeds]
    if not args.execute:
        print(json.dumps(dict(jobs=jobs, execution_started=False), indent=2))
        return 0
    if (args.manifest is None or args.workspace is None or not args.allocation_checked
            or args.threads < 1):
        parser.error("Execution requires the existing manifest, owned workspace and allocation check")
    workspace = args.workspace.resolve()
    if workspace == ROOT or workspace.is_relative_to(ROOT):
        parser.error("Keep current states outside the immutable source checkout")
    sys.path.insert(0, str(ROOT / "training" / "src"))
    from circuitflow.cli import numerical_environment
    numerical_environment("cuda", args.gpu)
    import platform
    import numpy as np
    import torch
    import torch_geometric
    from filelock import FileLock
    from circuitflow.io import write_json
    from circuitflow.training import ControlledPause
    if (platform.python_version_tuple()[:2] != ("3", "10")
            or torch.__version__.split("+")[0] != "2.1.2" or np.__version__ != "1.26.4"
            or torch_geometric.__version__ != "2.6.1"):
        raise RuntimeError("Reuse the published isolated 3.10/Torch2.1.2/NumPy1.26.4/PyG2.6.1 environment")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    workspace.mkdir(parents=True, exist_ok=True)
    trainer_class = trainer_type(
        variant=variant, reference_path=reference_path, allocation_path=allocation_path)
    lock_name = "CASE30_UNIT_COMPANY.lock" if variant == "Unit" else f"CASE30_{variant}_COMPANY.lock"
    with FileLock(str(workspace / lock_name), timeout=0):
        for job in jobs:
            folder = workspace / job["id"]
            if (folder / "COMPLETE.json").exists():
                complete = read(folder / "COMPLETE.json")
                if (complete.get("complete") is not True
                        or complete["identity"]["job"] != job
                        or digest(folder / "fixed_H200.pt") != complete["fixed_H200_sha256"]
                        or digest(folder / "inference.pt") != complete["inference_sha256"]):
                    raise ValueError("A completed-coordinate receipt or retained endpoint differs")
                print("REUSED_COMPLETE " + job["id"], flush=True)
                continue
            existed = folder.exists()
            if existed and not args.resume_existing:
                raise FileExistsError("Continue only the identified current full state with --resume-existing")
            folder.mkdir(exist_ok=True)
            if (folder / "FAILED.json").exists():
                raise RuntimeError("Preserve and diagnose the actual failure; never silently restart it")
            with FileLock(str(folder / "RUN.lock"), timeout=0):
                trainer = trainer_class(folder, job, args.manifest, "cuda", args.threads)
                if existed:
                    trainer.resume()
                else:
                    write_json(folder / "RUN_PLAN.json", trainer.identity)
                    trainer.save_current()
                write_json(folder / "OWNER.local.json", dict(
                    PID=os.getpid(), birth_unix=__import__("psutil").Process().create_time(),
                    job_id=job["id"], full_reference_sha256=digest(reference_path)))
                try:
                    trainer.run()
                except ControlledPause as error:
                    import shutil
                    can_save = shutil.disk_usage(folder).free >= trainer.disk_margin + trainer.maximum_state_bytes
                    if can_save:
                        trainer.save_current()
                    write_json(folder / "PAUSED.json", dict(
                        reason=str(error), step=trainer.progress["step"],
                        committed_step=trainer.last_committed_step,
                        uncommitted_updates=trainer.progress["step"] - trainer.last_committed_step,
                        current_full_state_saved=can_save, TEST_calls=0))
                    return 75
                except Exception as error:
                    write_json(folder / "FAILED.json", dict(
                        type=type(error).__name__, message=str(error),
                        returned_step=trainer.progress["step"],
                        committed_step=trainer.last_committed_step, TEST_calls=0))
                    raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
