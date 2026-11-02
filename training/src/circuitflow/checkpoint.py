"""Tensor-only full-state checkpoints, including the actual sampler and random states."""
import copy
import os
from pathlib import Path
import random

import numpy as np
import torch

from .io import sha256


def capture(runtime, optimizer, scheduler, progress, permutation, identity, device):
    numpy_state = np.random.get_state()
    rng = dict(python=random.getstate(), torch=torch.get_rng_state(),
               numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
               cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None)
    return dict(schema="circuitflow-training-v1", identity=identity,
                model=runtime.model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), progress=copy.deepcopy(progress),
                permutation=list(permutation), rng=rng, training=runtime.model.training)


def restore(saved, runtime, optimizer, scheduler, identity, device):
    if saved.get("schema") != "circuitflow-training-v1" or saved["identity"] != identity:
        raise ValueError("The checkpoint belongs to a different frozen job/data/source/runtime")
    runtime.model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    runtime.model.train(saved["training"])
    rng = saved["rng"]
    random.setstate(rng["python"])
    state = rng["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(rng["torch"])
    if torch.device(device).type == "cuda":
        if rng["cuda"] is None:
            raise ValueError("This CUDA continuation has no captured device RNG")
        torch.cuda.set_rng_state(rng["cuda"], device)
    elif rng["cuda"] is not None:
        raise ValueError("A CUDA trajectory cannot silently become a CPU replay")
    return copy.deepcopy(saved["progress"]), list(saved["permutation"])


def save(path, value, maximum_bytes):
    path = Path(path)
    pending = path.with_name(path.name+".pending")
    with pending.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if pending.stat().st_size > maximum_bytes:
        raise OSError("The measured checkpoint exceeds its reserved envelope; prior state retained")
    os.replace(pending, path)
    return dict(file=path.name, sha256=sha256(path), bytes=path.stat().st_size)
