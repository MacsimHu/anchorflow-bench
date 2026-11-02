"""Portable identity checks for the original Case30 Mixed paired controls."""
import hashlib
import json
import random


def schedule(epoch):
    indices = list(range(48600))
    random.Random(202610161509 + epoch).shuffle(indices)
    return [indices[start:start + 64] for start in range(0, len(indices), 64)]


def tensor_identity(state):
    import torch

    result = {}
    for name, value in state.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            result[name] = dict(
                dtype=str(tensor.dtype), shape=list(tensor.shape),
                sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
    return result


def cache_identity(data, slices):
    """Hash ordered tensor contents, independent of the torch.save container."""
    import torch

    from circuitflow.data import prune

    data = prune(data)
    stores = data.to_dict()
    slices = {key: value for key, value in slices.items() if key in stores}
    result = {}

    def visit(value, path):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            result[repr(path)] = dict(
                dtype=str(tensor.dtype), shape=list(tensor.shape),
                sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, (*path, key))
        elif value is None or isinstance(value, (str, bool, int, float)):
            result[repr(path)] = value
        else:
            raise TypeError("Unsupported scientific cache identity field: " + repr(path))

    visit(stores, ("data",))
    visit(slices, ("slices",))
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def authorize(allocation, seed, variant):
    identifier = f"case30-Mixed-{variant}-seed{seed}"
    jobs = [job for job in allocation["jobs"] if job["id"] == identifier]
    if len(jobs) != 1:
        raise ValueError("The exact coordinate is not in the distributed allocation")
    job = jobs[0]
    if job["owner"] != "company-A100" or job["action"] != "train":
        raise ValueError("This coordinate belongs to the original lab queue; do not duplicate it")
    return job
