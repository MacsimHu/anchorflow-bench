"""Frozen train-only signed-percentile power scaling for already-p.u. CANOS data."""
from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json
import math
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class PowerScale:
    Bpu: float
    mode: str
    source_id: str
    indices: tuple = ()
    training_data_sha256: str = ""
    nonzero_count: int = 0
    percentile: float = 95.0
    rule: str = "concatenated_nonzero_signed_Pd_Qd_Pg_Qg"
    generator_layout: str = "explicit_gen"

    def __post_init__(self):
        if isinstance(self.Bpu, bool) or not math.isfinite(self.Bpu) or self.Bpu <= 0:
            raise ValueError("Bpu must be finite and positive")
        if self.mode not in ("train_fitted", "fixed_cpu_canary", "externally_frozen"):
            raise ValueError("Unknown normalization mode")
        if self.generator_layout not in ("explicit_gen", "aggregate_pf"):
            raise ValueError("Unknown normalization generator layout")
        if not self.source_id:
            raise ValueError("A normalization source_id is required")
        if self.mode == "train_fitted" and (
            not self.indices or len(self.training_data_sha256) != 64 or self.nonzero_count <= 0
        ):
            raise ValueError("Fitted scales must bind training indices and statistics")
        object.__setattr__(self, "indices", tuple(self.indices))

    @classmethod
    def fixed_cpu_canary(cls, Bpu=1.0):
        if isinstance(Bpu, bool):
            raise ValueError("Bpu must be a positive real number, not boolean")
        return cls(float(Bpu), "fixed_cpu_canary", "explicit-fixed-scale-not-paper-fit")

    @classmethod
    def externally_frozen(cls, power_scale_pu, source_id="caller-supplied-scale-provenance-held-externally"):
        """An explicit caller-owned scalar, not a claim of local train-only fitting."""
        if isinstance(power_scale_pu, bool):
            raise ValueError("power_scale_pu must be a positive real number, not boolean")
        return cls(float(power_scale_pu), "externally_frozen", source_id)

    @property
    def power_scale_pu(self):
        return self.Bpu

    def to_dict(self):
        payload = asdict(self)
        payload["indices"] = list(self.indices)
        payload["sha256"] = self.fingerprint
        return payload

    @cached_property
    def fingerprint(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        digest = payload.pop("sha256")
        result = cls(**payload)
        if result.fingerprint != digest:
            raise ValueError("Normalization fingerprint mismatch")
        return result


@dataclass(frozen=True)
class TrainingPool:
    """Explicit provenance wrapper; callers must not relabel holdout pools as train."""
    graphs: object
    source_id: str
    split: str = "train"

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def fit_power_scale(train_pool, indices, *, generator_layout="explicit_gen"):
    """Offline only: reproduce upstream np.percentile(nonzero signed values, 95).

    Input powers, shunts and admittances are already on the local physical p.u.
    base, so Bpu is the upstream fitted MVA base divided by that original base.
    No per-sample fit, absolute values, extra /100 or angle conversion is used.
    """
    if getattr(train_pool, "split", None) != "train" or not getattr(train_pool, "source_id", None):
        raise ValueError("fit_power_scale requires an explicitly provenance-bound TRAIN pool")
    indices = tuple(indices)
    if (
        not indices or any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices)
        or len(set(indices)) != len(indices)
        or any(i < 0 or i >= len(train_pool) for i in indices)
    ):
        raise ValueError("Training indices must be unique, nonempty, in-range integers")
    indices = tuple(int(i) for i in indices)
    from .adapter import PFAdapter

    adapter = PFAdapter(PowerScale.fixed_cpu_canary(), dtype=None, generator_layout=generator_layout)
    chunks = []
    digest = hashlib.sha256()
    digest.update(train_pool.source_id.encode())
    digest.update(generator_layout.encode())
    for index in indices:
        graph = train_pool[index]
        targets = adapter.build_targets(graph)
        columns = [targets["bus"][:, 0], targets["bus"][:, 1],
                   targets["gen"][:, 0], targets["bus"][:, 2]]
        digest.update(str(index).encode())
        for column in columns:
            array = column.detach().cpu().double().numpy()
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite training normalization values")
            digest.update(str(array.shape).encode())
            digest.update(array.astype("<f8").tobytes())
            chunks.append(array[array != 0])
    values = np.concatenate(chunks)
    if values.size == 0:
        raise ValueError("Cannot fit a power scale without nonzero training powers")
    value = float(np.percentile(values, 95))
    return PowerScale(value, "train_fitted", train_pool.source_id, indices,
                      digest.hexdigest(), int(values.size), generator_layout=generator_layout)
